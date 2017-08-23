--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Generic model creating code. For the specific ResNet model see
--  models/resnet.lua
--
-- Code modified for DenseNet (https://arxiv.org/abs/1608.06993) by Gao Huang.
-- 
-- More details about the memory efficient implementation can be found in the 
-- technique report "Memory-Efficient Implementation of DenseNets" 
-- (https://arxiv.org/pdf/1707.06990.pdf)

require 'nn'
require 'cunn'
require 'cudnn'
require 'models/DenseConnectLayer'

local M = {}

function M.setup(opt, checkpoint)

   print('=> Creating model from file: models/' .. opt.netType .. '.lua')
   local model = require('models/' .. opt.netType)(opt)
   if checkpoint then
      local modelPath = paths.concat(opt.resume, checkpoint.modelFile)
      assert(paths.filep(modelPath), 'Saved model not found: ' .. modelPath)
      print('=> Resuming model from ' .. modelPath)
      local model0 = torch.load(modelPath):type(opt.tensorType)
      M.copyModel(model, model0)
   elseif opt.retrain ~= 'none' then
      assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
      print('Loading model from file: ' .. opt.retrain)
      local model0 = torch.load(opt.retrain):type(opt.tensorType)
      M.copyModel(model, model0)
   end

   -- First remove any DataParallelTable
   if torch.type(model) == 'nn.DataParallelTable' then
      model = model:get(1)
   end

   -- optnet is an general library for reducing memory usage in neural networks
   if opt.optnet or opt.optMemory == 1 then
      local optnet = require 'optnet'
      local imsize = opt.dataset == 'imagenet' and 224 or 32
      local sampleInput = torch.zeros(4,3,imsize,imsize):type(opt.tensorType)
      optnet.optimizeMemory(model, sampleInput, {inplace = false, mode = 'training'})
   end

   -- This is useful for fitting ResNet-50 on 4 GPUs, but requires that all
   -- containers override backwards to call backwards recursively on submodules
   if opt.shareGradInput or opt.optMemory >= 2 then
      M.shareGradInput(model, opt)
      M.sharePrevOutput(model, opt)
   end

   -- Share the contiguous (concatenated) outputs of previous layers in DenseNet.
   if opt.optMemory == 3 then
      M.sharePrevOutput(model, opt)
   end

   -- Share the output of BatchNorm in bottleneck layers of DenseNet. This requires
   -- forwarding the BN layer twice at each mini-batch, but makes the memory consumption  
   -- linear (instead of quadratic) in depth
   if opt.optMemory == 4 then
      M.shareBNOutput(model, opt)
   end

   -- For resetting the classifier when fine-tuning on a different Dataset
   if opt.resetClassifier and not checkpoint then
      print(' => Replacing classifier with ' .. opt.nClasses .. '-way classifier')

      local orig = model:get(#model.modules)
      assert(torch.type(orig) == 'nn.Linear',
         'expected last layer to be fully connected')

      local linear = nn.Linear(orig.weight:size(2), opt.nClasses)
      linear.bias:zero()

      model:remove(#model.modules)
      model:add(linear:type(opt.tensorType))
   end

   -- Set the CUDNN flags
   if opt.cudnn == 'fastest' then
      cudnn.fastest = true
      cudnn.benchmark = true
   elseif opt.cudnn == 'deterministic' then
      -- Use a deterministic convolution implementation
      model:apply(function(m)
         if m.setMode then m:setMode(1, 1, 1) end
      end)
   end

   -- Wrap the model with DataParallelTable, if using more than one GPU
   if opt.nGPU > 1 then
      local gpus = torch.range(1, opt.nGPU):totable()
      local fastest, benchmark = cudnn.fastest, cudnn.benchmark

      local dpt = nn.DataParallelTable(1, true, true)
         :add(model, gpus)
         :threads(function()
            local cudnn = require 'cudnn'
            require 'models/DenseConnectLayer'
            cudnn.fastest, cudnn.benchmark = fastest, benchmark
         end)
      dpt.gradInput = nil

      model = dpt:type(opt.tensorType)
   end

   local criterion = nn.CrossEntropyCriterion():type(opt.tensorType)
   return model, criterion
end

function M.shareGradInput(model, opt)
   local function sharingKey(m)
      local key = torch.type(m)
      if m.__shareGradInputKey then
         key = key .. ':' .. m.__shareGradInputKey
      end
      return key
   end

   -- Share gradInput for memory efficient backprop
   local cache = {}
   model:apply(function(m)
      local moduleType = torch.type(m)
      if torch.isTensor(m.gradInput) and moduleType ~= 'nn.ConcatTable' and moduleType ~= 'nn.Concat' then
         local key = sharingKey(m)
         if cache[key] == nil then
            cache[key] = torch[opt.tensorType:match('torch.(%a+)'):gsub('Tensor','Storage')](1)
         end
         m.gradInput = torch[opt.tensorType:match('torch.(%a+)')](cache[key], 1, 0)
      end
   end)
   for i, m in ipairs(model:findModules('nn.ConcatTable')) do
      if cache[i % 2] == nil then
         cache[i % 2] = torch[opt.tensorType:match('torch.(%a+)'):gsub('Tensor','Storage')](1)
      end
      m.gradInput = torch[opt.tensorType:match('torch.(%a+)')](cache[i % 2], 1, 0)
   end
   for i, m in ipairs(model:findModules('nn.Concat')) do
      if cache[i % 2] == nil then
         cache[i % 2] = torch[opt.tensorType:match('torch.(%a+)'):gsub('Tensor','Storage')](1)
      end
      m.gradInput = torch[opt.tensorType:match('torch.(%a+)')](cache[i % 2], 1, 0)
   end
end

function M.sharePrevOutput(model, opt)
   -- Share contiguous output for memory efficient densenet
   local buffer = nil
   model:apply(function(m)
      local moduleType = torch.type(m)
      if moduleType == 'nn.DenseConnectLayerCustom' then
         if buffer == nil then
            buffer = torch[opt.tensorType:match('torch.(%a+)'):gsub('Tensor','Storage')](1)
         end
         m.input_c = torch[opt.tensorType:match('torch.(%a+)')](buffer, 1, 0)
      end
   end)
end

function M.shareBNOutput(model, opt)
   -- Share BN.output for memory efficient densenet
   local buffer = nil
   model:apply(function(m)
      local moduleType = torch.type(m)
      if moduleType == 'nn.DenseConnectLayerCustom' then
         if buffer == nil then
            buffer = torch[opt.tensorType:match('torch.(%a+)'):gsub('Tensor','Storage')](1)
         end
         m.net1:get(1).output = torch[opt.tensorType:match('torch.(%a+)')](buffer, 1, 0)
      end
   end)
end

function M.copyModel(t, s)
   local wt, ws = t:parameters(), s:parameters()
   assert(#wt==#ws, 'Model configurations does not match the resumed model!')
   for l = 1, #wt do
      wt[l]:copy(ws[l])
   end
   local bn_t, bn_s = {}, {}
   for i, m in ipairs(s:findModules('cudnn.SpatialBatchNormalization')) do
      bn_s[i] = m
   end
   for i, m in ipairs(t:findModules('cudnn.SpatialBatchNormalization')) do
      bn_t[i] = m
   end
   assert(#bn_t==#bn_s, 'Model configurations does not match the resumed model!')
   for i = 1, #bn_s do
      bn_t[i].running_mean:copy(bn_s[i].running_mean)
      bn_t[i].running_var:copy(bn_s[i].running_var) 
   end
end

return M
