require 'nn'
require 'cunn'
require 'cudnn'
require 'models/DenseConnectLayer'

local function createModel(opt)

   --growth rate
   local growthRate = opt.growthRate

   --dropout rate, set it to 0 to disable dropout, non-zero number to enable dropout and set drop rate
   local dropRate = opt.dropRate

   --# channels before entering the first Dense-Block
   local nChannels = 2 * growthRate

   --compression rate at transition layers
   local reduction = opt.reduction

   --whether to use bottleneck structures
   local bottleneck = opt.bottleneck

   --N: # dense connected layers in each denseblock
   local N = (opt.depth - 4)/3
   if bottleneck then N = N/2 end


   function addLayer(model, nChannels, opt)
      if opt.optMemory >= 3 then
         model:add(nn.DenseConnectLayerCustom(nChannels, opt))
      else
         model:add(DenseConnectLayerStandard(nChannels, opt))     
      end
   end


   function addTransition(model, nChannels, nOutChannels, opt, last, pool_size)
      if opt.optMemory >= 3 then     
         model:add(nn.JoinTable(2))
      end

      model:add(cudnn.SpatialBatchNormalization(nChannels))
      model:add(cudnn.ReLU(true))      
      if last then
         model:add(cudnn.SpatialAveragePooling(pool_size, pool_size))
         model:add(nn.Reshape(nChannels))      
      else
         model:add(cudnn.SpatialConvolution(nChannels, nOutChannels, 1, 1, 1, 1, 0, 0))
         if opt.dropRate > 0 then model:add(nn.Dropout(opt.dropRate)) end
         model:add(cudnn.SpatialAveragePooling(2, 2))
      end      
   end


   local function addDenseBlock(model, nChannels, opt, N)
      for i = 1, N do 
         addLayer(model, nChannels, opt)
         nChannels = nChannels + opt.growthRate
      end
      return nChannels
   end


   -- Build DenseNet
   local model = nn.Sequential()

   if opt.dataset == 'cifar10' or opt.dataset == 'cifar100' then

      --Initial convolution layer
      model:add(cudnn.SpatialConvolution(3, nChannels, 3,3, 1,1, 1,1))      

      --Dense-Block 1 and transition
      nChannels = addDenseBlock(model, nChannels, opt, N)
      addTransition(model, nChannels, math.floor(nChannels*reduction), opt)
      nChannels = math.floor(nChannels*reduction)

      --Dense-Block 2 and transition
      nChannels = addDenseBlock(model, nChannels, opt, N)
      addTransition(model, nChannels, math.floor(nChannels*reduction), opt)
      nChannels = math.floor(nChannels*reduction)

      --Dense-Block 3 and transition
      nChannels = addDenseBlock(model, nChannels, opt, N)
      addTransition(model, nChannels, nChannels, opt, true, 8)

   elseif opt.dataset == 'imagenet' then

      --number of layers in each block
      if opt.depth == 121 then
         stages = {6, 12, 24, 16}
      elseif opt.depth == 169 then
         stages = {6, 12, 32, 32}
      elseif opt.depth == 201 then
         stages = {6, 12, 48, 32}
      elseif opt.depth == 161 then
         stages = {6, 12, 36, 24}
      else
         stages = {opt.d1, opt.d2, opt.d3, opt.d4}
      end

      --Initial transforms follow ResNet(224x224)
      model:add(cudnn.SpatialConvolution(3, nChannels, 7,7, 2,2, 3,3))
      model:add(cudnn.SpatialBatchNormalization(nChannels))
      model:add(cudnn.ReLU(true))
      model:add(nn.SpatialMaxPooling(3, 3, 2, 2, 1, 1))

      --Dense-Block 1 and transition (56x56)
      nChannels = addDenseBlock(model, nChannels, opt, stages[1])
      addTransition(model, nChannels, math.floor(nChannels*reduction), opt)
      nChannels = math.floor(nChannels*reduction)

      --Dense-Block 2 and transition (28x28)
      nChannels = addDenseBlock(model, nChannels, opt, stages[2])
      addTransition(model, nChannels, math.floor(nChannels*reduction), opt)
      nChannels = math.floor(nChannels*reduction)

      --Dense-Block 3 and transition (14x14)
      nChannels = addDenseBlock(model, nChannels, opt, stages[3])
      addTransition(model, nChannels, math.floor(nChannels*reduction), opt)
      nChannels = math.floor(nChannels*reduction)

      --Dense-Block 4 and transition (7x7)
      nChannels = addDenseBlock(model, nChannels, opt, stages[4])
      addTransition(model, nChannels, nChannels, opt, true, 7)

   end


   if opt.dataset == 'cifar10' then
      model:add(nn.Linear(nChannels, 10))
   elseif opt.dataset == 'cifar100' then
      model:add(nn.Linear(nChannels, 100))
   elseif opt.dataset == 'imagenet' then
      model:add(nn.Linear(nChannels, 1000))
   end

   --Initialization following ResNet
   local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
   end

   local function BNInit(name)
      for k,v in pairs(model:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
   end

   ConvInit('cudnn.SpatialConvolution')
   BNInit('cudnn.SpatialBatchNormalization')
   for k,v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
   end

   model:type(opt.tensorType)

   if opt.cudnn == 'deterministic' then
      model:apply(function(m)
         if m.setMode then m:setMode(1,1,1) end
      end)
   end

   model:get(1).gradInput = nil

   print(model)
   return model
end

return createModel