require 'nn'
require 'cunn'
require 'cudnn'

local function createModel(opt)
    if (opt.depth - 4 ) % 3 ~= 0 then
      error("Depth must be 3N + 4!")
    end

    --#layers in each denseblock
    local N = (opt.depth - 4)/3

    --growth rate
    local growthRate = 12

    --dropout rate, set it to nil to disable dropout, non-zero number to enable dropout and set drop rate
    local dropRate = nil

    --#channels before entering the first denseblock
    --set it to be comparable with growth rate
    local nChannels = 16

    local function addLayer(model, nChannels, nOutChannels, dropRate)
      concate = nn.Concat(2)
      concate:add(nn.Identity())

      convFactory = nn.Sequential()
      convFactory:add(cudnn.SpatialBatchNormalization(nChannels))
      convFactory:add(cudnn.ReLU(true))
      convFactory:add(cudnn.SpatialConvolution(nChannels, nOutChannels, 3, 3, 1, 1, 1,1))
      if dropRate then
        convFactory:add(nn.Dropout(dropRate))
      end
      concate:add(convFactory)
      model:add(concate)
    end

    local function addTransition(model, nChannels, nOutChannels, dropRate)
      model:add(cudnn.SpatialBatchNormalization(nChannels))
      model:add(cudnn.ReLU(true))
      model:add(cudnn.SpatialConvolution(nChannels, nOutChannels, 1, 1, 1, 1, 0, 0))
      if dropRate then
        model:add(nn.Dropout(dropRate))
      end
      model:add(cudnn.SpatialAveragePooling(2, 2))
    end

    print("Building model")
    model = nn.Sequential()

    model:add(cudnn.SpatialConvolution(3, nChannels, 3,3, 1,1, 1,1))

    for i=1, N do 
      addLayer(model, nChannels, growthRate, dropRate)
      nChannels = nChannels + growthRate
    end
    addTransition(model, nChannels, nChannels, dropRate)

    for i=1, N do
      addLayer(model, nChannels, growthRate, dropRate)
      nChannels = nChannels + growthRate
    end
    addTransition(model, nChannels, nChannels, dropRate)

    for i=1, N do
      addLayer(model, nChannels, growthRate, dropRate)
      nChannels = nChannels + growthRate
    end

    model:add(cudnn.SpatialBatchNormalization(nChannels))
    model:add(cudnn.ReLU(true))
    model:add(cudnn.SpatialAveragePooling(8,8)):add(nn.Reshape(nChannels))
    if opt.dataset == 'cifar100' then
      model:add(nn.Linear(nChannels, 100))
    elseif opt.dataset == 'cifar10' then
      model:add(nn.Linear(nChannels, 10))
    else
      error("Dataset not supported yet!")
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
    ConvInit('nn.SpatialConvolution')
    BNInit('fbnn.SpatialBatchNormalization')
    BNInit('cudnn.SpatialBatchNormalization')
    BNInit('nn.SpatialBatchNormalization')
    for k,v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
    end
    model:cuda()

    print(model)

    return model
end

return createModel