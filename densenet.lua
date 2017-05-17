require 'nn'
require 'cunn'
require 'cudnn'

local function createModel(opt)
    --growth rate
    local growthRate = 48

    --dropout rate, set it to 0 to disable dropout, non-zero number to enable dropout and set drop rate
    local dropRate = 0

    --#channels before entering the first denseblock
    local nChannels = 2 * growthRate

    --compression rate at transition layers
    local reduction = 0.5

    --whether to use bottleneck structures
    local bottleneck = true

    --In our paper, a DenseNet-BC uses compression rate 0.5 with bottleneck structures
    --a default DenseNet uses compression rate 1 without bottleneck structures

    --N: #transformations in each denseblock
    local N = (opt.depth - 4)/3
    if bottleneck then N = N/2 end

    --non-bottleneck transformation
    local function addSingleLayer(model, nChannels, nOutChannels, dropRate)
      concate = nn.Concat(2)
      concate:add(nn.Identity())

      convFactory = nn.Sequential()
      convFactory:add(cudnn.SpatialBatchNormalization(nChannels))
      convFactory:add(cudnn.ReLU(true))
      convFactory:add(cudnn.SpatialConvolution(nChannels, nOutChannels, 3, 3, 1, 1, 1,1))
      if dropRate>0 then
        convFactory:add(nn.Dropout(dropRate))
      end
      concate:add(convFactory)
      model:add(concate)
    end


    --bottleneck transformation
    local function addBottleneckLayer(model, nChannels, nOutChannels, dropRate)
      concate = nn.Concat(2)
      concate:add(nn.Identity())

      local interChannels = 4 * nOutChannels

      convFactory = nn.Sequential()
      convFactory:add(cudnn.SpatialBatchNormalization(nChannels))
      convFactory:add(cudnn.ReLU(true))
      convFactory:add(cudnn.SpatialConvolution(nChannels, interChannels, 1, 1, 1, 1, 0, 0))
      if dropRate>0 then
        convFactory:add(nn.Dropout(dropRate))
      end

      convFactory:add(cudnn.SpatialBatchNormalization(interChannels))
      convFactory:add(cudnn.ReLU(true))
      convFactory:add(cudnn.SpatialConvolution(interChannels, nOutChannels, 3, 3, 1, 1, 1, 1))
      if dropRate>0 then
        convFactory:add(nn.Dropout(dropRate))
      end

      concate:add(convFactory)
      model:add(concate)
    end

    if bottleneck then
      add = addBottleneckLayer
    else
      add = addSingleLayer
    end

    local function addTransition(model, nChannels, nOutChannels, dropRate)
      model:add(cudnn.SpatialBatchNormalization(nChannels))
      model:add(cudnn.ReLU(true))
      model:add(cudnn.SpatialConvolution(nChannels, nOutChannels, 1, 1, 1, 1, 0, 0))
      if dropRate>0 then
        model:add(nn.Dropout(dropRate))
      end
      model:add(cudnn.SpatialAveragePooling(2, 2))
    end

    model = nn.Sequential()


    --first conv before any dense blocks
    model:add(cudnn.SpatialConvolution(3, nChannels, 3,3, 1,1, 1,1))

    --1st dense block and transition
    for i=1, N do 
      add(model, nChannels, growthRate, dropRate)
      nChannels = nChannels + growthRate
    end
    addTransition(model, nChannels, math.floor(nChannels*reduction), dropRate)
    nChannels = math.floor(nChannels*reduction)

    --2nd dense block and transition
    for i=1, N do
      add(model, nChannels, growthRate, dropRate)
      nChannels = nChannels + growthRate
    end
    addTransition(model, nChannels, math.floor(nChannels*reduction), dropRate)
    nChannels = math.floor(nChannels*reduction)

    --3rd dense block
    for i=1, N do
      add(model, nChannels, growthRate, dropRate)
      nChannels = nChannels + growthRate
    end

    --global average pooling and classifier
    model:add(cudnn.SpatialBatchNormalization(nChannels))
    model:add(cudnn.ReLU(true))
    model:add(cudnn.SpatialAveragePooling(8,8)):add(nn.Reshape(nChannels))
    if opt.dataset == 'cifar100' then
      model:add(nn.Linear(nChannels, 100))
    elseif opt.dataset == 'cifar10' then
      model:add(nn.Linear(nChannels, 10))
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


    model:cuda()
    print(model)

    return model
end

return createModel