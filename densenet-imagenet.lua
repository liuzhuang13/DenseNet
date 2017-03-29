require 'nn'
require 'cunn'
require 'cudnn'

function createModel(opt)
    --depth
    local depth = 121
    --growth rate
    local growthRate = 32

    --#channels before entering the first denseblock
    --set it to be comparable with growth rate
    local nChannels = growthRate * 2

    --#number of transformations in each stage
    if depth == 121 then
      stages = {6, 12, 24, 16}
    elseif depth == 169 then
      stages = {6, 12, 32, 32}
    elseif depth == 201 then
      stages = {6, 12, 48, 32}
    elseif depth == 161 then
      stages = {6, 12, 36, 24}
      growthRate = 48
    end

    --feature maps reduction rate at transition layer
    local reduction = 0.5

    local function addLayer(model, nChannels, nOutChannels)
        concate = nn.Concat(2)
        concate:add(nn.Identity())

        local interChannels = 4 * nOutChannels

        convFactory = nn.Sequential()
        convFactory:add(cudnn.SpatialBatchNormalization(nChannels))
        convFactory:add(cudnn.ReLU(true))
        convFactory:add(cudnn.SpatialConvolution(nChannels, interChannels, 1, 1, 1, 1, 0, 0))

        convFactory:add(cudnn.SpatialBatchNormalization(interChannels))
        convFactory:add(cudnn.ReLU(true))
        convFactory:add(cudnn.SpatialConvolution(interChannels, nOutChannels, 3, 3, 1, 1, 1, 1))

        concate:add(convFactory)
        model:add(concate)
    end

    local function addTransition(model, nChannels, nOutChannels)
      model:add(cudnn.SpatialBatchNormalization(nChannels))
      model:add(cudnn.ReLU(true))
      model:add(cudnn.SpatialConvolution(nChannels, nOutChannels, 1, 1, 1, 1, 0, 0))
      model:add(cudnn.SpatialAveragePooling(2, 2))
    end

    print("Building model")
    model = nn.Sequential()

    --initial part following resnet on imagenet
    --input: 224*224
    model:add(cudnn.SpatialConvolution(3, nChannels, 7,7, 2,2, 3,3))
    model:add(cudnn.SpatialBatchNormalization(nChannels))
    model:add(cudnn.ReLU(true))
    model:add(nn.SpatialMaxPooling(3, 3, 2, 2, 1, 1))

    --stage 1: 56*56
    for i=1, stages[1] do 
      addLayer(model, nChannels, growthRate)
      nChannels = nChannels + growthRate
    end
    addTransition(model, nChannels, math.floor(nChannels*reduction))
    nChannels = math.floor(nChannels*reduction)

    --stage 2: 28*28
    for i=1, stages[2] do
      addLayer(model, nChannels, growthRate)
      nChannels = nChannels + growthRate
    end
    addTransition(model, nChannels, math.floor(nChannels*reduction))
    nChannels = math.floor(nChannels*reduction)

    --stage 3: 14*14
    for i=1, stages[3] do
      addLayer(model, nChannels, growthRate)
      nChannels = nChannels + growthRate
    end
    addTransition(model, nChannels, math.floor(nChannels*reduction))
    nChannels = math.floor(nChannels*reduction)

    --stage 3: 7*7
    for i=1, stages[4] do
      addLayer(model, nChannels, growthRate)
      nChannels = nChannels + growthRate
    end

    model:add(cudnn.SpatialBatchNormalization(nChannels))
    model:add(cudnn.ReLU(true))
    model:add(cudnn.SpatialAveragePooling(7,7)):add(nn.Reshape(nChannels))
    model:add(nn.Linear(nChannels, 1000))

    --Initialization following ResNet
    local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nInputPlane
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