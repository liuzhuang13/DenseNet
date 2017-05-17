# Densely Connected Convolutional Networks (DenseNets)

The code for:
        
Gao Huang\*, Zhuang Liu\*, Kilian Weinberger, Laurens van der Maaten, ["Densely Connected Convolutional Networks"](http://arxiv.org/abs/1608.06993), CVPR 2017. (\* equal contribution)

The code is based on [fb.resnet.torch](https://github.com/facebook/fb.resnet.torch).

Citation:

	@inproceedings{huang2017densely,
	  title={Densely connected convolutional networks},
	  author={Huang, Gao and Liu, Zhuang and Weinberger, Kilian Q and van der Maaten, Laurens},
	  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
	  year={2017}
	}
	
## Other Implementations
0. Our [Caffe Implementation](https://github.com/liuzhuang13/DenseNetCaffe)
0. Our space-efficient [Torch Implementation](https://github.com/gaohuang/DenseNet_lite).
0. Our (much more) space-efficient [Caffe Implementation](https://github.com/Tongcheng/DN_CaffeScript).
0. [PyTorch Implementation (with BC structure)](https://github.com/andreasveit/densenet-pytorch) by Andreas Veit.
0. [PyTorch Implementation (with BC structure)](https://github.com/bamos/densenet.pytorch) by Brandon Amos.
0. [MXNet Implementation](https://github.com/Nicatio/Densenet/tree/master/mxnet) by Nicatio.
0. [MXNet Implementation (supporting ImageNet)](https://github.com/bruinxiong/densenet.mxnet) by Xiong Lin.
0. [Tensorflow Implementation](https://github.com/YixuanLi/densenet-tensorflow) by Yixuan Li.
0. [Tensorflow Implementation](https://github.com/LaurentMazare/deep-models/tree/master/densenet) by Laurent Mazare.
0. [Tensorflow Implementation (with BC structure)](https://github.com/ikhlestov/vision_networks) by Illarion Khlestov.
0. [Lasagne Implementation](https://github.com/Lasagne/Recipes/tree/master/papers/densenet) by Jan Schlüter.
0. [Keras Implementation](https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/DenseNet) by tdeboissiere. 
0. [Keras Implementation](https://github.com/robertomest/convnet-study) by Roberto de Moura Estevão Filho.
0. [Keras Implementation (with BC structure)](https://github.com/titu1994/DenseNet) by Somshubra Majumdar.
0. [Chainer Implementation](https://github.com/t-hanya/chainer-DenseNet) by Toshinori Hanya.
0. [Chainer Implementation](https://github.com/yasunorikudo/chainer-DenseNet) by Yasunori Kudo.
0. [Fully Convolutional DenseNets for segmentation](https://github.com/SimJeg/FC-DenseNet) by Simon Jegou.

Note that we didn't list all implementations available on GitHub, and didn't label all implementations which support BC structures. 


## Introduction
DenseNet is a network architecture where each layer is directly connected to every other layer in a feed-forward fashion (within each *dense block*). For each layer, the feature maps of all preceding layers are treated as separate inputs whereas its own feature maps are passed on as inputs to all subsequent layers. This connectivity pattern yields state-of-the-art accuracies on CIFAR10/100 (with or without data augmentation) and SVHN.

<img src="https://cloud.githubusercontent.com/assets/8370623/17981494/f838717a-6ad1-11e6-9391-f0906c80bc1d.jpg" width="480">

Figure 1: A dense block with 5 layers and growth rate 4. 


![densenet](https://cloud.githubusercontent.com/assets/8370623/17981496/fa648b32-6ad1-11e6-9625-02fdd72fdcd3.jpg)
Figure 2: A deep DenseNet with three dense blocks. 



## Results on CIFAR
The table below shows the results of DenseNets on CIFAR datasets. The "+" mark at the end denotes standard data augmentation (crop after zero-padding, and horizontal flip). For a DenseNet model, L denotes its depth and k denotes its growth rate. On CIFAR-10 and CIFAR-100 (without augmentation), Dropout with 0.2 drop rate is adopted.

Model | Parameters| CIFAR-10 | CIFAR-10+ | CIFAR-100 | CIFAR-100+ 
-------|:-------:|:--------:|:--------:|:--------:|:--------:|
DenseNet (L=40, k=12) |1.0M |7.00 |5.24 | 27.55|24.42
DenseNet (L=100, k=12)|7.0M |5.77 |4.10 | 23.79|20.20
DenseNet (L=100, k=24)|27.2M |5.83 |3.74 | 23.42|19.25
DenseNet-BC (L=100, k=12)|0.8M |5.92 |4.51 | 24.15|22.27
DenseNet-BC (L=250, k=24)|15.3M |**5.19** |3.62 | **19.64**|17.60
DenseNet-BC (L=190, k=40)|25.6M |- |**3.46** | -|**17.18**

## Wide-DenseNet for Reducing Memory and Time Consumption

If you use DenseNet as a model in your learning task, to reduce the memory and time consumption, we recommend use a wide and shallow DenseNet, following the strategy of [wide residual networks](https://github.com/szagoruyko/wide-residual-networks). To obtain a wide DenseNet we set the depth to be smaller (e.g., L=40) and the growthRate to be larger (e.g., k=48).

 We test a set of Wide-DenseNet-BCs and compareED the memory and time with the DenseNet-BC (L=100, k=12) shown above. We obtained the statistics using a single TITAN X card, with batch size 64, and without the optnet package in Torch.


Model | Parameters| CIFAR-10+ | CIFAR-100+ | Time per Iteration | Memory 
-------|:-------:|:--------:|:--------:|:--------:|:--------:|
DenseNet-BC (L=100, k=12)|0.8M |4.51 |22.27 | 0.156s | 5452MB
Wide-DenseNet-BC (L=40, k=36)|1.5M |4.58 |22.30 | 0.130s|4008MB
Wide-DenseNet-BC (L=40, k=48)|2.7M |3.99 |20.29 | 0.165s|5245MB
Wide-DenseNet-BC (L=40, k=60)|4.3M |4.01 |19.99 | 0.223s|6508MB

Obersevations:

1. Wide-DenseNet-BC (L=40, k=36) uses less memory/time while achieves about the same accuracy as DenseNet-BC (L=100, k=12). 
2. Wide-DenseNet-BC (L=40, k=48) uses about the same memory/time as DenseNet-BC (L=100, k=12), while is much more accurate.

Thus, for practical use, we suggest picking one model from those Wide-DenseNet-BCs.



## ImageNet and Pretrained Models
### Torch
The Torch models are trained under the same setting as in [fb.resnet.torch](https://github.com/facebook/fb.resnet.torch). The error rates shown are 224x224 1-crop test errors.

| Network       |  Top-1 error | Torch Model |
| ------------- | ----------- | ----------- |
| DenseNet-121 (k=32)    |   25.0     | [Download (64.5MB)](https://drive.google.com/open?id=0B8ReS-sYUS-HWFViYlVlZk9sdHc)       |
| DenseNet-169 (k=32)    | 23.6       | [Download (114.4MB)](https://drive.google.com/open?id=0B8ReS-sYUS-HY25Dc2VrUGlVSGc)       |
| DenseNet-201 (k=32)    | 22.5       | [Download (161.8MB)](https://drive.google.com/open?id=0B8ReS-sYUS-HaDdpNmlWRjJkd3c)       |
| DenseNet-161 (k=48)    | 22.2       | [Download (230.8MB)](https://drive.google.com/open?id=0B8ReS-sYUS-HVXp2RExSTmMzZVU)

### Caffe
For ImageNet pretrained Caffe models, please see https://github.com/shicai/DenseNet-Caffe. Also, we would like to thank @szq0214 for help on Caffe models.


### PyTorch
In PyTorch, ImageNet pretrained models can be directly loaded by 

```
import torchvision.models as models
densenet = models.densenet_161(pretrained=True)
```

For ImageNet training, customized models can be constructed by simply calling

```
DenseNet(growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000)
```

See more details at [PyTorch documentation on models](http://pytorch.org/docs/torchvision/models.html?highlight=densenet) and the [code for DenseNet](https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py). We would like to thank @gpleiss for this nice work in PyTorch.

### Keras, Tensorflow and Theano
Please see https://github.com/flyyufelix/DenseNet-Keras.



## Usage 
For training on CIFAR dataset,

0. Install Torch ResNet (https://github.com/facebook/fb.resnet.torch) following the instructions there. To reduce memory consumption, we recommend to install the [optnet](https://github.com/fmassa/optimize-net) package. 
1. Add the file densenet.lua to the folder models/.
2. Change the learning rate schedule in the file train.lua: inside function learningRate(), change line 171/173
from 
```decay = epoch >= 122 and 2 or epoch >= 81 and 1 or 0```
 to 
 ```decay = epoch >= 225 and 2 or epoch >= 150 and 1 or 0 ```
3. For example, train a Wide-DenseNet-BC (L=40, k=48) on CIFAR-10+ using

```
th main.lua -netType densenet -depth 40 -dataset cifar10 -batchSize 64 -nEpochs 300 -optnet true
``` 

The file densenet-imagenet.lua is for training ImageNet models presented in the paper. The usage is very similar. Please refer to [fb.resnet.torch](https://github.com/facebook/fb.resnet.torch) for data preparation.

### Note
On CIFAR, by default, the growth rate k is set to 48, bottleneck transformation is used, compression rate at transiton layers is 0.5,  dropout is disabled. On ImageNet, the default model is densenet-121. To experiment with other settings, please change densenet.lua accordingly (see the comments in the code).

If you use the model on images of different sizes from CIFAR/ImageNet, you may need to choose a different downsampling strategy.

## Updates

12/03/2016:

1. Add Imagenet results and pretrained models.
2. Add DenseNet-BC structures.

03/29/2017:

1. Add the code for imagenet training.

04/20/2017:

1. Add usage of models in PyTorch.

05/17/2017:

1. Add Wide-DenseNet.
2. Add keras, tf, theano link for pretrained models.




## Contact
liuzhuangthu at gmail.com  
gh349 at cornell.edu   
Any discussions, suggestions and questions are welcome!






