# Densely Connected Convolutional Networks (DenseNets)

This repository contains the code for DenseNet introduced in the following paper 

[Densely Connected Convolutional Networks](http://arxiv.org/abs/1608.06993) (CVPR 2017, Best Paper Award) 

[Gao Huang](http://www.cs.cornell.edu/~gaohuang/)\*, [Zhuang Liu](https://liuzhuang13.github.io/)\*, [Laurens van der Maaten](https://lvdmaaten.github.io/) and [Kilian Weinberger](https://www.cs.cornell.edu/~kilian/) (\* Authors contributed equally).

and its journal version

[Convolutional Networks with Dense Connectivity](http://www.gaohuang.net/papers/DenseNet_Journal.pdf) (TPAMI 2019) 

[Gao Huang](http://www.cs.cornell.edu/~gaohuang/), [Zhuang Liu](https://liuzhuang13.github.io/), [Geoff Pleiss](https://geoffpleiss.com/), [Laurens van der Maaten](https://lvdmaaten.github.io/) and [Kilian Weinberger](https://www.cs.cornell.edu/~kilian/).


**Now with memory-efficient implementation!** Please check the [technical report](https://arxiv.org/pdf/1707.06990.pdf) and [code](https://github.com/liuzhuang13/DenseNet/tree/master/models) for more infomation.
 
The code is built on [fb.resnet.torch](https://github.com/facebook/fb.resnet.torch).

### Citation
If you find DenseNet useful in your research, please consider citing:

	@article{huang2019convolutional,
	 title={Convolutional Networks with Dense Connectivity},
	 author={Huang, Gao and Liu, Zhuang and Pleiss, Geoff and Van Der Maaten, Laurens and Weinberger, Kilian},
	 journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
	 year={2019}
	 }
	 
	@inproceedings{huang2017densely,
	  title={Densely Connected Convolutional Networks},
	  author={Huang, Gao and Liu, Zhuang and van der Maaten, Laurens and Weinberger, Kilian Q },
	  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
	  year={2017}
	}
	



## Other Implementations
Our [[Caffe]](https://github.com/liuzhuang13/DenseNetCaffe), Our memory-efficient [[Caffe]](https://github.com/Tongcheng/DN_CaffeScript), Our memory-efficient [[PyTorch]](https://github.com/gpleiss/efficient_densenet_pytorch), 
[[PyTorch]](https://github.com/andreasveit/densenet-pytorch) by Andreas Veit, [[PyTorch]](https://github.com/bamos/densenet.pytorch) by Brandon Amos, [[PyTorch]](https://github.com/baldassarreFe/pytorch-densenet-tiramisu) by Federico Baldassarre,
[[MXNet]](https://github.com/Nicatio/Densenet/tree/master/mxnet) by Nicatio, 
[[MXNet]](https://github.com/bruinxiong/densenet.mxnet) by Xiong Lin, 
[[MXNet]](https://github.com/miraclewkf/DenseNet) by miraclewkf,
[[Tensorflow]](https://github.com/YixuanLi/densenet-tensorflow) by Yixuan Li, 
[[Tensorflow]](https://github.com/LaurentMazare/deep-models/tree/master/densenet) by Laurent Mazare,
[[Tensorflow]](https://github.com/ikhlestov/vision_networks) by Illarion Khlestov, 
[[Lasagne]](https://github.com/Lasagne/Recipes/tree/master/papers/densenet) by Jan Schlüter, 
[[Keras]](https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/DenseNet) by tdeboissiere,  
[[Keras]](https://github.com/robertomest/convnet-study) by Roberto de Moura Estevão Filho, 
[[Keras]](https://github.com/titu1994/DenseNet) by Somshubra Majumdar, 
[[Chainer]](https://github.com/t-hanya/chainer-DenseNet) by Toshinori Hanya, 
[[Chainer]](https://github.com/yasunorikudo/chainer-DenseNet) by Yasunori Kudo, 
[[Torch 3D-DenseNet]](https://github.com/barrykui/3ddensenet.torch) by Barry Kui,
[[Keras]](https://github.com/cmasch/densenet) by Christopher Masch,
[[Tensorflow2]](https://github.com/okason97/DenseNet-Tensorflow2) by Gaston Rios and Ulises Jeremias Cornejo Fandos.



Note that we only listed some early implementations here. If you would like to add yours, please submit a pull request.

## Some Following up Projects
0. [Multi-Scale Dense Convolutional Networks for Efficient Prediction](https://github.com/gaohuang/MSDNet)
0. [DSOD: Learning Deeply Supervised Object Detectors from Scratch](https://github.com/szq0214/DSOD)
0. [CondenseNet: An Efficient DenseNet using Learned Group Convolutions](https://github.com/ShichenLiu/CondenseNet)
0. [Fully Convolutional DenseNets for Semantic Segmentation](https://github.com/SimJeg/FC-DenseNet)
0. [Pelee: A Real-Time Object Detection System on Mobile Devices](https://github.com/Robert-JunWang/Pelee)




## Contents
1. [Introduction](#introduction)
2. [Usage](#usage)
3. [Results on CIFAR](#results-on-cifar)
4. [Results on ImageNet and Pretrained Models](#results-on-imagenet-and-pretrained-models)
5. [Updates](#updates)


## Introduction
DenseNet is a network architecture where each layer is directly connected to every other layer in a feed-forward fashion (within each *dense block*). For each layer, the feature maps of all preceding layers are treated as separate inputs whereas its own feature maps are passed on as inputs to all subsequent layers. This connectivity pattern yields state-of-the-art accuracies on CIFAR10/100 (with or without data augmentation) and SVHN. On the large scale ILSVRC 2012 (ImageNet) dataset, DenseNet achieves a similar accuracy as ResNet, but using less than half the amount of parameters and roughly half the number of FLOPs.

<img src="https://cloud.githubusercontent.com/assets/8370623/17981494/f838717a-6ad1-11e6-9391-f0906c80bc1d.jpg" width="480">

Figure 1: A dense block with 5 layers and growth rate 4. 


![densenet](https://cloud.githubusercontent.com/assets/8370623/17981496/fa648b32-6ad1-11e6-9625-02fdd72fdcd3.jpg)
Figure 2: A deep DenseNet with three dense blocks. 


## Usage 
0. Install Torch and required dependencies like cuDNN. See the instructions [here](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md) for a step-by-step guide.
1. Clone this repo: ```git clone https://github.com/liuzhuang13/DenseNet.git```

As an example, the following command trains a DenseNet-BC with depth L=100 and growth rate k=12 on CIFAR-10:
```
th main.lua -netType densenet -dataset cifar10 -batchSize 64 -nEpochs 300 -depth 100 -growthRate 12
``` 
As another example, the following command trains a DenseNet-BC with depth L=121 and growth rate k=32 on ImageNet:
```
th main.lua -netType densenet -dataset imagenet -data [dataFolder] -batchSize 256 -nEpochs 90 -depth 121 -growthRate 32 -nGPU 4 -nThreads 16 -optMemory 3
``` 
Please refer to [fb.resnet.torch](https://github.com/facebook/fb.resnet.torch) for data preparation.

### DenseNet and DenseNet-BC
By default, the code runs with the DenseNet-BC architecture, which has 1x1 convolutional *bottleneck* layers, and *compresses* the number of channels at each transition layer by 0.5. To run with the original DenseNet, simply use the options *-bottleneck false* and *-reduction 1*

### Memory efficient implementation (newly added feature on June 6, 2017)
There is an option *-optMemory* which is very useful for reducing GPU memory footprint when training a DenseNet. By default, the value is set to 2, which activates the *shareGradInput* function (with small modifications from [here](https://github.com/facebook/fb.resnet.torch/blob/master/models/init.lua#L102)). There are two extreme memory efficient modes (*-optMemory 3* or *-optMemory 4*) which use a customized densely connected layer. With *-optMemory 4*, the largest 190-layer DenseNet-BC on CIFAR can be trained on a single NVIDIA TitanX GPU (uses 8.3G of 12G) instead of fully using four GPUs with the standard (recursive concatenation) implementation . 

More details about the memory efficient implementation are discussed [here](https://github.com/liuzhuang13/DenseNet/tree/master/models).


## Results on CIFAR
The table below shows the results of DenseNets on CIFAR datasets. The "+" mark at the end denotes for standard data augmentation (random crop after zero-padding, and horizontal flip). For a DenseNet model, L denotes its depth and k denotes its growth rate. On CIFAR-10 and CIFAR-100 without data augmentation, a Dropout layer with drop rate 0.2 is introduced after each convolutional layer except the very first one.

Model | Parameters| CIFAR-10 | CIFAR-10+ | CIFAR-100 | CIFAR-100+ 
-------|:-------:|:--------:|:--------:|:--------:|:--------:|
DenseNet (L=40, k=12) |1.0M |7.00 |5.24 | 27.55|24.42
DenseNet (L=100, k=12)|7.0M |5.77 |4.10 | 23.79|20.20
DenseNet (L=100, k=24)|27.2M |5.83 |3.74 | 23.42|19.25
DenseNet-BC (L=100, k=12)|0.8M |5.92 |4.51 | 24.15|22.27
DenseNet-BC (L=250, k=24)|15.3M |**5.19** |3.62 | **19.64**|17.60
DenseNet-BC (L=190, k=40)|25.6M |- |**3.46** | -|**17.18**


## Results on ImageNet and Pretrained Models
### Torch

Note: the pre-trained models in Torch are deprecated and no longer maintained. Please use PyTorch's pre-trained [DenseNet models](https://pytorch.org/vision/stable/models.html) instead.

#### Models in the original paper
The Torch models are trained under the same setting as in [fb.resnet.torch](https://github.com/facebook/fb.resnet.torch). The error rates shown are 224x224 1-crop test errors.

| Network       |  Top-1 error | Torch Model |
| ------------- | ----------- | ----------- |
| DenseNet-121 (k=32)  |  25.0    | [Download (64.5MB)](https://drive.google.com/open?id=0B8ReS-sYUS-HWFViYlVlZk9sdHc)       |
| DenseNet-169 (k=32)  | 23.6     | [Download (114.4MB)](https://drive.google.com/open?id=0B8ReS-sYUS-HY25Dc2VrUGlVSGc)       |
| DenseNet-201 (k=32)  | 22.5     | [Download (161.8MB)](https://drive.google.com/open?id=0B8ReS-sYUS-HaDdpNmlWRjJkd3c)       |
| DenseNet-161 (k=48)  | 22.2     | [Download (230.8MB)](https://drive.google.com/open?id=0B8ReS-sYUS-HVXp2RExSTmMzZVU)

#### Models in the tech report
More accurate models trained with the memory efficient implementation in the [technical report](https://arxiv.org/pdf/1707.06990.pdf). 


| Network       |  Top-1 error | Torch Model |
| ------------- | ----------- | ------------ |
| DenseNet-264 (k=32)  |  22.1 |    [Download (256MB)](https://drive.google.com/file/d/0By1NwtA2JPGzdVRqOEotMUZrbTA/view?usp=sharing)
| DenseNet-232 (k=48)  | 21.2 | [Download (426MB)](https://drive.google.com/open?id=0By1NwtA2JPGzdkRDaWQ5M3VHTDg)
| DenseNet-cosine-264 (k=32) | 21.6  |  [Download (256MB)](https://drive.google.com/file/d/0By1NwtA2JPGzRDhxWGo2a3pOTjA/view?usp=sharing)
| DenseNet-cosine-264 (k=48) | 20.4   | [Download (557MB)](https://drive.google.com/file/d/0By1NwtA2JPGzcnFDSE1HQVh4c0k/view?usp=sharing)


### Caffe
https://github.com/shicai/DenseNet-Caffe.

### PyTorch
[PyTorch documentation on models](http://pytorch.org/docs/torchvision/models.html?highlight=densenet). We would like to thank @gpleiss for this nice work in PyTorch.

### Keras, Tensorflow and Theano
https://github.com/flyyufelix/DenseNet-Keras.

### MXNet
https://github.com/miraclewkf/DenseNet.


## Wide-DenseNet for better Time/Accuracy and Memory/Accuracy Tradeoff

If you use DenseNet as a model in your learning task, to reduce the memory and time consumption, we recommend use a wide and shallow DenseNet, following the strategy of [wide residual networks](https://github.com/szagoruyko/wide-residual-networks). To obtain a wide DenseNet we set the depth to be smaller (e.g., L=40) and the growthRate to be larger (e.g., k=48).

We test a set of Wide-DenseNet-BCs and compared the memory and time with the DenseNet-BC (L=100, k=12) shown above. We obtained the statistics using a single TITAN X card, with batch size 64, and without any memory optimization.


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



## Updates
**12/10/2019:**

1. Journal version (accepted by IEEE TPAMI) released.

08/23/2017:

1. Add supporting code, so one can simply *git clone* and run.

06/06/2017:

1. Support **ultra memory efficient** training of DenseNet with *customized densely connected layer*.

2. Support **memory efficient** training of DenseNet with *standard densely connected layer* (recursive concatenation) by fixing the *shareGradInput* function.

05/17/2017:

1. Add Wide-DenseNet.
2. Add keras, tf, theano link for pretrained models.

04/20/2017:

1. Add usage of models in PyTorch.

03/29/2017:

1. Add the code for imagenet training.

12/03/2016:

1. Add Imagenet results and pretrained models.
2. Add DenseNet-BC structures.



## Contact
liuzhuangthu at gmail.com  
gaohuang at tsinghua.edu.cn   
Any discussions, suggestions and questions are welcome!

