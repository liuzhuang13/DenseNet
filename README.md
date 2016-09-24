#Densely Connected Convolutional Networks (DenseNets)
This repository contains the code for the paper [Densely Connected Convolutional Networks](http://arxiv.org/abs/1608.06993). 


The code is based on Facebook's implementation of ResNet (https://github.com/facebook/fb.resnet.torch).

Also, see 

1. Our [Caffe Implementation] (https://github.com/liuzhuang13/DenseNetCaffe)
2. Our more memory-efficient [Torch Implementation] (https://github.com/gaohuang/DenseNet_lite).
3. [Tensorflow Implementation] (https://github.com/YixuanLi/densenet-tensorflow) by Yixuan Li.
4. [Keras Implementation] (https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/DenseNet) by tdeboissiere. 
5. [Chainer Implementation] (https://github.com/t-hanya/chainer-DenseNet) by t-hanya.
6. [Chainer Implementation] (https://github.com/yasunorikudo/chainer-DenseNet) by yasunorikudo.

If you find this helps your research, please consider citing:

     @article{Huang2016Densely,
     		author = {Huang, Gao and Liu, Zhuang and Weinberger, Kilian Q.},
     		title = {Densely Connected Convolutional Networks},
     		journal = {arXiv preprint arXiv:1608.06993},
     		year = {2016}
     }


##Table of Contents
0. [Introduction](#introduction)
0. [Results](#results)
0. [Usage](#usage)
0. [Contact](#contact)

##Introduction
DenseNet is a network architecture where each layer is directly connected to every other layer in a feed-forward fashion (within each *dense block*). For each layer, the feature maps of all preceding layers are treated as separate inputs whereas its own feature maps are passed on as inputs to all subsequent layers. This connectivity pattern yields state-of-the-art accuracies on CIFAR10/100 (with or without data augmentation) and SVHN.

<img src="https://cloud.githubusercontent.com/assets/8370623/17981494/f838717a-6ad1-11e6-9391-f0906c80bc1d.jpg" width="480">

Figure 1: A dense block with 5 layers and growth rate 4. 


![densenet](https://cloud.githubusercontent.com/assets/8370623/17981496/fa648b32-6ad1-11e6-9625-02fdd72fdcd3.jpg)
Figure 2: A deep DenseNet with three dense blocks. 



##Results
The table below shows the results of DenseNets on CIFAR datasets. The "+" mark at the end denotes standard data augmentation (crop after zero-padding, and horizontal flip). For a DenseNet model, L denotes its depth and k denotes its growth rate. On CIFAR-10 and CIFAR-100 (without augmentation), Dropout with 0.2 drop rate is adopted.

Method | CIFAR-10 | CIFAR-10+ | CIFAR-100 | CIFAR-100+ 
-------|:--------:|:--------:|:--------:|:--------:|
DenseNet (L=40, k=12) |7.00 |5.24 | 27.55|24.42
DenseNet (L=100, k=12)|**5.77** |4.10 | 23.79|20.20
DenseNet (L=100, k=24)|5.83 |**3.74** | **23.42**|**19.25**

We're experimenting on ImageNet dataset, some preliminary results and pretrained models are coming out soon.


##Usage 
0. Install Torch ResNet (https://github.com/facebook/fb.resnet.torch) following the instructions there. To reduce memory consumption, we recommend to install the [optnet](https://github.com/fmassa/optimize-net) package. 
1. Add the file densenet.lua to the folder models/.
2. Change the learning rate schedule in the file train.lua: inside function learningRate(), change line 171/173
from ```decay = epoch >= 122 and 2 or epoch >= 81 and 1 or 0```
 to 
 ```decay = epoch >= 225 and 2 or epoch >= 150 and 1 or 0 ```
3. Train a DenseNet (L=40, k=12) on CIFAR-10+ using

```
th main.lua -netType densenet -depth 40 -dataset cifar10 -batchSize 64 -nEpochs 300 -optnet true
``` 
In our experiment environment (cudnn v5.1, one TITAN X GPU), this takes about 7 hours to finish

###Note
By default, the growth rate k is set to 12, dropout is disabled. To experiment with other settings, please change densenet.lua accordingly (see the comments in the code).

##Contact
liuzhuangthu at gmail.com  
gh349 at cornell.edu   
Any discussions, suggestions and questions are welcome!






