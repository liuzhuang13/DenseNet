# Memory Efficient Implementation of DenseNets

The standard (orginal) implementation of DenseNet with recursive concatenation is very memory inefficient. This can be an obstacle when we need to train DenseNets on high resolution images (such as for object detection and localization tasks) or on devices with limited memory.

In theory, DenseNet should use memory more efficiently than other networks, because one of its key features is that it encourages *feature reusing* in the network. The fact that DenseNet is "memory hungry" in practice is simply an artifact of implementation. In particular, the culprit is the recursive concatenation which *re-allocates memory* for all previous outputs at each layer. Consider a dense block with N layers, the first layer's output has N copies in the memory, the second layer's output has (N-1) copies, ..., leading to a quadratic increase (1+2+...+N) in memory consumption as the network depth grows.

Using *optnet* (*-optMemory 1*) or *shareGradInput* (*-optMemory 2*), we can significantly reduce the run-time memory footprint of the standard implementaion (with recursive concatenation). However, the memory consumption is still a quadratic function in depth. 

We implement a customized densely connected layer (largely motivated by the [Caffe implementation of memory efficient DenseNet](https://github.com/Tongcheng/DN_CaffeScript) by [Tongcheng](https://github.com/Tongcheng)), which uses shared buffers to store the concatenated outputs and gradients, thus dramatically reducing the memory footprint of DenseNet during training. The mode *-optMemory 3* activates shareGradInput and shared output buffers, while the mode *-optMemory 4* further shares the memory to store the output of the Batch-Normalization layer before each 1x1 convolution layer. The latter makes the memory consumption *linear* in network depth, but introduces a training time overhead due to the need to re-forward these Batch-Normalization layers in the backward pass.

In practice, we suggest using the default *-optMemory 2*, as it does not require customized layers, while the memory consumption is moderate. When GPU memory is really the bottleneck, we can adopt the customized implementation by setting *-optMemory* to 3 or 4, e.g.,
```
th main.lua -netType densenet -dataset cifar10 -batchSize 64 -nEpochs 300 -depth 100 -growthRate 12 -optMemory 4
``` 

The following time and memory footprint are benchmarked on a DenseNet-BC (l=100, k=12) on CIFAR-10, and on an NVIDIA TitanX GPU:

optMemory | Memory | Time (s/mini-batch) |  Description | 
:-------:|:-------:|:--------:|:-------|
 | 0 | 5453M | 0.153 | Original implementation
 | 1 | 3746M | 0.153 | Original implementation with optnet
 | 2 | 2969M | 0.152 | Original implementation with shareGradInput 
 | 3 | 2188M | 0.155 | Customized implementation with shareGradInput and sharePrevOutput
 | 4 | 1655M | 0.175 | Customized implementation with shareGradInput, sharePrevOutput and shareBNOutput 