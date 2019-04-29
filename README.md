# ScaleNet

By Yi Li, Zhanghui Kuang, Yimin Chen, Wayne Zhang

SenseTime.

### Table of Contents
0. [Introduction](#introduction)
0. [Citation](#citation)
0. [Approach](#approach)
0. [Trained models](#trained-models)
0. [Experiments](#experiments)
0. [GPU time](#gpu-time)

### Introduction

This is a PyTorch implementation of [Data-Driven Neuron Allocation for Scale Aggregation Networks](https://arxiv.org/pdf/1904.09460.pdf).(CVPR2019) with pretrained models. 

### Citation

If you use these models in your research, please cite:

	@inproceedings{Li2019ScaleNet,
        title={Data-Driven Neuron Allocation for Scale Aggregation Networks},
        author={Li, Yi and Kuang, Zhanghui and Chen, Yimin and Zhang, Wayne},
        booktitle={CVPR},
        year={2019}
	}

### Approach
<div align="center">
  <img src="https://github.com/Eli-YiLi/ScaleNet/blob/master/figures/architecture.png" width="300">
</div>
<p align="center">
  Figure 1: architecture of ScaleNet-50.
</p>

<div align="center">
  <img src="https://github.com/Eli-YiLi/ScaleNet/blob/master/figures/sablock.png" width="200">
</div>
<p align="center">
  Figure 2: scale aggregation block.
</p>

### Trained models
| Model | Top-1 err. | Top-5 err. |
|:-:|:-:|:-:|
| ScaleNet-50-light | 22.80 | 6.57 |
| ScaleNet-50 | 22.02 | 6.05 |
| ScaleNet-101 | 20.82 | 5.42 |
| ScaleNet-152 | 20.06 | 5.18 |

Pytorch:

```
from pytorch.scalenet import *
```
```
model = scalenet50(structure_path='structures/scalenet50.json', ckpt=None) # train from stratch
```
```
model = scalenet50(structure_path='structures/scalenet50.json', ckpt='weights/scalenet50.pth') # load pretrained model
```

The weights are available on [BaiduYun](https://pan.baidu.com/s/1NOjFWzkAVmMNkZh6jIcMzA) with extract code: f1c5

Unlike the paper, we used better training settings: increase the epochs to 120 and replace multi-step learning rate by cosine learning rate.

### Experiments

<div align="center">
  <img src="https://github.com/Eli-YiLi/ScaleNet/blob/master/figures/imagenet.png" width="300">
</div>
<p align="center">
  Figure 3: experiments on imagenet classification.
</p>

<div align="center">
  <img src="https://github.com/Eli-YiLi/ScaleNet/blob/master/figures/coco.png" width="300">
</div>
<p align="center">
  Figure 4: experiments on ms-coco detection.
</p>

### GPU time
| Model | Top-1 err. | FLOPs(10^9) | GPU time(ms)|
|:-:|:-:|:-:|:-:|
| ResNet-50 | 24.02 | 4.1 | 95 |
| SE-ResNet-50 | 23.29 | 4.1 | 98 |
| ResNeXt-50 | 22.2 | 4.2 | 147 |
| ScaleNet-50 | 22.2 | 3.8 | 93 |

TensorFlow:
(empty models of ResNet, SE-ResNet, ResNeXt, ScaleNet for speed test)
```
python3 tensorflow/test_speed.py scale|res|se|next
```

All networks were tested using Tensorflow with GTX 1060 GPU and i7 CPU at batch size 16 and image side 224 on 1000 runs.

Some static-graph frameworks like Tensorflow and TensorRT execute multi-branch models in parallel, while Pytorch and Caffe do not. So we suggest to deploy ScaleNets on Tensorflow and TensorRT.
