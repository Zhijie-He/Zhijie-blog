# 安装

这里主要介绍PyTorch的基础教学以及如何使用Apple M1芯片进行Machine learning的训练。

> Until now, PyTorch training on Mac only leveraged the CPU, but with the upcoming PyTorch v1.12 release, developers and researchers can take advantage of Apple silicon GPUs for significantly faster model training.
> Accelerated GPU training is enabled using Apple’s Metal Performance Shaders (MPS) as a backend for PyTorch.

## 资源
* <https://developer.apple.com/metal/pytorch/>
* <https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/>
* <https://medium.com/@manyi.yim/pytorch-on-mac-m1-gpu-installation-and-performance-698442a4af1e>
* torch算术运算 <https://pytorch.org/docs/stable/torch.html>
* UvA Deep Learning Tutorials <https://uvadlc-notebooks.readthedocs.io/en/latest/index.html>
* official tutorials <https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html>


## MacOS 安装PyTorch

> To get started, just install the latest Preview (Nightly) build on your Apple silicon Mac running macOS 12.3 or later with a native version (arm64) of Python.

这里使用conda进行安装。对于conda的使用与安装可以查看该教程：[Conda](/coding-tips/conda)

* 安装PyTorch

到[PyTorch安装官网](https://pytorch.org/get-started/locally/)，去查看对应的安装命令, 这里以Mac M1为例：

```python
conda install pytorch-nightly::pytorch torchvision torchaudio -c pytorch-nightly
```

### 验证
``` python
>>> import torch
>>> print(torch.backends.mps.is_available())
True
```

