# PyTorch Lightning 安装
> The deep learning framework to pretrain, finetune and deploy AI models.

除了这个扩充包，还有其他的类似扩充包比如：fast.ai和ignite

- Easier to reproduce 可以方便设置随机数


这里使用conda进行安装。对于conda的使用与安装可以查看该教程：[Conda](/coding-tips/conda)

1. 创建环境

`conda create -n pytorch-lightning python=3.9`

2. 安装PyTorch Lightning

Pip users

`pip install lightning`

Conda users

`conda install lightning -c conda-forge`

3. 安装pyTorch

到[pyTorch安装官网](https://pytorch.org/get-started/locally/)，去查看对应的安装命令，这里以MacOS为例：

`conda install pytorch::pytorch torchvision torchaudio -c pytorch`


4. 安装tensorboard

`conda install matplotlib tensorboard`

Note: 这里当我用python 3.12版本的时候，使用tensorboard会出现bug，返回到python 3.9之后就没问题。原因是 `from collections import Mapping` 在python 3.10之后就停止工作了。

## 教程
- [AI葵](https://www.youtube.com/watch?v=O7dNXpgdWbo&list=PLDV2CyUo4q-JFVFS52gorFnfFJC7w7ElJ)
- [official tutorial](https://lightning.ai/docs/pytorch/stable/search.html?q=tutorial&check_keywords=yes&area=default)



## FAQ

- 当按照上述过程安装后，在import torchvision过程中出现userwarning: Failed to load image Python extension
解决方案：<https://discuss.pytorch.org/t/failed-to-load-image-python-extension-could-not-find-module/140278>

```
(pytorch-lightning) zhijiehe@zhijiedeAir pytorch_lightning % conda list | grep torch
# packages in environment at /opt/anaconda3/envs/pytorch-lightning:
pytorch                   2.3.0                   py3.9_0    pytorch
pytorch-lightning         2.2.2              pyhd8ed1ab_0    conda-forge
torchaudio                2.3.0                  py39_cpu    pytorch
torchmetrics              1.3.2              pyhd8ed1ab_0    conda-forge
torchvision               0.15.2          cpu_py39h31aa045_0  
```
我尝试downgrade torchvision和torch
>Hi,
>I faced the same problem on my MBA@M1 and downgrading of the torch and torchvision helped me.
>
>pip install --upgrade torch==1.9.0 \
>pip install --upgrade torchvision==0.10.0

但是问题是当我重装torchvision到0.15.2的时候，warnings没有再出现。具体原因不知道为什么。

最终
```
(pytorch-lightning) zhijiehe@zhijiedeAir pytorch_lightning % conda list | grep torch
# packages in environment at /opt/anaconda3/envs/pytorch-lightning:
pytorch-lightning         2.2.2              pyhd8ed1ab_0    conda-forge
torch                     2.0.1                    pypi_0    pypi
torchaudio                2.3.0                  py39_cpu    pytorch
torchmetrics              1.3.2              pyhd8ed1ab_0    conda-forge
torchvision               0.18.0                   pypi_0    pypi
```