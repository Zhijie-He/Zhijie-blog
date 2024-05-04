# 进阶教程

## GPU 训练
图形处理单元 (GPU) 是一种专用硬件加速器，旨在加速游戏和深度学习中使用的数学计算。

### Train on GPUs
默认情况下，Trainer会使用所有可用的GPU用于训练。
```python
# run on as many GPUs as available by default
trainer = Trainer(accelerator="auto", devices="auto", strategy="auto")
# equivalent to
trainer = Trainer()

# run on one GPU
trainer = Trainer(accelerator="gpu", devices=1)
# run on multiple GPUs
trainer = Trainer(accelerator="gpu", devices=8)
# choose the number of devices automatically
trainer = Trainer(accelerator="gpu", devices="auto")
```
> Note: Setting accelerator="gpu" will also automatically choose the “mps” device on Apple sillicon GPUs. If you want to avoid this, you can set accelerator="cuda" instead.

### 指定GPU
```python
# DEFAULT (int) specifies how many GPUs to use per node
Trainer(accelerator="gpu", devices=k)

# Above is equivalent to
Trainer(accelerator="gpu", devices=list(range(k)))

# Specify which GPUs to use (don't use when running on cluster)
Trainer(accelerator="gpu", devices=[0, 1])

# Equivalent using a string
Trainer(accelerator="gpu", devices="0, 1")

# To use all available GPUs put -1 or '-1'
# equivalent to `list(range(torch.cuda.device_count())) and `"auto"`
Trainer(accelerator="gpu", devices=-1)
```

### 找到可用的CUDA
> If you want to run several experiments at the same time on your machine, for example for a hyperparameter sweep, then you can use the following utility function to pick GPU indices that are “accessible”, without having to change your code every time.

```python
from lightning.pytorch.accelerators import find_usable_cuda_devices

# Find two GPUs on the system that are not already occupied
trainer = Trainer(accelerator="cuda", devices=find_usable_cuda_devices(2))

from lightning.fabric.accelerators import find_usable_cuda_devices

# Works with Fabric too
fabric = Fabric(accelerator="cuda", devices=find_usable_cuda_devices(2))
```

## TPU
- [TPU](https://lightning.ai/docs/pytorch/stable/accelerators/tpu_basic.html)
张量处理单元（TPU）是谷歌专门针对神经网络开发的人工智能加速器专用集成电路（ASIC）。

TPU 有 8 个核心，其中每个核心都针对 128x128 矩阵乘法进行了优化。一般来说，单个 TPU 的速度大约与 5 个 V100 GPU 一样快！

一个 TPU Pod 上承载许多 TPU。目前，TPU v3 Pod 拥有多达 2048 个 TPU 核心和 32 TiB 内存！您可以从 Google 云请求一个完整的 pod，或者一个“切片”，它可以为您提供这 2048 个核心的一些子集。

- 如果你的工作或项目主要集中在图形处理或你正在使用Apple设备，MPS可能是最好的选择。
- 对于广泛的科学计算和深度学习任务，特别是在你需要灵活地使用各种框架和库的情况下，GPU可能是最合适的选择。
- 如果你主要使用TensorFlow，并且涉及到大规模的模型训练或推理任务，尤其是在云环境中，TPU可能提供最好的性能。

## 模块化项目

A datamodule encapsulates the five steps involved in data processing in PyTorch:

1. Download / tokenize / process.
2. Clean and (maybe) save to disk.
3. Load inside Dataset.
4. Apply transforms (rotate, tokenize, etc…).
5. Wrap inside a DataLoader.

>This class can then be shared and used anywhere:
```python
model = LitClassifier()
trainer = Trainer()

imagenet = ImagenetDataModule()
trainer.fit(model, datamodule=imagenet)

cifar10 = CIFAR10DataModule()
trainer.fit(model, datamodule=cifar10)
```
### 什么是 DataModule
`LightningDataModule`是PyTorch Lightning一种管理数据的方式，封装了训练，验证，测试和预测的dataloader，同时还有对于数据下载，处理，转换等必要步骤。

常规PyTorch的数据处理步骤:
```python
# regular PyTorch
test_data = MNIST(my_path, train=False, download=True)
predict_data = MNIST(my_path, train=False, download=True)
train_data = MNIST(my_path, train=True, download=True)
train_data, val_data = random_split(train_data, [55000, 5000])

train_loader = DataLoader(train_data, batch_size=32)
val_loader = DataLoader(val_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)
predict_loader = DataLoader(predict_data, batch_size=32)
```

对应的DataModule使用相同的代码，但是使得这个代码在多个项目可重用性更高：
```python
class MNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: str):
        self.mnist_test = MNIST(self.data_dir, train=False)
        self.mnist_predict = MNIST(self.data_dir, train=False)
        mnist_full = MNIST(self.data_dir, train=True)
        self.mnist_train, self.mnist_val = random_split(
            mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        ...
```

随着处理复杂性的增加（转换、多 GPU 训练），可以让 Lightning 处理这些细节，同时使该数据集可重复使用，以便可以共享或在不同的项目中使用。
```python
mnist = MNISTDataModule(my_path)
model = LitClassifier()

trainer = Trainer()
trainer.fit(model, mnist)
```

这是一个更实际、更复杂的 DataModule，它显示了数据模块的可重用性有多大。
```python
import lightning as L
from torch.utils.data import random_split, DataLoader

# Note - you must have torchvision installed for this example
from torchvision.datasets import MNIST
from torchvision import transforms


class MNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./"):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

        if stage == "predict":
            self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=32)
```


### LightningDataModule API
为了定义`DataModule`，需要实现下面的方法去创建 训练/验证/测试/预测的 dataloaders:

- prepare_data (how to download, tokenize, etc…)
- setup (how to split, define dataset, etc…)
- train_dataloader
- val_dataloader
- test_dataloader
- predict_dataloader

#### prepare_data

setup() 在prepare_data 之后调用，中间有一个屏障，确保一旦数据准备好并可供使用，所有进程都会继续进行设置。

- download, i.e. download data only once on the disk from a single process
- tokenize. Since it’s a one time process, it is not recommended to do it on all processes

```python
class MNISTDataModule(L.LightningDataModule):
    def prepare_data(self):
        # download
        MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
        MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())
```
注意：prepare_data是从主进程调用，不推荐在这里进行赋值操作。

#### setup
Use setup() to do things like:
- count number of classes
- build vocabulary
- perform train/val/test splits
- create datasets
- apply transforms (defined explicitly in your datamodule)

```python
import lightning as L


class MNISTDataModule(L.LightningDataModule):
    def setup(self, stage: str):
        # Assign Train/val split(s) for use in Dataloaders
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, download=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
            )

        # Assign Test split(s) for use in Dataloaders
        if stage == "test":
            self.mnist_test = MNIST(self.data_dir, train=False, download=True, transform=self.transform)
```


例如，如果正在处理 NLP 任务，需要对文本进行标记并使用它，那么可以执行如下操作：

```python
class LitDataModule(L.LightningDataModule):
    def prepare_data(self):
        dataset = load_Dataset(...)
        train_dataset = ...
        val_dataset = ...
        # tokenize
        # save it to disk

    def setup(self, stage):
        # load it back here
        dataset = load_dataset_from_disk(...)
```
这个方法需要`stage`参数，它用于分离训练器的设置逻辑。{fit,validate,test,predict}。

> setup is called from every process across all the nodes. Setting state here is recommended.

#### train_dataloader
使用train_dataloader方法去生成training dataloaders. 这是 Trainer fit() 方法使用的数据加载器。
```python
import lightning as L


class MNISTDataModule(L.LightningDataModule):
    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=64)
```
#### val_dataloader
> This is the dataloader that the Trainer fit() and validate() methods uses.

```
import lightning as L


class MNISTDataModule(L.LightningDataModule):
    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=64)
```

#### test_dataloader
> This is the dataloader that the Trainer test() method uses.

```python
import lightning as L


class MNISTDataModule(L.LightningDataModule):
    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=64)
```

#### predict_loader
This is the dataloader that the Trainer predict() method uses.
```python
import lightning as L


class MNISTDataModule(L.LightningDataModule):
    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=64)
```


### 使用Data Module
```python
dm = MNISTDataModule()
model = Model()
trainer.fit(model, datamodule=dm)
trainer.test(datamodule=dm)
trainer.validate(datamodule=dm)
trainer.predict(datamodule=dm)
```