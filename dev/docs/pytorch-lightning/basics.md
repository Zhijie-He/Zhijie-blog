# 基础教程
<https://lightning.ai/docs/pytorch/stable/levels/core_skills.html>

## 训练模型

### import需要的库
```python
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import lightning as L
```

### 定义PyTorch模型
```python
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x):
        return self.l1(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    def forward(self, x):
        return self.l1(x)
```

### 定义LightningModule
这里LightningModule负责了训练模型的全部过程，定义了`nn.Module`将如何交互：
- `training_step`方法定义了`nn.Module`如何交互
- `configure_optimizers`方法定义了模型的optimizer

```python
class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
```

### 定义训练数据集
这里和[PyTorch](/pytorch)中一样，使用`DataLoader`去存储训练数据集：
```python
dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
train_loader = DataLoader(dataset)
```

### 训练模型
这里使用Lightning中的[Trainer](https://lightning.ai/docs/pytorch/stable/common/trainer.html)去负责模型的训练：
```python
# model
autoencoder = LitAutoEncoder(Encoder(), Decoder())

# train model
trainer = L.Trainer()
trainer.fit(model=autoencoder, train_dataloaders=train_loader)
```
实际上，这里Trainer代替我们执行下面的代码：
``` python
autoencoder = LitAutoEncoder(Encoder(), Decoder())
optimizer = autoencoder.configure_optimizers()

for batch_idx, batch in enumerate(train_loader):
    loss = autoencoder.training_step(batch, batch_idx)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

通过Trainer 我们可以极大的减少代码量。 特别是当需要验证数据集，测试数据集，分布式训练等等，Trainer可以轻易的帮助我们完成。

> With Lightning, you can add mix all these techniques together without needing to rewrite a new loop every time.


## 验证和测试循环
在上一节我们只添加了训练数据集，这一节我们将介绍如何添加验证(val)和测试(test)数据，以防模型的过拟合。

### 测试循环
首先获得测试数据集
```python{6-8}
i0ort torch.utils.data as data
from torchvision import datasets
import torchvision.transforms as transforms

# Load data sets
transform = transforms.ToTensor()
train_set = datasets.MNIST(root="MNIST", download=True, train=True, transform=transform)
test_set = datasets.MNIST(root="MNIST", download=True, train=False, transform=transform)
```

然后LightningModule里实现测试循环的方法
```python
class LitAutoEncoder(L.LightningModule):
    def training_step(self, batch, batch_idx):
        ...

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = F.mse_loss(x_hat, x)
        self.log("test_loss", test_loss)
```
一旦模型训练结束，我们即可调用`.test`
```
from torch.utils.data import DataLoader

# initialize the Trainer
trainer = Trainer()

# test the model
trainer.test(model, dataloaders=DataLoader(test_set))
```

### 验证循环
首先获得验证数据集，惯例来说验证数据集是训练数据集的20%。不过这个量依据情况而定。
```python
# use 20% of training data for validation
train_set_size = int(len(train_set) * 0.8)
valid_set_size = len(train_set) - train_set_size

# split the train set into two
seed = torch.Generator().manual_seed(42)
train_set, valid_set = data.random_split(train_set, [train_set_size, valid_set_size], generator=seed)
```
同样的在LightningModule里实现验证循环的方法

```python
class LitAutoEncoder(L.LightningModule):
    def training_step(self, batch, batch_idx):
        ...

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        val_loss = F.mse_loss(x_hat, x)
        self.log("val_loss", val_loss)
```

为了调用，只需要在`.fit`函数中添加测试数据集
```python
from torch.utils.data import DataLoader

train_loader = DataLoader(train_set)
valid_loader = DataLoader(valid_set)
model = LitAutoEncoder(...)

# train with both splits
trainer = L.Trainer()
trainer.fit(model, train_loader, valid_loader)
```

## 保存和加载模型

### Checkpoint
> It is a best practice to save the state of a model throughout the training process. This gives you a version of the model, a checkpoint, at each key point during the development of the model.

> Checkpoints also enable your training to resume from where it was in case the training process is interrupted.

Lightning Checkpoint包含了模型全部的内部状态。

> Unlike plain PyTorch, Lightning saves everything you need to restore a model even in the most complex distributed training environments.


Inside a Lightning checkpoint you’ll find:

- 16-bit scaling factor (if using 16-bit precision training)
- Current epoch
- Global step
- LightningModule’s state_dict
- State of all optimizers
- State of all learning rate schedulers
- State of all callbacks (for stateful callbacks)
- State of datamodule (for stateful datamodules)
- The hyperparameters (init arguments) with which the model was created
- The hyperparameters (init arguments) with which the datamodule was created
- State of Loops

#### 保存checkpoint
Lightning自动在当前的工作目录中保存checkpoint, 同时保存最后epoch的state，以便可以重启训练。
```python
# simply by using the Trainer you get automatic checkpointing
trainer = Trainer()
```

通过修改`default_root_dir`参数，修改checkpoint的存放目录
```python
# saves checkpoints to 'some/path/' at every epoch end
trainer = Trainer(default_root_dir="some/path/")
```

#### load checkpoint

为了加载LightningModule以及其权重和超参：
``` python
model = MyLightningModule.load_from_checkpoint("/path/to/checkpoint.ckpt")

# disable randomness, dropout, etc...
model.eval()

# predict with the model
y_hat = model(x)
```

#### 保存超参数
通过将超参传递给init函数并且在init函数里面调用`self.save_hyperparameters()`函数，LightningModule会自动保存所有的超参数。
```python
class MyLightningModule(LightningModule):
    def __init__(self, learning_rate, another_parameter, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
```

The hyperparameters are saved to the “hyper_parameters” key in the checkpoint

```python
checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
print(checkpoint["hyper_parameters"])
# {"learning_rate": the_value, "another_parameter": the_other_value}
```

The LightningModule also has access to the Hyperparameters
```python
model = MyLightningModule.load_from_checkpoint("/path/to/checkpoint.ckpt")
print(model.learning_rate)
```
#### load checkpoint到nn.Module
> Lightning checkpoints are fully compatible with plain torch nn.Modules.

Once the autoencoder has trained, pull out the relevant weights for your torch nn.Module:

```python
checkpoint = torch.load(CKPT_PATH)
encoder_weights = {k: v for k, v in checkpoint["state_dict"].items() if k.startswith("encoder.")}
decoder_weights = {k: v for k, v in checkpoint["state_dict"].items() if k.startswith("decoder.")}
```


#### 取消保存checkpoint
You can disable checkpointing by passing:

```python
trainer = Trainer(enable_checkpointing=False)
```

#### 继续训练
If you don’t just want to load weights, but instead restore the full training, do the following:
```python
model = LitModel()
trainer = Trainer()

# automatically restores model, epoch, step, LR schedulers, etc...
trainer.fit(model, ckpt_path="some/path/to/my_checkpoint.ckpt")
```


## Early Stopping
* [EarlyStopping](https://lightning.ai/docs/pytorch/stable/common/early_stopping.html)

### EarlyStopping Callback
`EarlyStopping` callback可以用来作为检测一个metric的改变，当没有新的提升出现时，停止训练模型。

为了启用`EarlyStopping`, 需要做以下步骤：
- Import EarlyStopping callback.
- Log the metric you want to monitor using log() method.
- Init the callback, and set monitor to the logged metric of your choice.
- Set the mode based on the metric needs to be monitored.
- Pass the EarlyStopping callback to the Trainer callbacks flag.

```python
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


class LitModel(LightningModule):
    def validation_step(self, batch, batch_idx):
        loss = ...
        self.log("val_loss", loss)


model = LitModel()
trainer = Trainer(callbacks=[EarlyStopping(monitor="val_loss", mode="min")])
trainer.fit(model)
```
You can customize the callbacks behaviour by changing its parameters.

```python
early_stop_callback = EarlyStopping(monitor="val_accuracy", min_delta=0.00, patience=3, verbose=False, mode="max")
trainer = Trainer(callbacks=[early_stop_callback])
```


## 迁移学习
> Any model that is a PyTorch nn.Module can be used with Lightning (because LightningModules are nn.Modules also).

### 使用预训练的LightningModule

```python
class Encoder(torch.nn.Module):
    ...


class AutoEncoder(LightningModule):
    def __init__(self):
        self.encoder = Encoder()
        self.decoder = Decoder()


class CIFAR10Classifier(LightningModule):
    def __init__(self):
        # init the pretrained LightningModule
        self.feature_extractor = AutoEncoder.load_from_checkpoint(PATH)
        self.feature_extractor.freeze()

        # the autoencoder outputs a 100-dim representation and CIFAR-10 has 10 classes
        self.classifier = nn.Linear(100, 10)

    def forward(self, x):
        representations = self.feature_extractor(x)
        x = self.classifier(representations)
        ...
```
We used our pretrained Autoencoder (a LightningModule) for transfer learning!

### Example: Imagenet
```python
import torchvision.models as models


class ImagenetTransferLearning(LightningModule):
    def __init__(self):
        super().__init__()

        # init a pretrained resnet
        backbone = models.resnet50(weights="DEFAULT")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        self.feature_extractor.eval()

        # use the pretrained model to classify cifar-10 (10 image classes)
        num_target_classes = 10
        self.classifier = nn.Linear(num_filters, num_target_classes)

    def forward(self, x):
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        ...

model = ImagenetTransferLearning()
trainer = Trainer()
trainer.fit(model)
```

And use it to predict your data of interest
```python
model = ImagenetTransferLearning.load_from_checkpoint(PATH)
model.freeze()

x = some_images_from_cifar10()
predictions = model(x)
```

> We used a pretrained model on imagenet, finetuned on CIFAR-10 to predict on CIFAR-10. In the non-academic world we would finetune on a tiny dataset you have and predict on your dataset.

## 设置超参数
### ArgumentParser
ArgumentParser 是 Python 中的一项内置功能，可让您构建 CLI 程序。您可以使用它从命令行提供超参数和其他训练设置：
```python
from argparse import ArgumentParser

parser = ArgumentParser()

# Trainer arguments
parser.add_argument("--devices", type=int, default=2)

# Hyperparameters for the model
parser.add_argument("--layer_1_dim", type=int, default=128)

# Parse the user inputs and defaults (returns a argparse.Namespace)
args = parser.parse_args()

# Use the parsed arguments in your program
trainer = Trainer(devices=args.devices)
model = MyModel(layer_1_dim=args.layer_1_dim)
```
可以这样去调用
`python trainer.py --layer_1_dim 64 --devices 1`

### LightningCLI
* [LightningCLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_intermediate.html)


## Debug model

> The Lightning Trainer has a lot of arguments devoted to maximizing your debugging productivity.

### Run all your model code once quickly
if you’ve ever trained a model for days only to crash during validation or testing then this trainer argument is about to become your best friend.

The fast_dev_run argument in the trainer runs 5 batch of training, validation, test and prediction data through your trainer to see if there are any bugs:

```python
trainer = Trainer(fast_dev_run=True)
```


To change how many batches to use, change the argument to an integer. Here we run 7 batches of each:
```python
trainer = Trainer(fast_dev_run=7)
```
This argument will disable tuner, checkpoint callbacks, early stopping callbacks, loggers and logger callbacks like LearningRateMonitor and DeviceStatsMonitor.

### Shorten the epoch length
Sometimes it’s helpful to only use a fraction of your training, val, test, or predict data (or a set number of batches). For example, you can use 20% of the training set and 1% of the validation set.

On larger datasets like Imagenet, this can help you debug or test a few things faster than waiting for a full epoch.

```python
# use only 10% of training data and 1% of val data
trainer = Trainer(limit_train_batches=0.1, limit_val_batches=0.01)

# use 10 batches of train and 5 batches of val
trainer = Trainer(limit_train_batches=10, limit_val_batches=5)
```
### Print LightningModule weights summary
Whenever the .fit() function gets called, the Trainer will print the weights summary for the LightningModule.

```shell
  | Name  | Type        | Params
----------------------------------
0 | net   | Sequential  | 132 K
1 | net.0 | Linear      | 131 K
2 | net.1 | BatchNorm1d | 1.0 K
```

## 可视化metrics

### Track metrics
Metric visualization is the most basic but powerful way of understanding how your model is doing throughout the model development process.

To track a metric, simply use the self.log method available inside the LightningModule

```python
class LitModel(L.LightningModule):
    def training_step(self, batch, batch_idx):
        value = ...
        self.log("some_value", value)
```
To log multiple metrics at once, use self.log_dict
```python
values = {"loss": loss, "acc": acc, "metric_n": metric_n}  # add more items if needed
self.log_dict(values)
```

### View in the commandline
To view metrics in the commandline progress bar, set the prog_bar argument to True.
```python
self.log(..., prog_bar=True)

Epoch 3:  33%|███▉        | 307/938 [00:01<00:02, 289.04it/s, loss=0.198, v_num=51, acc=0.211, metric_n=0.937]
```

### 在浏览器中展示
To view metrics in the browser you need to use an experiment manager with these capabilities.

By Default, Lightning uses Tensorboard (if available) and a simple CSV logger otherwise.

To launch the tensorboard dashboard run the following command on the commandline.
```python
tensorboard --logdir=lightning_logs/
```

If you’re using a notebook environment such as colab or kaggle or jupyter, launch Tensorboard with this command
```python
%reload_ext tensorboard
%tensorboard --logdir=lightning_logs/
```


## Load model and predict

```python
model = LitModel.load_from_checkpoint("best_model.ckpt")
model.eval()
x = torch.randn(1, 64)

with torch.no_grad():
    y_hat = model(x)
```

### 使用LightningModule
可以在predict_step实现函数

```python
class MyModel(LightningModule):
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

data_loader = DataLoader(...)
model = MyModel()
trainer = Trainer()
predictions = trainer.predict(model, data_loader)
```
