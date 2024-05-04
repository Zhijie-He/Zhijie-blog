# 应用

## MNIST和FashionMNIST

### 准备数据集
这里以MNIST和FashionMNIST数据集为例, 这两个数据集可以通过[torchvision.datasets](https://pytorch.org/vision/stable/datasets.html)下载，需要指定一些参数：

> torchvision.datasets.MNIST(root: Union[str, Path], train: bool = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False)

- root:(str) 数据集的所在根目录
- train:(bool, optional) True读取训练数据集，False读取测试数据集
- download: (bool, optional) True则从网上下载数据集，并放在根目录中，如果数据集已经在根目录，则不会重新下载
- transform: (callable, optional) 对x进行的一些转换操作，比如将PIL图片变成tensor存储
- target_transform: (callable, optional)对x进行的一些转换操作, 比如转换成one hot存储

```python
import torchvision
FashionMNIST_dataset = torchvision.datasets.FashionMNIST(root='data', train=True,  download=True)
MNIST_dataset = torchvision.datasets.MNIST(root='data', train=True,  download=True)
print(FashionMNIST_dataset[0])
print(MNIST_dataset[0])

(<PIL.Image.Image image mode=L size=28x28 at 0x110D83080>, 9)
(<PIL.Image.Image image mode=L size=28x28 at 0x110D83080>, 7)
```
可以看出在torchvision.datasets中，每一个数据是以tuple的方式进行存储，即(PIL图片, class). 

PyTorch模型训练期待的训练数据是tensor类型，所以这里针对图片需要进行转换。
```python
to_tensor_transform = transforms.ToTensor()
print(to_tensor_transform(FashionMNIST_dataset[0][0]).shape)
print(to_tensor_transform(MNIST_dataset[0][0]).shape)

torch.Size([1, 28, 28])
torch.Size([1, 28, 28])
```

在下载图片的时候，提供了transform的选项。
```python
import torchvision
from torchvision import transforms
FashionMNIST_dataset = torchvision.datasets.FashionMNIST(root='data', train=True,  download=True, transform=transforms.ToTensor())
MNIST_dataset = torchvision.datasets.MNIST(root='data', train=True,  download=True, transform=transforms.ToTensor())
print(FashionMNIST_dataset[0][0].shape, type(FashionMNIST_dataset[0][0]))

torch.Size([1, 28, 28]) <class 'torch.Tensor'>
```

#### 对数据集进行归一化处理
目前PIL图片通过ToTensor()函数进行处理，这里ToTensor不仅将数据从PIL图片格式变为tensor存储，同时将图片的值从 [0, 255] 范围自动缩放到 [0.0, 1.0] 的浮点数范围。
>  Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1) or if the numpy.ndarray has dtype = np.uint8

然而，虽然 ToTensor() 转换确实将数据归一化到了 [0, 1]，这个区间的数据通常仍不是最理想的输入形式。大多数现代神经网络期望输入数据是以 0 为中心（zero-centered）的，这意味着数据的平均值应当接近 0。这种进一步的标准化通常有助于网络更有效地学习（特别是深层网络），因为它有助于保持激活函数输出的非线性。

为什么需要进一步归一化？

归一化变换会改变数据的平均值和标准差，目标是使数据集的平均值接近 0 且标准差接近 1。


- 零中心化（Zero-centering）：
尽管 ToTensor() 将数据缩放到了 [0, 1]，但这不意味着数据的平均值是 0.5。因此，通过减去均值，可以进一步将数据的均值移到接近 0 的位置，这通常称为零中心化。
对数据进行零中心化有助于神经网络权重的更新更加稳定，因为它确保了网络各层激活函数输入的分布更加对称和均匀。
- 归一化方差（Unit Variance）：
除以标准差是为了将数据的方差归一化到 1。这有助于保持网络训练过程中的数值稳定性，避免某些层输出过大或过小，从而导致梯度消失或梯度爆炸。
让所有输入特征的标准差保持一致，有助于优化算法更有效地收敛。

参数的选择
```python
"""Normalize a tensor image with mean and standard deviation.
This transform does not support PIL Image.
Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
channels, this transform will normalize each channel of the input
``torch.*Tensor`` i.e.,
``output[channel] = (input[channel] - mean[channel]) / std[channel]``

.. note::
    This transform acts out of place, i.e., it does not mutate the input tensor.

Args:
    mean (sequence): Sequence of means for each channel.
    std (sequence): Sequence of standard deviations for each channel.
    inplace(bool,optional): Bool to make this operation in-place.

"""
```


在 transforms.Normalize() 中使用的参数 (mean, std) 是特定于数据集的。这些参数用于将数据的每个通道规范化到零均值和单位方差：

- mean：用于从每个通道中减去的均值。
- std：每个通道分别除以的标准差。
对于 MNIST 和 FashionMNIST 数据集：

- MNIST 数据集中，像素值的全局平均约为 0.1307，标准差约为 0.3081。
- FashionMNIST 数据集常用的归一化参数为 0.5，0.5，

```python
# Choose the dataset based on the dataset_type parameter
FashionMNIST_dataset = torchvision.datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # FashionMNIST 数据集的均值和标准差
        ]
    ),
)

MNIST_dataset = torchvision.datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
)
```
这里的MNIST和FashionMNIST只有一个通道。 注意要使用仅从训练数据集计算得出的归一化参数（即均值和标准差）。这种做法确保了测试数据在模型评估阶段的处理方式与训练数据保持一致，同时防止了潜在的数据泄露问题。

验证结果
```python
# 定义 DataLoader
loader = DataLoader(MNIST_dataset, batch_size=64, shuffle=False)

def calculate_mean_std(loader):
    mean = 0.0
    std = 0.0
    total_images_count = 0

    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count

    return mean, std

# 计算归一化后数据的平均值和标准差
mean, std = calculate_mean_std(loader)
print(f"Normalized data mean: {mean}")
print(f"Normalized data std: {std}")

Normalized data mean: tensor([-0.0001])
Normalized data std: tensor([0.9786])
```
可以看到，这样的归一化处理结果是符合预期的。


#### 使用DataLoader
使用DataLoader我们可以分批(batch)进行训练而不是一个一个样本的去训练，这样可以提高时间，还有助于改善模型的性能。

```python
from torch.utils.data import DataLoader

def get_data_loader(dataset_root_dir, val_percent, batch_size, dataset_name='MNIST'):

    # Choose the dataset based on the dataset_type parameter
    if dataset_name == 'FashionMNIST':
        Dataset = torchvision.datasets.FashionMNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # FashionMNIST 数据集的均值和标准差
        ])
    else:
        Dataset = torchvision.datasets.MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    train_dataset = Dataset(
        root=dataset_root_dir, train=True, download=True, transform=transform)
    test_dataset = Dataset(
        root=dataset_root_dir, train=False,  download=True,  transform=transform)

    # Percentage of validation data
    N_val_samples = round(val_percent * len(train_dataset))

    # Split into two subsets
    train_set, val_set = torch.utils.data.random_split(
        train_dataset, [len(train_dataset) - N_val_samples, N_val_samples])

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=4)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=4)

    return train_loader, val_loader, test_loader
```

这里使用`torch.utils.data.DataLoader`创建loader, 一些参数进行设置：
- *dataset*: 这是一个数据集对象，需要是一个实现了 __getitem__ 和 __len__ 方法的可迭代对象，通常是 PyTorch 的 Dataset 类的子类。这个参数指定了 DataLoader 从哪个数据集中加载数据。
- *batch_size*: 默认1, 每一批多少个样本, 这直接影响到模型每次接收并处理数据的数量。较大的 batch_size 可以提高内存利用率和处理速度，但也可能影响模型训练的收敛行为和最终性能。
- *shuffle*: 是否在每个训练周期（epoch）开始时随机打乱数据, 这有助于模型泛化，防止模型过拟合到数据加载顺序的特定模式。
- *num_workers*:  指定有多少个子线程用于数据加载。更多的 num_workers 可以加速数据加载过程，特别是在处理大规模数据集和进行复杂转换时。0表示使用主线程进行加载数据操作
- *drop_last*: 当数据集中的样本总数不能被 batch_size 整除时，是否丢弃最后一个不完整的批次。这在某些情况下很有用，特别是在使用批标准化（Batch Normalization）时，因为小批量可能会导致统计估计不准确。
- *pin_memory*: 当设置为 True 时，pin_memory 参数会在将数据传递给 GPU 之前，先将数据加载到 CPU 的固定（锁页）内存中。这通常可以减少将数据从 CPU 转移到 GPU 时的时间，因为锁页内存的数据复制到 GPU 的速度更快。 适用场景：在使用 CUDA 加速的深度学习训练中，设置 pin_memory=True 可以提高数据传输效率，加快训练过程。但如果不使用 GPU，这个选项则没有影响。

### 创建LeNet5模型
```python
from torch import nn

class LetNet5(nn.Module):

  def __init__(self):
    super(LetNet5, self).__init__()
    ## MNIST images are 28x28 but LeNet5 expects 32x32
    ## -> we pad the images with zeroes
    self.padding = nn.ZeroPad2d(2)
    ## First convolution
    self.conv1 = nn.Conv2d(in_channels = 1, out_channels= 6 , kernel_size = 5)
    ## Second convolution
    self.conv2 = nn.Conv2d(6,16,5)
    ## Pooling (subsampling) layer
    self.maxpool = nn.MaxPool2d(2)
    ## Activation layer
    self.relu = nn.ReLU()
    ## Fully connected layers
    self.fc1 = nn.Linear(in_features = 400, out_features = 120)
    self.fc2 = nn.Linear(120, 84)
    self.output = nn.Linear(84, 10)
    ## Final activation layer
    self.softmax = nn.LogSoftmax(dim=1)

  def forward(self, x):
    ## Pad the input
    x = self.padding(x)
    ## First convolution + activation
    x = self.conv1(x)
    x = self.relu(x)
    ## First pooling
    x = self.maxpool(x)
    ## Second Convolution + activation
    x = self.conv2(x)
    x = self.relu(x)
    ## Second Pooling
    x = self.maxpool(x)
    ## "Flatten" the output to make it 1D
    x = x.view(-1, 16*5*5)
    ## First full connection
    x = self.fc1(x)
    x = self.relu(x)
    ## Second full connection
    x = self.fc2(x)
    x = self.relu(x)
    ## Output layer
    x = self.output(x)
    y = self.softmax(x)
    return y
```

### 训练模型
设置一些模型必须的参数，比如损失函数，优化器
```python
# Create an instance of our network and move it to device
model = LetNet5().to(device)

# Negative log likelihood loss
loss_fn = nn.NLLLoss()

# Stochastic Gradient Descent
optimizer = torch.optim.SGD(model.parameters(), lr=hparams.lr, momentum=hparams.momentum)
```

在这里将训练，验证，测试写在同一个函数里面
```python
def run_epoch(loader, model, loss_fn, optimizer, device, mode='train'):
    if mode == 'train':
        model.train()
    else:
        model.eval()

    running_loss, running_accuracy = 0.0, 0.0
    total_samples = len(loader.dataset)

    for i, (x, labels) in enumerate(tqdm(loader, desc=f"{mode.capitalize()} Epoch")):
        x, labels = x.to(device), labels.to(device)

        # Forward pass
        with torch.set_grad_enabled(mode == 'train'):
            outputs = model(x)
            loss = loss_fn(outputs, labels)

        # Backward and optimize
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Calculate statistics
        with torch.no_grad():
            running_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            running_accuracy += (predictions ==
                                 labels).type(torch.float).sum().item()

    average_loss = running_loss / total_samples
    accuracy = running_accuracy / total_samples

    print(f"{mode.capitalize()} Loss: {average_loss}, Accuracy: {accuracy}")

    return average_loss, accuracy
```

开始训练

```python
for epoch in range(hparams.num_epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")
    train_loss, train_acc = run_epoch(
        train_loader, model, loss_fn, optimizer, device, mode='train')
    val_loss, val_acc = run_epoch(
        val_loader, model, loss_fn, optimizer, device, mode='eval')
```
这里因为MNIST和FashionMNIST的数据格式是一致的,所以可以直接训练,完整代码。