# 教程

## Tensors 张量
> Tensors are a specialized data structure that are very similar to arrays and matrices. In PyTorch, we use tensors to encode the inputs and outputs of a model, as well as the model’s parameters.

tensors和numpy的多维数组很相似，区别是tensors可以运行在GPUs以及其他的硬件加速器上。

> In fact, tensors and NumPy arrays can often share the same underlying memory, eliminating the need to copy data.
> Tensors are also optimized for automatic differentiation (we’ll see more about that later in the Autograd section). 
### 代码可复现
```python
torch.manual_seed(42) # Setting the seed
```

### 创建tensor
1. Directly from data, data type会自动推断
```python
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
```
2. from Numpy Array
```python
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
```
3. from another tensor
新的张量会保存参数张量的特性,shape和datatype, 除非特别指定。

```python
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")
```
```python
Ones Tensor:
 tensor([[1, 1],
        [1, 1]])

Random Tensor:
 tensor([[0.4223, 0.1719],
        [0.3184, 0.2631]])
```
4. 给定张量的维度,生成随机或者固定值
常用方法：
- torch.zeros: 创建一个值全为0的张量
- torch.ones: 创建一个值全为1的张量
- torch.rand: 创建一个张量，其随机值在 0 和 1 之间均匀采样
- torch.randn: 创建一个张量，其中包含从平均值为 0、方差为 1 的正态分布中采样的随机值
- torch.arange: Creates a tensor containing the values 
- torch.Tensor (input list): Creates a tensor from the list elements you provide
```python
torch.arange(6)
tensor([0, 1, 2, 3, 4, 5])

x = torch.Tensor(2, 3, 4)
print(x.shape)
torch.Size([2, 3, 4])
```
shape是一个tuple代表张量的维度
```python
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")
```

### Tensor to Numpy, Numpy to Tensor
```python
np_arr = np.array([[1, 2], [3, 4]])
tensor = torch.from_numpy(np_arr)

tensor = torch.arange(4)
np_arr = tensor.numpy()
```
注意：当使用tensor.numpy()方法的时候需要确保tensor在CPU上，如果在GPU上，需要先调用.cpu()方法，即 `np_arr = tensor.cpu().numpy()`.

### 张量的属性
Tensor attributes describe their shape, datatype, and the device on which they are stored.
```python
>>> tensor = torch.rand(3,4)
>>> print(f"Shape of tensor: {tensor.shape}")
Shape of tensor: torch.Size([3, 4])
>>> print(f"Datatype of tensor: {tensor.dtype}")
Datatype of tensor: torch.float32
>>> print(f"Device tensor is stored on: {tensor.device}")
Device tensor is stored on: cpu
```

对于tensor的shape除了可以使用和numpy中.shape一样的方式，还可与你使用`.size()`方法
```python
size = x.size()
print("Size:", size)

dim1, dim2, dim3 = x.size()
print("Size:", dim1, dim2, dim3)

Size: torch.Size([2, 3, 4])
Size: 2 3 4
```


### 张量操作
Over 100 tensor operations, including arithmetic, linear algebra, matrix manipulation (transposing, indexing, slicing), sampling and more are comprehensively described [here](https://pytorch.org/docs/stable/torch.html).
所有的运算都可以在GPU上运行，这个速度比在CPU上快很多。默认情况下，张量在CPU上创建，如果需要移动到GPU上，需要使用`.to`方法。

`⚠️ 在跨deivces上复制较大的张量会严重降低速度和占用内存`

```python
# We move our tensor to the GPU if available
if torch.cuda.is_available():
  tensor = tensor.to('cuda')
```


* 修改tensor的shape
size为 (2,3) 的张量可以重新组织为具有相同数量元素的任何其他形状（例如大小为 (6) 或 (3,2)...的张量）。在 PyTorch 中，这个操作为`view`：
```python
x = torch.arange(6)
x = x.view(2, 3)
print("X", x)
X tensor([[0, 1, 2],
        [3, 4, 5]])

x = x.permute(1, 0) # Swapping dimension 0 and 1
print("X", x)
X tensor([[0, 3],
        [1, 4],
        [2, 5]])
```


* 拥有和numpy array一样的下标索引与切片
```python
tensor = torch.arange(12).view(3,4)
print(tensor)
print('First row: ',tensor[0]) # or tensor[0,:] tensor[0,...]
print('First column: ', tensor[:, 0]) # or tensor[..., 0]
print('Last column:', tensor[..., -1]) # or tensor[:, -1]
print('First two rows, last column', tensor[:2, -1]) 
print('Middle two rows', tensor[1:3, :]) 

tensor[:,1] = 0
print(tensor)

tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
First row:  tensor([0, 1, 2, 3])
First column:  tensor([0, 4, 8])
Last column: tensor([ 3,  7, 11])
First two rows, last column tensor([3, 7])
Middle two rows tensor([[ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
tensor([[ 0,  0,  2,  3],
        [ 4,  0,  6,  7],
        [ 8,  0, 10, 11]])
```






* 合并张量
可以使用`torch.cat`将一串张量按照给定的维度进行合并。或者用`torch.stack`将张量合并，但是会产生新的维度。

> torch.cat (张量拼接)\
> 功能：torch.cat用于将一系列张量沿着现有的维度进行拼接。\
> 参数：主要参数包括一系列的张量和拼接的维度（dim）。\
> 用法示例：如果你有两个形状为（2, 3）的张量，使用torch.cat沿着第一个维度（dim=0）拼接，结果张量的形状将是（4, 3）。如果沿着第二个维度（dim=1）拼接，结果张量的形状将是（2, 6）。

> torch.stack (张量堆叠) \
> 功能：torch.stack用于将一系列张量堆叠成一个新的张量，这会创建一个新的维度。\
> 参数：主要参数包括一系列的张量和堆叠的维度（dim）。\
> 用法示例：如果你有两个形状为（2, 3）的张量，使用torch.stack沿着新的维度（假设为dim=0）堆叠，结果张量的形状将是（2, 2, 3）。这里，每个原始张量完整地保留，并作为新维度的一个切片。

注意：使用`torch.stack`时，所有的张量都必须具有一致的size. 这是因为 `torch.stack` 通过在新的维度上堆叠这些张量来创建一个更高维度的张量。为了保证这个新创建的张量的每个维度都是均匀和规整的，每个被堆叠的张量都必须有相同的形状. 
对于 `torch.cat`（张量拼接）的使用，情况与 `torch.stack` 略有不同。在使用 `torch.cat` 时，输入张量**在被拼接的维度（dim 参数指定的维度）可以有不同的大小，但在其他所有维度上，它们必须具有相同的尺寸**。这样做是为了确保在非拼接维度上数据的一致性和整齐排列。

假设我们有两个二维张量，一个形状为 (3, 4)，另一个形状为 (3, 2)。 这两个张量可以沿着第二维（列）进行拼接，因为它们在第一维（行）上具有相同的大小。

```python
# 定义两个张量，第一维大小相同
x = torch.randn(3, 4)
y = torch.randn(3, 2)

# 沿第二维拼接这两个张量
z = torch.cat((x, y), dim=1)

print(z.shape)  # 输出：torch.Size([3, 6])
```
在这个例子中，由于 x 和 y 在第一维（行）上的大小相同，它们可以沿着第二维进行拼接，结果是一个新的张量，其形状是 (3, 6)，这个新形状的第二维大小是原来两个张量第二维大小的和。

### 算数运算
```python
# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)


# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
```
需要注意的是第三赋值方式，先根据目标张量创建具有相同维度的张量，然后将最后的结果赋值给他，通过`out=`这个参数。

* 单一值张量

> If you have a one-element tensor, for example by aggregating all values of a tensor into one value, you can convert it to a Python numerical value using item():

```python
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

12.0 <class 'float'>
```

* In-place operations
> Operations that store the result into the operand are called in-place. They are denoted by a _ suffix. For example: x.copy_(y), x.t_(), will change x.

```python
print(tensor, "\n")
tensor.add_(5)
print(tensor)

tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])

tensor([[6., 5., 6., 6.],
        [6., 5., 6., 6.],
        [6., 5., 6., 6.],
        [6., 5., 6., 6.]])
```

注意： In-place operations save some memory, but can be problematic when computing derivatives because of an immediate loss of history. Hence, their use is discouraged.




### tensors和numpy数组共享内存
从Tensors到numpy 可以使用numpy() 
```python
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")
```
t: tensor([1., 1., 1., 1., 1.]) \
n: [1. 1. 1. 1. 1.]

当改变tensors的值，numpy的值也对应改变。
```python
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")
```
t: tensor([2., 2., 2., 2., 2.]) \
n: [2. 2. 2. 2. 2.]

反过来也是一样，从numpy到tensors可以使用`torch.from_numpy()`
```python
n = np.ones(5)
t = torch.from_numpy(n)
```
```
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
```

t: tensor([2., 2., 2., 2., 2.], dtype=torch.float64)\
n: [2. 2. 2. 2. 2.]


## Datasets & DataLoaders
* Image datasets <https://pytorch.org/vision/stable/datasets.html>
* Text datasets <https://pytorch.org/text/stable/datasets.html>
* Audio dataset <https://pytorch.org/audio/stable/datasets.html>

> PyTorch provides two data primitives: torch.utils.data.DataLoader and torch.utils.data.Dataset that allow you to use pre-loaded datasets as well as your own data. 

> Dataset stores the samples and their corresponding labels, and DataLoader wraps an iterable around the Dataset to enable easy access to the samples.

PyTorch的库提供了一些预加载的数据集，比如FashionMNIST。这些数据集可以用来快速实现模型以及benchmark模型。

### 加载数据集
这里以加载TorchVision中的[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)为例.
> Fashion-MNIST is a dataset of Zalando’s article images consisting of 60,000 training examples and 10,000 test examples. Each example comprises a 28×28 grayscale image and an associated label from one of 10 classes.

We load the FashionMNIST Dataset with the following parameters:

* root is the path where the train/test data is stored,
* train specifies training or test dataset,
* download=True downloads the data from the internet if it’s not available at root.
* transform and target_transform specify the feature and label transformations

```
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
```
这里的torchvision提供了很多内置的datasets在torchvision.datasets的模块中。所有的数据集都是`torch.utils.data.Dataset`的子集，也就是说他们具有`__getitem__`和`__len__`的方法。因此他们能够作为参数传给`torch.utils.data.DataLoader`.

```
imagenet_data = torchvision.datasets.ImageNet('path/to/imagenet_root/')
data_loader = torch.utils.data.DataLoader(imagenet_data,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=args.nThreads)
```

### 遍历数据集
可以通过类似数组一样去下表访问对应的值 `training_data[index]`.
```
每一项是一个tuple，分别包括img的值和对应的label
img, label = training_data[sample_idx]
```


### 自定义dataset
> A custom Dataset class must implement three functions: __init__, __len__, and __getitem__. 

> Take a look at this implementation; the FashionMNIST images are stored in a directory img_dir, and their labels are stored separately in a CSV file annotations_file.

```
import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
```

`__init__`

> The __init__ function is run once when instantiating the Dataset object. We initialize the directory containing the images, the annotations file, and both transforms.

The labels.csv file looks like:
```
tshirt1.jpg, 0
tshirt2.jpg, 0
......
ankleboot999.jpg, 9
```

`__len__`

The __len__ function returns the number of samples in our dataset.

`__getitem__`

> The __getitem__ function loads and returns a sample from the dataset at the given index idx. Based on the index, it identifies the image’s location on disk, converts that to a tensor using read_image, retrieves the corresponding label from the csv data in self.img_labels, calls the transform functions on them (if applicable), and returns the tensor image and corresponding label in a tuple.

#### 使用DataLoader去训练
Dataset每次读取一个样本和其对应的label，但是常常在训练模型的时候，我们想以batch的方式去训练，并且在每一个epoch的时候reshuffle数据来降低过拟合的可能性。同时使用python的多线程来加速数据的读取。

```
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
```
#### 通过DataLoader去遍历

> We have loaded that dataset into the DataLoader and can iterate through the dataset as needed. Each iteration below returns a batch of train_features and train_labels (containing batch_size=64 features and labels respectively). Because we specified shuffle=True, after we iterate over all batches the data is shuffled 

```
# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")

Feature batch shape: torch.Size([64, 1, 28, 28])
Labels batch shape: torch.Size([64])
Label: 5
```


## Transforms
通过transforms来对数据进行操作，使其适合训练。所有的torchvision.datasets都具有两个参数，`transform`用于修改features, `target_transform`勇于修改labels.

* torchvision.transforms <https://pytorch.org/vision/stable/transforms.html>

比如FashionMNIST的features存储是以PIL图片格式，为了方便训练，我们需要normalized的张量格式，以及将labels存储为one-hot张量格式。为了处理最初的数据，我们使用`ToTensor`和`Lambda`去转变最初的数据.
```
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)
```

### ToTensor()
ToTensor将一个PIL图片或者NumPy数组转变成`FloatTensor`并且将图片的值**转变到[0., 1.]之间**。

### Lambda Transforms

> Lambda transforms apply any user-defined lambda function. Here, we define a function to turn the integer into a one-hot encoded tensor. It first creates a zero tensor of size 10 (the number of labels in our dataset) and calls scatter_ which assigns a value=1 on the index as given by the label y.

```
target_transform = Lambda(lambda y: 
torch.zeros(10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
```
torch.Tensor.scatter_ 是一个用于修改张量内容的 PyTorch 函数，通常用于高效地对张量的特定位置进行更新。这个函数是一个原地（in-place）操作，意味着它会直接修改调用它的张量，而不会创建新的张量。scatter_ 的用法可以有点复杂，但它非常有用，特别是在处理诸如构建索引或进行一些特殊形式的赋值时。

功能和参数
* scatter_ 的基本用途是根据索引从一个源张量中取值，并将这些值放置到当前张量的指定位置。其参数通常包括：
* dim：指定要在哪一个维度上进行操作。
* index：一个与原张量同形状的张量，包含了要更新的元素的索引。
* src：可以是一个与原张量同形状的张量或一个单一的标量值，表示要填充到指定位置的数据。

### transforms和transforms.v2
> Torchvision supports common computer vision transformations in the torchvision.transforms and torchvision.transforms.v2 modules. Transforms can be used to transform or augment data for training or inference of different tasks (image classification, detection, segmentation, video classification).

目前有两个版本，一个是torchvision.transforms另一个是torchvision.transforms.v2.

为什么有两个版本
* 向后兼容性：原始的 torchvision.transforms 被广泛使用，在许多项目和代码库中都有实现。直接替换或大幅修改它可能会破坏现有代码。因此，新的 v2 模块提供了一个平滑过渡的路径，同时保持旧版本的功能。
* 功能和设计改进：随着技术的发展和用户反馈的积累，torchvision 团队识别出了原有 transforms 的一些设计限制和性能瓶颈。v2 模块的引入允许他们在不影响现有用户的情况下，实现这些改进。

#### transforms v2
* [Getting started with transforms v2](https://pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_getting_started.html#sphx-glr-auto-examples-transforms-plot-transforms-getting-started-py)


## 构建神经网络模型

神经网络由模块(modules)/层(layers)所组成，`torch.nn`命名空间提供了所有的神经网络模块/层。每个模块都是`nn.Module`的子类。

### 获得硬件类型

```
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
```

### 定义Class

通过实现`nn.Module`子类来定义自己的神经网络模型，并且在`__init__`里面去初始化模型所需的模块。每个`nn.Module`子类需要在`forward`方法里面实现对输入数据的操作。

```
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
```

然后创建`NeuralNetwork`的实例，并将其移动到`device`上。
```
model = NeuralNetwork().to(device)
print(model)

NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
  )
)
```
注意：对于tensor.to()函数是返回一个copy的值，而nn.Module.to()的函数是一个in-place修改。

为了使用这个模型，我们将数据传递给模型。这个会执行模型的forward函数，注意不要直接传递数据给`model.forward()`.
```
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

Predicted class: tensor([7], device='cuda:0')
```
通过将结果传递给`nn.Softmax`模块获得最终的预测概率。

### 模型解读
以刚刚定义的模型为例，解读数据传入之后进行的操作。这里以一个minibatch为3的样本为例，其中每张图片的大小为28x28。
`input_image = torch.rand(3,28,28)`.

#### nn.Flatten
* [Flatten](https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html)
我们初始化 nn.Flatten 层，将每个 2D 28x28 图像转换为 784 个像素值的连续数组（batch的维度dim=0会被保留）。
```
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

torch.Size([3, 784])
```

`torch.nn.Flatten(start_dim=1, end_dim=-1)` 
- start_dim (int) – first dim to flatten (default = 1).
- end_dim (int) – last dim to flatten (default = -1).

```
>>> input = torch.randn(32, 1, 5, 5)
>>> # With default parameters
>>> m = nn.Flatten()
>>> output = m(input)
>>> output.size()
torch.Size([32, 25])
>>> # With non-default parameters
>>> m = nn.Flatten(0, 2)
>>> output = m(input)
>>> output.size()
torch.Size([160, 5])
```

#### nn.Linear
线性层是一个使用其存储的weights和bias对输入应用线性变换的模块。

```
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

torch.Size([3, 20])
```
`torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)`
- in_features (int) – size of each input sample
- out_features (int) – size of each output sample
- bias (bool) – If set to False, the layer will not learn an additive bias. Default: True

关于线性层的weight和bias的初始化，可以看[torch.nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)。


#### nn.ReLU
> Non-linear activations are what create the complex mappings between the model’s inputs and outputs.
在线性变换后应用以引入非线性，帮助神经网络学习各种现象。

在线性层之间使用[nn.ReLU](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html)非线性激活函数，同时还有其他类型的激活函数。
```
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")
```
`torch.nn.ReLU(inplace=False)`
* inplace (bool) – can optionally do the operation in-place. Default: False

#### nn.Sequential

`nn.Sequential`是模块的有序容器。数据按照定义的相同顺序传递通过所有模块。
```
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)
```


`torch.nn.Sequential` A sequential container.
```
# Using Sequential to create a small model. When `model` is run,
# input will first be passed to `Conv2d(1,20,5)`. The output of
# `Conv2d(1,20,5)` will be used as the input to the first
# `ReLU`; the output of the first `ReLU` will become the input
# for `Conv2d(20,64,5)`. Finally, the output of
# `Conv2d(20,64,5)` will be used as input to the second `ReLU`
model = nn.Sequential(
          nn.Conv2d(1,20,5),
          nn.ReLU(),
          nn.Conv2d(20,64,5),
          nn.ReLU()
        )

# Using Sequential with OrderedDict. This is functionally the
# same as the above code
model = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(1,20,5)),
          ('relu1', nn.ReLU()),
          ('conv2', nn.Conv2d(20,64,5)),
          ('relu2', nn.ReLU())
        ]))
```
- append(module)

module (nn.Module) – module to append

#### nn.Softmax
> The last linear layer of the neural network returns logits - raw values in [-infty, infty] - which are passed to the nn.Softmax module. The logits are scaled to values [0, 1] representing the model’s predicted probabilities for each class. dim parameter indicates the dimension along which the values must sum to 1.

```
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)
```

### 模型参数 Model Parameters
> Many layers inside a neural network are parameterized, i.e. have associated weights and biases that are optimized during training. Subclassing nn.Module automatically tracks all fields defined inside your model object, and makes all parameters accessible using your model’s parameters() or named_parameters() methods.

In this example, we iterate over each parameter, and print its size and a preview of its values.
```
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

Model structure: NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
  )
)

Layer: linear_relu_stack.0.weight | Size: torch.Size([512, 784]) | Values : tensor([[ 0.0273,  0.0296, -0.0084,  ..., -0.0142,  0.0093,  0.0135],
        [-0.0188, -0.0354,  0.0187,  ..., -0.0106, -0.0001,  0.0115]],
       device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.0.bias | Size: torch.Size([512]) | Values : tensor([-0.0155, -0.0327], device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.2.weight | Size: torch.Size([512, 512]) | Values : tensor([[ 0.0116,  0.0293, -0.0280,  ...,  0.0334, -0.0078,  0.0298],
        [ 0.0095,  0.0038,  0.0009,  ..., -0.0365, -0.0011, -0.0221]],
       device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.2.bias | Size: torch.Size([512]) | Values : tensor([ 0.0148, -0.0256], device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.4.weight | Size: torch.Size([10, 512]) | Values : tensor([[-0.0147, -0.0229,  0.0180,  ..., -0.0013,  0.0177,  0.0070],
        [-0.0202, -0.0417, -0.0279,  ..., -0.0441,  0.0185, -0.0268]],
       device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.4.bias | Size: torch.Size([10]) | Values : tensor([ 0.0070, -0.0411], device='cuda:0', grad_fn=<SliceBackward0>)
```

## Autograd
* [Automatic Differentiation with torch.autograd](https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)

## Optimization
训练模型是一个迭代过程；在每次迭代中，模型都会对输出进行猜测，计算其猜测的误差（损失），收集误差相对于其参数的导数，并使用梯度下降优化这些参数。
```
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()
```

### 超参数
* [Hyperparameter tuning with Ray Tune](https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html)
超参数是可调整的参数，可以控制模型优化过程。不同的[超参数](https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html)值会影响模型训练和收敛速度。


- Number of Epochs - the number times to iterate over the dataset
- Batch Size - the number of data samples propagated through the network before the parameters are updated
- Learning Rate - how much to update models parameters at each batch/epoch. Smaller values yield slow learning speed, while large values may result in unpredictable behavior during training.

```
learning_rate = 1e-3
batch_size = 64
epochs = 5
```

### Optimization Loop
> Once we set our hyperparameters, we can then train and optimize our model with an optimization loop. Each iteration of the optimization loop is called an **epoch**.

Each epoch consists of two main parts:
- The Train Loop - iterate over the training dataset and try to converge to optimal parameters.
- The Validation/Test Loop - iterate over the test dataset to check if model performance is improving.

#### loss function
* [MSELoss](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss)
* [NLLLoss](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss)

torch.nn.NLLLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean') 
> The negative log likelihood loss. It is useful to train a classification problem with C classes. \
> If provided, the optional argument weight should be a 1D Tensor assigning weight to each of the  classes. This is particularly useful when you have an unbalanced training set.
* [CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss)

torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', label_smoothing=0.0)

> The input is expected to contain the unnormalized logits for each class (which do not need to be positive or sum to 1, in general). 

当提供一些训练数据时，我们未经训练的网络可能不会给出正确的答案。损失函数衡量的是得到的结果与目标值的不相似程度，它是我们在训练时想要最小化的损失函数。为了计算损失，我们使用给定数据样本的输入进行预测，并将其与真实数据标签值进行比较。

> Common loss functions include nn.MSELoss (Mean Square Error) for regression tasks, and nn.NLLLoss (Negative Log Likelihood) for classification. nn.CrossEntropyLoss combines nn.LogSoftmax and nn.NLLLoss.

We pass our model’s output logits to `nn.CrossEntropyLoss`, which will normalize the logits and compute the prediction error.

```
# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()
```

#### Optimizer
* [torch.optim](https://pytorch.org/docs/stable/optim.html)

优化是调整模型参数以减少每个训练步骤中模型误差的过程。优化算法定义了如何执行此过程（在本例中我们使用随机梯度下降）。所有优化逻辑都封装在优化器对象中。这里，我们使用SGD优化器；此外，PyTorch 中还有许多不同的优化器，例如 ADAM 和 RMSProp，它们可以更好地处理不同类型的模型和数据。

> We initialize the optimizer by registering the model’s parameters that need to be trained, and passing in the learning rate hyperparameter.

```
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
```

Inside the training loop, optimization happens in three steps:
1. Call optimizer.zero_grad() to reset the gradients of model parameters. Gradients by default add up; to prevent double-counting, we explicitly zero them at each iteration.
2. Backpropagate the prediction loss with a call to loss.backward(). PyTorch deposits the gradients of the loss w.r.t. each parameter.
3. Once we have our gradients, we call optimizer.step() to adjust the parameters by the gradients collected in the backward pass.

### Full implementation
We define train_loop that loops over our optimization code, and test_loop that evaluates the model’s performance against our test data.


```
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```

We initialize the loss function and optimizer, and pass it to train_loop and test_loop. Feel free to increase the number of epochs to track the model’s improving performance.

```
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
```


## 保存和加载模型
在本节中，我们将了解如何持久保存模型状态、加载模型和运行模型预测。

### 保存和加载模型权重
PyTorch模型存储学习到的参数在一个内置的状态字典里 `state_dict`. 他们可以通过`torch.save`方法进行保存。

```
model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), 'model_weights.pth')
```

为了加载模型的权重，我们首先需要创立一个与对应的模型一样的模型实例，然后通过`load_state_dict()`方法去加载保存的模型参数。
```
model = models.vgg16() # we do not specify ``weights``, i.e. create untrained model
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()
```

注意：加载模型参数之后，如果需要进行推断，则需要将模型设置为eval()模式，这样会把dropout和BN层设置为evaluation模式。否则模型可能会产生不一致的结果。

### 直接保存模型以及其权重
我们也许想直接加载一整个模型以及其权重，而不是首先实例模型，然后加载参数。

通过将模型本身传入到`save`方法中，而不是`model.state_dict()`可以实现这个功能：
```
torch.save(model, 'model.pth')
```
加载模型时需要注意是通过另一个函数：
```
model = torch.load('model.pth')
```

## 代码复现

