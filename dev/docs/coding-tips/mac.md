# MacOS

1. 检测当前硬件详细信息 `system_profiler SPDisplaysDataType`
```shell
(base) zhijiehe@zhijiedeAir coding-tips % system_profiler SPDisplaysDataType
Graphics/Displays:

    Apple M1:

      Chipset Model: Apple M1
      Type: GPU
      Bus: Built-In
      Total Number of Cores: 7
      Vendor: Apple (0x106b)
      Metal Support: Metal 3
      Displays:
        LG ULTRAWIDE:
          Resolution: 2560 x 1080 (UW-UXGA - Ultra Wide - Ultra Extended Graphics Array)
          UI Looks like: 2560 x 1080 @ 75.00Hz
          Main Display: Yes
          Mirror: Off
          Online: Yes
          Rotation: Supported
        Color LCD:
          Display Type: Built-In Retina LCD
          Resolution: 2560 x 1600 Retina
          Mirror: Off
          Online: Yes
          Automatically Adjust Brightness: Yes
          Connection Type: Internal
```

2. MPS和GPU的区别

关于MPS（Metal Performance Shaders）和GPU数量的概念，有必要区分一下两者之间的区别：

  - GPU（图形处理单元）:
    GPU是硬件设备，具体指的是电脑、手机或其他设备中用于处理图形和计算任务的实体。在一台设备上，可以有一个或多个GPU。例如，一些高端的计算机或专业的图形工作站可能装有多个独立的GPU卡，用于高性能计算或图形处理。
  - MPS（Metal Performance Shaders）:
    MPS并不是一个硬件设备，而是一套运行在Apple设备上，用于高效执行图形和计算任务的库和API。MPS设计来利用Apple设备上的GPU，优化和加速各种操作，比如图像处理、机器学习等。
  
  MPS的“数量”并不像GPU那样计量。你不会说一个设备上有多少个MPS，因为MPS只是软件层面的技术实现。你可能会关注的是设备上有多少GPU，以及这些GPU是否支持MPS，以及支持到什么程度。


## Mac的命令行配置
- [工具 - 打造 Mac “完美”终端（Terminal），一篇就够了](https://makeoptim.com/tool/terminal/)
- [Mac OS 命令行终端工具iTerm2 + Oh my Zsh的安装配置](https://blog.csdn.net/zangxueyuan88/article/details/113937379)

### 安装iTerm2

去这个网址<https://iterm2.com/>进行安装即可

- iTerm2 是一款完全免费，专为 Mac OS 用户打造多命令行应用。
- 安装完成后，在/bin目录下会多出一个zsh的文件。
- Mac系统默认使用dash作为终端，可以使用命令修改默认使用zsh：chsh -s /bin/zsh
- 如果想修改回默认dash，同样使用chsh命令即可：chsh -s /bin/bash
- Zsh 是一款强大的虚拟终端，既是一个系统的虚拟终端，也可以作为一个脚本语言的交互解析器。
#### 设置iTerm2的的背景图
点击iTerm2>settings>profiles，然后点击window这个选项就可以自定义背景图片。
其次如果想要一张图片作为所有panel的背景图，则在iTerm2>settings>Appearence然后点击Panes，取消选择Sepearate background images per pane.



#### 设置iTerm2的主题
- [iterm2的主题](https://github.com/ohmyzsh/ohmyzsh/wiki/Themes)

这个网站上有他对应的主题，找到想要的主题然后修改`~/.zshrc`文件里面的主题配置即可

#### 使用Powerlevel10k主题
Oh My Zsh 有上百个自带主题，以及其他的外部主题。而 Powerlevel10k 正是现在最流行的主题之一。

执行以下命令，安装 Powerlevel10k。

```shell
git clone --depth=1 https://gitee.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/themes/powerlevel10k
```

在 zsh 的配置文件 ~/.zshrc 中设置 `ZSH_THEME="powerlevel10k/powerlevel10k"`

设置完成后，重启 iTerm2 会提示安装需要的字体，根据提示安装即可。

如果对安装主题配置不满意，可以重新配置主题，在命令行中输入`p10k configure`。


### 安装[on my zsh](https://ohmyz.sh/) 
Oh My Zsh 是一款社区驱动的命令行工具，它基于 zsh 命令行，提供了主题配置，插件机制，已经内置的便捷操作。给我们一种全新的方式使用命令行。

> Oh My Zsh is a delightful, open source, community-driven framework for managing your Zsh configuration. It comes bundled with thousands of helpful functions, helpers, plugins, themes, and a few things that make you shout...

如果要安装on my zsh可以通过以下的命令
```shell
通过curl
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
or 
通过wget
sh -c "$(wget https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh -O -)"
```

注意： 安装完成on my zsh之后，发现conda命令访问不到了
- [on my zsh找不到conda](https://zhuanlan.zhihu.com/p/158703094)
> 终端中zsh的可访问的程序一般放在/bin, /usr/bin, /usr/local/bin，~/bin目录下；而最新安装的Anaconda会默认安装在/Users/username下或者/opt下，导致环境变量没有写入到终端配置文件。笔者的Anaconda默认被安装在了~/opt目录下，直接采用网络上的代码行不通，需要改一下路径。

解决办法：找到anacoda的路径，我的anaconda3的安装路径在/opt/anaconda3下面。所以进行修改~/.zshrc

```shell
#切记先返回跟目录
cd ~
#vim打开zsh配置文件
vi .zshrc
#添加指令
export PATH="/opt/anaconda3/bin:$PATH"
#激活配置文件
source .zshrc
```

激活之后发现可以找到conda但是切换环境还需要`conda init zsh`，这样做之后记得`source .zshrc`这个文件之后，就发现conda的默认环境就激活了。


### 安装语法高亮插件
下载插件
```shell
$ cd ~/.oh-my-zsh/custom/plugins
$ git clone https://github.com/zsh-users/zsh-syntax-highlighting.git
```
启用插件
```shell
$ vi ~/.zshrc
...
plugins=(
  git
  zsh-syntax-highlighting
)
...

$ source ~/.zshrc
```

### 自动补全插件
下载插件
```shell
$ cd ~/.oh-my-zsh/custom/plugins
$ git clone https://github.com/zsh-users/zsh-autosuggestions.git
```
启用插件
```shell
$ vi ~/.zshrc
...
plugins=(
  git
  zsh-syntax-highlighting
  zsh-autosuggestions
)
...

$ source ~/.zshrc
```

### 自动跳转插件 autojump
autojump 可以记录下之前 cd 命令访过的所有目录，下次要去那个目录时不需要输入完整的路径，直接 j somedir 即可到达，甚至那个目标目录的名称只输入开头即可。

执行以下命令，安装 autojump。
```shell
brew install autojump
```
在 zsh 的配置文件 ~/.zshrc 中的 plugins 中加入 autojump
启用插件
```shell
$ vi ~/.zshrc
...
plugins=(
  git
  zsh-syntax-highlighting
  zsh-autosuggestions
  autojump
)
...

$ source ~/.zshrc
```

### 设置VS code字体
默认情况下，在 VSCode 中选择 zsh 作为默认 Shell 会出现乱码现象。原因是 Oh My Zsh 配置完成后，使用了 `MesloLGS NF` 字体。

因此，修复乱码只需要在设置中找到 code > preferences > settings, 输入font，设置**terminal font**成 `MesloLGS NF` 即可。



## iTerm2 快捷键

- crtl + d 垂直分屏
- crtl + shift + d 水平分屏
- command + f 搜索&查找，如果输入搜索内容后， shift+tab，则自动将查找内容的左边选中并复制。按 esc 退出搜索。
- command + r 或 ctrl + l 清空屏幕，而且只是换到新一屏，不会像 clear 一样创建一个空屏
