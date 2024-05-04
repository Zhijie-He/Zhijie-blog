# Conda


## 安装
安装conda有多种选项，常见的安装方式有 Anaconda, Miniconda or Miniforge。

这里使用Anaconda为例，到Anaconda官网进行下载安装 <https://www.anaconda.com/>。


## 常见命令

conda cheat sheet <https://docs.conda.io/projects/conda/en/stable/user-guide/cheatsheet.html>


### 环境管理

1. 创建环境

Conda允许创建多个不同版本的环境，这样方便管理不同版本的python以及一些包的版本。 \
最简单的方式去创建环境是使用:\
`conda create -n <env-name>`\
如果想在创建环境的同时安装一些指定的包，可以通过这个命令：\
`conda create -n myenvironment python numpy pandas`\
 这个命令代表着创建一个名为`myenviroment`的环境的同时，安装`python numpy pandas`的包。 \
如果想要指定包的版本，比如安装特定的python版本，可以通过这个命令\
`conda create -n myenvironment python=3.9`

2. 查看当前的所有环境

`conda env list ` \
或者 \
`conda info --env` \
类似这样的输出, 这里的*代表目前选择的环境
```shell
(pytorch-lightning) ----- % conda env list
# conda environments:
#
base                     /opt/anaconda3
pytorch-lightning     *  /opt/anaconda3/envs/pytorch-lightning
```
如果想要切换环境，可以使用 `conda activate <env-name>`的方式，如果不指定`<env-name>` 则会默认切换到`base`环境。

3. 为环境安装包

为特定环境安装包，可以通过activate到指定环境后安装，也可以指定环境名进行安装。
```shell
# via environment activation
conda activate myenvironment
conda install matplotlib

# via command line option
conda install --name myenvironment matplotlib
```
如果想要指定安装包的来源，可以使用这个命令\ 
`conda install conda-forge::numpy`

如果不指定安装包的来源，conda将会从[默认的来源](https://docs.conda.io/projects/conda/en/stable/user-guide/configuration/settings.html#default-channels)去搜索。

4. 克隆环境

`conda create --clone <env-name> -n <new-env-name>`

5. 重命名

`conda rename -n <env-name> <new-env-name>`

6. 删除环境

`conda remove -n  <env-name>  --all`

7. 查看对环境的修改

`conda list -n <env-name> --revisions`

得到类似如下的输出
```
(pytorch-lightning) ---- % conda list -n pytorch-lightning --revisions
2024-04-24 14:32:06  (rev 0)

2024-04-24 14:43:48  (rev 1)
    +bzip2-1.0.8 (defaults/osx-arm64)
    +ca-certificates-2024.3.11 (defaults/osx-arm64)
    +expat-2.6.2 (defaults/osx-arm64)
    +libcxx-14.0.6 (defaults/osx-arm64)
    +libffi-3.4.4 (defaults/osx-arm64)
    +ncurses-6.4 (defaults/osx-arm64)
    +openssl-3.0.13 (defaults/osx-arm64)
    +pip-23.3.1 (defaults/osx-arm64)
    +python-3.12.3 (defaults/osx-arm64)
    +readline-8.2 (defaults/osx-arm64)
    +setuptools-68.2.2 (defaults/osx-arm64)
    +sqlite-3.41.2 (defaults/osx-arm64)
    +tk-8.6.12 (defaults/osx-arm64)
    +tzdata-2024a (defaults/noarch)
    +wheel-0.41.2 (defaults/osx-arm64)
    +xz-5.4.6 (defaults/osx-arm64)
    +zlib-1.2.13 (defaults/osx-arm64)
```
如果想要回溯修改，可以使用如下命令：
`conda install -n <env-name> --revision <number>`

```
(pytorch-lightning) ---- % conda install -n pytorch-lightning --revision 0
Collecting package metadata (current_repodata.json): done
Reverting to revision 0: done

## Package Plan ##

  environment location: /opt/anaconda3/envs/pytorch-lightning

  added / updated specs:
    - python


The following packages will be REMOVED:

  bzip2-1.0.8-h80987f9_5
  ca-certificates-2024.3.11-hca03da5_0
  expat-2.6.2-h313beb8_0
  libcxx-14.0.6-h848a8c0_0
  libffi-3.4.4-hca03da5_0
  ncurses-6.4-h313beb8_0
  openssl-3.0.13-h1a28f6b_0
  pip-23.3.1-py312hca03da5_0
  python-3.12.3-h99e199e_0
  readline-8.2-h1a28f6b_0
  setuptools-68.2.2-py312hca03da5_0
  sqlite-3.41.2-h80987f9_0
  tk-8.6.12-hb8d0fd4_0
  tzdata-2024a-h04d1e81_0
  wheel-0.41.2-py312hca03da5_0
  xz-5.4.6-h80987f9_0
  zlib-1.2.13-h5a0b063_0


Proceed ([y]/n)? y

Preparing transaction: done
Verifying transaction: done
Executing transaction: done
```

```
(pytorch-lightning) ---- % conda list -n pytorch-lightning --revisions
2024-04-24 14:32:06  (rev 0)

2024-04-24 14:43:48  (rev 1)
    +bzip2-1.0.8 (defaults/osx-arm64)
    +ca-certificates-2024.3.11 (defaults/osx-arm64)
    +expat-2.6.2 (defaults/osx-arm64)
    +libcxx-14.0.6 (defaults/osx-arm64)
    +libffi-3.4.4 (defaults/osx-arm64)
    +ncurses-6.4 (defaults/osx-arm64)
    +openssl-3.0.13 (defaults/osx-arm64)
    +pip-23.3.1 (defaults/osx-arm64)
    +python-3.12.3 (defaults/osx-arm64)
    +readline-8.2 (defaults/osx-arm64)
    +setuptools-68.2.2 (defaults/osx-arm64)
    +sqlite-3.41.2 (defaults/osx-arm64)
    +tk-8.6.12 (defaults/osx-arm64)
    +tzdata-2024a (defaults/noarch)
    +wheel-0.41.2 (defaults/osx-arm64)
    +xz-5.4.6 (defaults/osx-arm64)
    +zlib-1.2.13 (defaults/osx-arm64)

2024-04-24 14:59:01  (rev 2)
    -bzip2-1.0.8 (defaults/osx-arm64)
    -ca-certificates-2024.3.11 (defaults/osx-arm64)
    -expat-2.6.2 (defaults/osx-arm64)
    -libcxx-14.0.6 (defaults/osx-arm64)
    -libffi-3.4.4 (defaults/osx-arm64)
    -ncurses-6.4 (defaults/osx-arm64)
    -openssl-3.0.13 (defaults/osx-arm64)
    -pip-23.3.1 (defaults/osx-arm64)
    -python-3.12.3 (defaults/osx-arm64)
    -readline-8.2 (defaults/osx-arm64)
    -setuptools-68.2.2 (defaults/osx-arm64)
    -sqlite-3.41.2 (defaults/osx-arm64)
    -tk-8.6.12 (defaults/osx-arm64)
    -tzdata-2024a (defaults/noarch)
    -wheel-0.41.2 (defaults/osx-arm64)
    -xz-5.4.6 (defaults/osx-arm64)
    -zlib-1.2.13 (defaults/osx-arm64)
```

### 其他

* 检查conda的安装与信息

`conda info`