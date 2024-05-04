# Python


## 常用命令
### related to Machine Learning
* Check current env info

`python -m torch.utils.collect_env`
```shell
(pytorch-m1)  % python -m torch.utils.collect_env
<frozen runpy>:128: RuntimeWarning: 'torch.utils.collect_env' found in sys.modules after import of package 'torch.utils', but prior to execution of 'torch.utils.collect_env'; this may result in unpredictable behaviour
Collecting environment information...
PyTorch version: 2.4.0.dev20240424
Is debug build: False
CUDA used to build PyTorch: None
ROCM used to build PyTorch: N/A

OS: macOS 14.4.1 (arm64)
GCC version: Could not collect
Clang version: 15.0.0 (clang-1500.3.9.4)
CMake version: Could not collect
Libc version: N/A

Python version: 3.12.3 | packaged by Anaconda, Inc. | (main, Apr 19 2024, 11:44:52) [Clang 14.0.6 ] (64-bit runtime)
Python platform: macOS-14.4.1-arm64-arm-64bit
Is CUDA available: False
CUDA runtime version: No CUDA
CUDA_MODULE_LOADING set to: N/A
GPU models and configuration: No CUDA
Nvidia driver version: No CUDA
cuDNN version: No CUDA
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

CPU:
Apple M1

Versions of relevant libraries:
[pip3] numpy==1.26.4
[pip3] torch==2.4.0.dev20240424
[pip3] torchaudio==2.2.0.dev20240424
[pip3] torchvision==0.19.0.dev20240424
[conda] numpy                     1.26.4          py312h7f4fdc5_0  
[conda] numpy-base                1.26.4          py312he047099_0  
[conda] pytorch                   2.4.0.dev20240424        py3.12_0    pytorch-nightly
[conda] torchaudio                2.2.0.dev20240424       py312_cpu    pytorch-nightly
[conda] torchvision               0.19.0.dev20240424       py312_cpu    pytorch-nightly
```