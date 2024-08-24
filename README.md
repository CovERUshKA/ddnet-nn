# ddnet-nn

To compile you would need to install libtorch with cuda support.

Steps to build on Windows:
1) Install CUDA the same version that libtorch with cuda support is built for.
2) Install NVTX from CUDA 11.8.
3) Download libtorch with cuda support and add it to ddnet-libs
4) Call cmake so it will generate files to build
5) Link dependencies in Visual Studio for DDNet-Server:
    ## Libraries:
    D:\GitHub\ddnet-nn\ddnet-libs\libtorch-win-shared-with-deps-latest+cu\libtorch\lib\c10.lib \
    D:\GitHub\ddnet-nn\ddnet-libs\libtorch-win-shared-with-deps-latest+cu\libtorch\lib\kineto.lib \
    D:\GitHub\ddnet-nn\ddnet-libs\libtorch-win-shared-with-deps-latest+cu\libtorch\lib\caffe2_nvrtc.lib \
    D:\GitHub\ddnet-nn\ddnet-libs\libtorch-win-shared-with-deps-latest+cu\libtorch\lib\c10_cuda.lib \
    D:\GitHub\ddnet-nn\ddnet-libs\libtorch-win-shared-with-deps-latest+cu\libtorch\lib\torch.lib \
    D:\GitHub\ddnet-nn\ddnet-libs\libtorch-win-shared-with-deps-latest+cu\libtorch\lib\torch_cuda.lib \
    C:\Program Files\NVIDIA Corporation\NvToolsExt\lib\x64\nvToolsExt64_1.lib \
    D:\GitHub\ddnet-nn\ddnet-libs\libtorch-win-shared-with-deps-latest+cu\libtorch\lib\torch_cpu.lib \
    -INCLUDE:?warp_size@cuda@at@@YAHXZ \
    C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\lib\x64\cudart.lib
    ## Include directories
    C:/Program Files/NVIDIA Corporation/NvToolsExt \
    C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/include \
    D:\GitHub\ddnet-nn\ddnet-libs\libtorch-win-shared-with-deps-latest+cu\libtorch\include\torch\csrc\api\include \
    D:\GitHub\ddnet-nn\ddnet-libs\libtorch-win-shared-with-deps-latest+cu\libtorch\include
6) Now try to compile DDNet-Server.
