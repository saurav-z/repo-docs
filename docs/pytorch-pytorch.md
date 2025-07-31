<div align="center">
  <a href="https://github.com/pytorch/pytorch">
    <img src="https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/pytorch-logo-dark.png" alt="PyTorch Logo" width="200"/>
  </a>
</div>

# PyTorch: The Open Source Deep Learning Framework

**PyTorch empowers researchers and developers to build cutting-edge machine learning models with speed, flexibility, and efficiency.**

## Key Features

*   **Tensor Computation with GPU Acceleration:** Leverage the power of GPUs for fast, efficient tensor operations, similar to NumPy but with added speed benefits.
*   **Dynamic Neural Networks:** Benefit from PyTorch's tape-based autograd system, enabling dynamic neural networks for maximum flexibility and research exploration.
*   **Python-First Design:** Seamlessly integrate PyTorch into your existing Python workflows and extend its functionality with popular libraries like NumPy and SciPy.
*   **Imperative Programming:** Experience intuitive and straightforward code execution, simplifying debugging and enhancing development speed.
*   **Fast and Lean:** Enjoy minimal framework overhead and optimized performance, with support for acceleration libraries like Intel MKL, cuDNN, and NCCL.
*   **Easy Extensibility:** Easily create new neural network modules and interface with PyTorch's Tensor API, supported by a straightforward extension API for C/C++ layers.

## Getting Started

*   [Tutorials](https://pytorch.org/tutorials/)
*   [Examples](https://github.com/pytorch/examples)
*   [API Reference](https://pytorch.org/docs/)

## Table of Contents

*   [More About PyTorch](#more-about-pytorch)
    *   [A GPU-Ready Tensor Library](#a-gpu-ready-tensor-library)
    *   [Dynamic Neural Networks: Tape-Based Autograd](#dynamic-neural-networks-tape-based-autograd)
    *   [Python First](#python-first)
    *   [Imperative Experiences](#imperative-experiences)
    *   [Fast and Lean](#fast-and-lean)
    *   [Extensions Without Pain](#extensions-without-pain)
*   [Installation](#installation)
    *   [Binaries](#binaries)
        *   [NVIDIA Jetson Platforms](#nvidia-jetson-platforms)
    *   [From Source](#from-source)
        *   [Prerequisites](#prerequisites)
            *   [NVIDIA CUDA Support](#nvidia-cuda-support)
            *   [AMD ROCm Support](#amd-rocm-support)
            *   [Intel GPU Support](#intel-gpu-support)
        *   [Get the PyTorch Source](#get-the-pytorch-source)
        *   [Install Dependencies](#install-dependencies)
        *   [Install PyTorch](#install-pytorch)
            *   [Adjust Build Options (Optional)](#adjust-build-options-optional)
    *   [Docker Image](#docker-image)
        *   [Using pre-built images](#using-pre-built-images)
        *   [Building the image yourself](#building-the-image-yourself)
    *   [Building the Documentation](#building-the-documentation)
        *   [Building a PDF](#building-a-pdf)
    *   [Previous Versions](#previous-versions)
*   [Getting Started](#getting-started)
*   [Resources](#resources)
*   [Communication](#communication)
*   [Releases and Contributing](#releases-and-contributing)
*   [The Team](#the-team)
*   [License](#license)

## More About PyTorch

[Learn the basics of PyTorch](https://pytorch.org/tutorials/beginner/basics/intro.html)

PyTorch is a comprehensive library with several key components:

| Component             | Description                                                                                                    |
| --------------------- | -------------------------------------------------------------------------------------------------------------- |
| **torch**             | A Tensor library with GPU support, similar to NumPy. [torch](https://pytorch.org/docs/stable/torch.html) |
| **torch.autograd**    | An automatic differentiation library supporting differentiable Tensor operations. [torch.autograd](https://pytorch.org/docs/stable/autograd.html) |
| **torch.jit**         | A compilation stack (TorchScript) for serializable and optimizable models.  [torch.jit](https://pytorch.org/docs/stable/jit.html)  |
| **torch.nn**          | A neural networks library designed for maximum flexibility. [torch.nn](https://pytorch.org/docs/stable/nn.html) |
| **torch.multiprocessing** | Python multiprocessing with memory sharing of torch Tensors. [torch.multiprocessing](https://pytorch.org/docs/stable/multiprocessing.html)   |
| **torch.utils**       | DataLoader and utility functions. [torch.utils](https://pytorch.org/docs/stable/data.html)  |

PyTorch is typically used for:

*   Replacing NumPy with GPU-accelerated operations.
*   Building a flexible deep learning research platform.

### A GPU-Ready Tensor Library

PyTorch provides Tensors that can be deployed on either the CPU or the GPU, accelerating computations significantly.

![Tensor illustration](./docs/source/_static/img/tensor_illustration.png)

It offers a wide array of tensor routines for scientific computation, including slicing, indexing, mathematical operations, linear algebra, and reductions, all optimized for speed.

### Dynamic Neural Networks: Tape-Based Autograd

PyTorch uses reverse-mode automatic differentiation ("autograd"), allowing flexible neural network design.

Most frameworks have a static view of the world. This means the network structure is defined upfront and remains fixed during training. Dynamic computation graphs remove this constraint and allow arbitrary changes to the network's structure at runtime, which opens up lots of possibilities for new architectures.

![Dynamic graph](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/dynamic_graph.gif)

### Python First

PyTorch is designed with deep integration into Python. Build your models in Python using your favorite libraries like NumPy, SciPy, Cython, and others.

### Imperative Experiences

PyTorch is designed to be intuitive, linear in thought, and easy to use. Experience straightforward debugging with informative stack traces and without opaque execution engines.

### Fast and Lean

PyTorch minimizes framework overhead, utilizing acceleration libraries like Intel MKL, cuDNN, and NCCL for speed.

PyTorch's memory usage is efficient, enabling the training of larger deep learning models through custom GPU memory allocators.

### Extensions Without Pain

Extending PyTorch is straightforward with minimal abstraction, allowing users to write custom neural network modules with the torch API in Python, or with C/C++ and a convenient extension API.

## Installation

Install PyTorch using the instructions on our website: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

### Binaries

#### NVIDIA Jetson Platforms

Pre-built Python wheels are available for NVIDIA Jetson platforms [here](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048). L4T container is published [here](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch)

### From Source

#### Prerequisites

*   Python 3.9 or later
*   A compiler that supports C++17, such as clang or gcc (gcc 9.4.0 or newer on Linux)
*   Visual Studio or Visual Studio Build Tool (Windows only)

#### NVIDIA CUDA Support

If you want to compile with CUDA support, [select a supported version of CUDA from our support matrix](https://pytorch.org/get-started/locally/), then install the following:
-   [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads)
-   [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) v8.5 or above
-   [Compiler](https://gist.github.com/ax3l/9489132) compatible with CUDA

If you want to disable CUDA support, export the environment variable `USE_CUDA=0`.

#### AMD ROCm Support

If you want to compile with ROCm support, install
-   [AMD ROCm](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html) 4.0 and above installation

If you want to disable ROCm support, export the environment variable `USE_ROCM=0`.

#### Intel GPU Support

If you want to compile with Intel GPU support, follow these instructions:
-   [PyTorch Prerequisites for Intel GPUs](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpus.html)

If you want to disable Intel GPU support, export the environment variable `USE_XPU=0`.

#### Get the PyTorch Source

```bash
git clone https://github.com/pytorch/pytorch
cd pytorch
git submodule sync
git submodule update --init --recursive
```

#### Install Dependencies

**Common**

```bash
conda install cmake ninja
pip install -r requirements.txt
```

**On Linux**

```bash
pip install mkl-static mkl-include
# CUDA only: Add LAPACK support for the GPU if needed
# magma installation: run with active conda environment. specify CUDA version to install
.ci/docker/common/install_magma_conda.sh 12.4

# (optional) If using torch.compile with inductor/triton, install the matching version of triton
# Run from the pytorch directory after cloning
# For Intel GPU support, please explicitly `export USE_XPU=1` before running command.
make triton
```

**On MacOS**

```bash
pip install mkl-static mkl-include
conda install pkg-config libuv
```

**On Windows**

```bash
pip install mkl-static mkl-include
conda install -c conda-forge libuv=1.39
```

#### Install PyTorch

**On Linux**

If you're compiling for AMD ROCm then first run this command:

```bash
# Only run this if you're compiling for ROCm
python tools/amd_build/build_amd.py
```

Install PyTorch

```bash
export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"
python -m pip install --no-build-isolation -v -e .
```

**On macOS**

```bash
python -m pip install --no-build-isolation -v -e .
```

**On Windows**

Refer to [Building on legacy code and CUDA](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md#building-on-legacy-code-and-cuda) for building legacy python code.

**CPU-only builds**

```cmd
python -m pip install --no-build-isolation -v -e .
```

**CUDA based build**

```cmd
cmd
:: Set the environment variables after you have downloaded and unzipped the mkl package,
:: else CMake would throw an error as `Could NOT find OpenMP`.
set CMAKE_INCLUDE_PATH={Your directory}\mkl\include
set LIB={Your directory}\mkl\lib;%LIB%

:: Read the content in the previous section carefully before you proceed.
:: [Optional] If you want to override the underlying toolset used by Ninja and Visual Studio with CUDA, please run the following script block.
:: "Visual Studio 2019 Developer Command Prompt" will be run automatically.
:: Make sure you have CMake >= 3.12 before you do this when you use the Visual Studio generator.
set CMAKE_GENERATOR_TOOLSET_VERSION=14.27
set DISTUTILS_USE_SDK=1
for /f "usebackq tokens=*" %i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -version [15^,17^) -products * -latest -property installationPath`) do call "%i\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=%CMAKE_GENERATOR_TOOLSET_VERSION%

:: [Optional] If you want to override the CUDA host compiler
set CUDAHOSTCXX=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.27.29110\bin\HostX64\x64\cl.exe

python -m pip install --no-build-isolation -v -e .
```

**Intel GPU builds**

Make sure you have the common and Intel GPU prerequisites installed, and then run:

```cmd
:: CMD Commands:
:: Set the CMAKE_PREFIX_PATH to help find corresponding packages
:: %CONDA_PREFIX% only works after `conda activate custom_env`

if defined CMAKE_PREFIX_PATH (
    set "CMAKE_PREFIX_PATH=%CONDA_PREFIX%\Library;%CMAKE_PREFIX_PATH%"
) else (
    set "CMAKE_PREFIX_PATH=%CONDA_PREFIX%\Library"
)

python -m pip install --no-build-isolation -v -e .
```

##### Adjust Build Options (Optional)

Adjust the configuration of cmake variables by doing the following.

On Linux

```bash
export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"
CMAKE_ONLY=1 python setup.py build
ccmake build  # or cmake-gui build
```

On macOS

```bash
export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"
MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ CMAKE_ONLY=1 python setup.py build
ccmake build  # or cmake-gui build
```

### Docker Image

#### Using pre-built images

```bash
docker run --gpus all --rm -ti --ipc=host pytorch/pytorch:latest
```

#### Building the image yourself

```bash
make -f docker.Makefile
```

### Building the Documentation

```bash
cd docs/
pip install -r requirements.txt
make html
make serve
```

#### Building a PDF

```bash
make latexpdf
make LATEXOPTS="-interaction=nonstopmode"
```

### Previous Versions

Installation instructions and binaries for previous PyTorch versions may be found
on [our website](https://pytorch.org/get-started/previous-versions).

## Getting Started

See resources for getting started:

*   [Tutorials](https://pytorch.org/tutorials/)
*   [Examples](https://github.com/pytorch/examples)
*   [API Reference](https://pytorch.org/docs/)

## Resources

*   [PyTorch.org](https://pytorch.org/)
*   [PyTorch Tutorials](https://pytorch.org/tutorials/)
*   [PyTorch Examples](https://github.com/pytorch/examples)
*   [PyTorch Models](https://pytorch.org/hub/)
*   [Intro to Deep Learning with PyTorch from Udacity](https://www.udacity.com/course/deep-learning-pytorch--ud188)
*   [Intro to Machine Learning with PyTorch from Udacity](https://www.udacity.com/course/intro-to-machine-learning-nanodegree--nd229)
*   [Deep Neural Networks with PyTorch from Coursera](https://www.coursera.org/learn/deep-neural-networks-with-pytorch)
*   [PyTorch Twitter](https://twitter.com/PyTorch)
*   [PyTorch Blog](https://pytorch.org/blog/)
*   [PyTorch YouTube](https://www.youtube.com/channel/UCWXI5YeOsh03QvJ59PMaXFw)

## Communication

*   Forums: Discuss implementations, research, etc. https://discuss.pytorch.org
*   GitHub Issues: Bug reports, feature requests, install issues, RFCs, thoughts, etc.
*   Slack: The [PyTorch Slack](https://pytorch.slack.com/) hosts a primary audience of moderate to experienced PyTorch users and developers for general chat, online discussions, collaboration, etc. If you are a beginner looking for help, the primary medium is [PyTorch Forums](https://discuss.pytorch.org). If you need a slack invite, please fill this form: https://goo.gl/forms/PP1AGvNHpSaJP8to1
*   Newsletter: No-noise, a one-way email newsletter with important announcements about PyTorch. You can sign-up here: https://eepurl.com/cbG0rv
*   Facebook Page: Important announcements about PyTorch. https://www.facebook.com/pytorch
*   For brand guidelines, please visit our website at [pytorch.org](https://pytorch.org/)

## Releases and Contributing

PyTorch typically has three minor releases per year.  Report bugs [filing an issue](https://github.com/pytorch/pytorch/issues).

Contribute bug-fixes or new features by first opening an issue. See [Contribution page](CONTRIBUTING.md) and [Release page](RELEASE.md)

## The Team

PyTorch is a community-driven project maintained by [Soumith Chintala](http://soumith.ch), [Gregory Chanan](https://github.com/gchanan), [Dmytro Dzhulgakov](https://github.com/dzhulgakov), [Edward Yang](https://github.com/ezyang), and [Nikita Shulga](https://github.com/malfet) with major contributions from many talented individuals, including: [Trevor Killeen](https://github.com/killeent), [Sasank Chilamkurthy](https://github.com/chsasank), [Sergey Zagoruyko](https://github.com/szagoruyko), [Adam Lerer](https://github.com/adamlerer), [Francisco Massa](https://github.com/fmassa), [Alykhan Tejani](https://github.com/alykhantejani), [Luca Antiga](https://github.com/lantiga), [Alban Desmaison](https://github.com/albanD), [Andreas Koepf](https://github.com/andreaskoepf), [James Bradbury](https://github.com/jekbradbury), [Zeming Lin](https://github.com/ebetica), [Yuandong Tian](https://github.com/yuandong-tian), [Guillaume Lample](https://github.com/glample), [Marat Dukhan](https://github.com/Maratyszcza), [Natalia Gimelshein](https://github.com/ngimel), [Christian Sarofeen](https://github.com/csarofeen), [Martin Raison](https://github.com/martinraison), [Edward Yang](https://github.com/ezyang), [Zachary Devito](https://github.com/zdevito).

## License

PyTorch is licensed under a BSD-style license, found in the [LICENSE](LICENSE) file.

**[Back to Top](#pytorch-the-open-source-deep-learning-framework)**