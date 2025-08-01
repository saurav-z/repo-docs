[![PyTorch Logo](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/pytorch-logo-dark.png)](https://github.com/pytorch/pytorch)

# PyTorch: The Deep Learning Framework for Research and Production

PyTorch is a leading open-source deep learning framework built to accelerate the path from research prototyping to production deployment. Explore the [PyTorch GitHub repository](https://github.com/pytorch/pytorch) to get started!

**Key Features:**

*   **GPU-Accelerated Tensor Computations:** Utilize GPU power for faster numerical computations, similar to NumPy but with GPU support.
*   **Dynamic Neural Networks with Autograd:** Build and modify neural networks on the fly, offering flexibility and speed for cutting-edge research.
*   **Python-First Approach:** Integrate seamlessly with existing Python libraries and workflows.
*   **Imperative Programming:** Enjoy an intuitive, easy-to-debug experience with straightforward code execution.
*   **Fast and Lean:** Leverage optimized acceleration libraries for speed and memory efficiency.
*   **Extensible Architecture:** Easily create custom neural network modules and interface with PyTorch's Tensor API.

## Table of Contents

-   [More About PyTorch](#more-about-pytorch)
    -   [A GPU-Ready Tensor Library](#a-gpu-ready-tensor-library)
    -   [Dynamic Neural Networks: Tape-Based Autograd](#dynamic-neural-networks-tape-based-autograd)
    -   [Python First](#python-first)
    -   [Imperative Experiences](#imperative-experiences)
    -   [Fast and Lean](#fast-and-lean)
    -   [Extensions Without Pain](#extensions-without-pain)
-   [Installation](#installation)
    -   [Binaries](#binaries)
        -   [NVIDIA Jetson Platforms](#nvidia-jetson-platforms)
    -   [From Source](#from-source)
        -   [Prerequisites](#prerequisites)
            -   [NVIDIA CUDA Support](#nvidia-cuda-support)
            -   [AMD ROCm Support](#amd-rocm-support)
            -   [Intel GPU Support](#intel-gpu-support)
        -   [Get the PyTorch Source](#get-the-pytorch-source)
        -   [Install Dependencies](#install-dependencies)
        -   [Install PyTorch](#install-pytorch)
            -   [Adjust Build Options (Optional)](#adjust-build-options-optional)
    -   [Docker Image](#docker-image)
        -   [Using pre-built images](#using-pre-built-images)
        -   [Building the image yourself](#building-the-image-yourself)
    -   [Building the Documentation](#building-the-documentation)
        -   [Building a PDF](#building-a-pdf)
    -   [Previous Versions](#previous-versions)
-   [Getting Started](#getting-started)
-   [Resources](#resources)
-   [Communication](#communication)
-   [Releases and Contributing](#releases-and-contributing)
-   [The Team](#the-team)
-   [License](#license)

## More About PyTorch

[Learn the basics of PyTorch](https://pytorch.org/tutorials/beginner/basics/intro.html)

PyTorch is a comprehensive library comprising essential components:

| Component                     | Description                                                                                                         |
| :----------------------------- | :------------------------------------------------------------------------------------------------------------------ |
| [**torch**](https://pytorch.org/docs/stable/torch.html)       | Tensor library with GPU support, similar to NumPy.                                            |
| [**torch.autograd**](https://pytorch.org/docs/stable/autograd.html) | Tape-based automatic differentiation library, supporting all differentiable Tensor operations. |
| [**torch.jit**](https://pytorch.org/docs/stable/jit.html)       | Compilation stack (TorchScript) for serializing and optimizing PyTorch code.                 |
| [**torch.nn**](https://pytorch.org/docs/stable/nn.html)         | Neural networks library, integrated with autograd, for maximum flexibility.                 |
| [**torch.multiprocessing**](https://pytorch.org/docs/stable/multiprocessing.html) | Python multiprocessing with shared memory for Tensors, useful for data loading and training.          |
| [**torch.utils**](https://pytorch.org/docs/stable/data.html)      | Data loading and utility functions, including DataLoader.                                      |

Use PyTorch as:

*   A replacement for NumPy to leverage GPUs.
*   A flexible deep learning research platform.

### A GPU-Ready Tensor Library

PyTorch offers Tensors that can reside on CPUs or GPUs, significantly accelerating computation.  It provides many tensor routines for scientific computation, including slicing, indexing, mathematical operations, linear algebra, and reductions, all optimized for speed.

![Tensor illustration](./docs/source/_static/img/tensor_illustration.png)

### Dynamic Neural Networks: Tape-Based Autograd

PyTorch uses reverse-mode automatic differentiation for maximum flexibility in neural network design.  This enables on-the-fly modifications without significant overhead.

![Dynamic graph](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/dynamic_graph.gif)

### Python First

PyTorch is designed for seamless Python integration. It works naturally with NumPy, SciPy, and scikit-learn, enabling you to write custom layers in Python using your preferred libraries and packages such as Cython and Numba.

### Imperative Experiences

PyTorch focuses on an intuitive, easy-to-debug experience. Code executes line-by-line, and stack traces clearly indicate the source of errors.

### Fast and Lean

PyTorch utilizes acceleration libraries like Intel MKL and NVIDIA cuDNN/NCCL for maximum speed. Its CPU and GPU Tensor and neural network backends are mature and efficient.

### Extensions Without Pain

PyTorch simplifies the creation of new neural network modules and interfacing with the Tensor API. You can write custom layers in Python using the torch API or NumPy-based libraries, or in C/C++ with a convenient extension API.

## Installation

### Binaries

Install binaries via Conda or pip wheels:  [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

#### NVIDIA Jetson Platforms

Wheels for NVIDIA Jetson Nano, TX1/TX2, Xavier NX/AGX, and AGX Orin are provided [here](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048) and the L4T container is published [here](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch).

They require JetPack 4.2 and above and are maintained by [@dusty-nv](https://github.com/dusty-nv) and [@ptrblck](https://github.com/ptrblck).

### From Source

#### Prerequisites

-   Python 3.9 or later
-   C++17 compatible compiler (gcc 9.4.0 or newer on Linux)
-   Visual Studio or Visual Studio Build Tool (Windows)

**Example Environment Setup:**

*   **Linux:**

```bash
$ source <CONDA_INSTALL_DIR>/bin/activate
$ conda create -y -n <CONDA_NAME>
$ conda activate <CONDA_NAME>
```

*   **Windows:**

```bash
$ source <CONDA_INSTALL_DIR>\Scripts\activate.bat
$ conda create -y -n <CONDA_NAME>
$ conda activate <CONDA_NAME>
$ call "C:\Program Files\Microsoft Visual Studio\<VERSION>\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
```

#### NVIDIA CUDA Support

Install the following to compile with CUDA:

-   [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads) (select a supported version from [our support matrix](https://pytorch.org/get-started/locally/))
-   [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) v8.5 or above
-   [Compiler](https://gist.github.com/ax3l/9489132) compatible with CUDA

Set `USE_CUDA=0` to disable CUDA support. Ensure `PATH` includes the nvcc location if CUDA is installed in a non-standard location.

Instructions for Jetson Nano are [available here](https://devtalk.nvidia.com/default/topic/1049071/jetson-nano/pytorch-for-jetson-nano/).

#### AMD ROCm Support

Install the following to compile with ROCm:

-   [AMD ROCm](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html) 4.0 and above (Linux only)

Set `ROCM_PATH` if ROCm is installed outside the default `/opt/rocm`. Set `PYTORCH_ROCM_ARCH` to specify the AMD GPU architecture. Set `USE_ROCM=0` to disable ROCm support.

#### Intel GPU Support

Follow [PyTorch Prerequisites for Intel GPUs](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpus.html) for Intel GPU support.

Set `USE_XPU=0` to disable Intel GPU support.

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
# Add this package on intel x86 processor machines only
pip install mkl-static mkl-include
# Add these packages if torch.distributed is needed
conda install pkg-config libuv
```

**On Windows**

```bash
pip install mkl-static mkl-include
# Add these packages if torch.distributed is needed.
# Distributed package support on Windows is a prototype feature and is subject to changes.
conda install -c conda-forge libuv=1.39
```

#### Install PyTorch

**On Linux**

```bash
# Only run this if you're compiling for ROCm
python tools/amd_build/build_amd.py
```

Install PyTorch:

```bash
export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"
python -m pip install --no-build-isolation -v -e .
```

**On macOS**

```bash
python -m pip install --no-build-isolation -v -e .
```

**On Windows**

Refer to [Building on legacy code and CUDA](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md#building-on-legacy-code-and-cuda) for legacy code.

**CPU-only builds:**

```cmd
python -m pip install --no-build-isolation -v -e .
```

**CUDA based build**

[NVTX](https://docs.nvidia.com/gameworks/content/gameworkslibrary/nvtx/nvidia_tools_extension_library_nvtx/nvtx.htm) is needed to build Pytorch with CUDA.
Additional libraries such as
[Magma](https://developer.nvidia.com/magma), [oneDNN, a.k.a. MKLDNN or DNNL](https://github.com/oneapi-src/oneDNN), and [Sccache](https://github.com/mozilla/sccache) are often needed. Please refer to the [installation-helper](https://github.com/pytorch/pytorch/tree/main/.ci/pytorch/win-test-helpers/installation-helpers) to install them.

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

Make sure the prerequisites are installed: [common prerequisites](#prerequisites) and [the prerequisites for Intel GPU](#intel-gpu-support). For build tool support, `Visual Studio 2022` is required.

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

```bash
export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"
CMAKE_ONLY=1 python setup.py build
ccmake build  # or cmake-gui build
```

#### On macOS

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
# images are tagged as docker.io/${your_docker_username}/pytorch
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

Find installation instructions and binaries at [our website](https://pytorch.org/get-started/previous-versions).

## Getting Started

*   [Tutorials](https://pytorch.org/tutorials/)
*   [Examples](https://github.com/pytorch/examples)
*   [API Reference](https://pytorch.org/docs/)
*   [Glossary](https://github.com/pytorch/pytorch/blob/main/GLOSSARY.md)

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

*   Forums: [discuss.pytorch.org](https://discuss.pytorch.org)
*   GitHub Issues
*   Slack: [PyTorch Slack](https://pytorch.slack.com/) (invite: https://goo.gl/forms/PP1AGvNHpSaJP8to1)
*   Newsletter: [eepurl.com/cbG0rv](https://eepurl.com/cbG0rv)
*   Facebook Page: [https://www.facebook.com/pytorch](https://www.facebook.com/pytorch)
*   Brand guidelines: [pytorch.org](https://pytorch.org/)

## Releases and Contributing

Report bugs via [filing an issue](https://github.com/pytorch/pytorch/issues). Contribute new features by discussing them in a GitHub issue first. See the [Contribution page](CONTRIBUTING.md) and [Release page](RELEASE.md) for details.

## The Team

PyTorch is a community-driven project. [See the team](https://github.com/pytorch/pytorch/blob/main/README.md#the-team).

## License

PyTorch is licensed under a BSD-style license ([LICENSE](LICENSE)).