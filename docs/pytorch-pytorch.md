![PyTorch Logo](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/pytorch-logo-dark.png)

# PyTorch: Deep Learning Made Flexible and Fast

**PyTorch is a leading open-source machine learning framework that provides a flexible and efficient platform for building and training deep learning models.**  [Explore the PyTorch Repository](https://github.com/pytorch/pytorch).

**Key Features:**

*   **Tensor Computation with GPU Acceleration:**  Like NumPy, but with GPU support for faster computation.
*   **Dynamic Neural Networks:**  Built on a tape-based autograd system, allowing for flexible and dynamic network architectures.
*   **Python-First Design:** Deeply integrated with Python, enabling seamless use of familiar libraries like NumPy and SciPy.
*   **Imperative Programming Style:** Provides an intuitive and easy-to-debug development experience.
*   **Fast and Lean:** Optimized for speed with minimal framework overhead, leveraging acceleration libraries like Intel MKL and NVIDIA cuDNN.
*   **Simplified Extension:**  Easily write new neural network modules and interface with PyTorch's Tensor API.

## Contents

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

At its core, PyTorch is a library composed of the following key components:

| Component                     | Description                                                                                                                               |
| :----------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------ |
| [**torch**](https://pytorch.org/docs/stable/torch.html)           | A Tensor library, similar to NumPy, but with GPU support.                                           |
| [**torch.autograd**](https://pytorch.org/docs/stable/autograd.html)  | A tape-based automatic differentiation library, supporting all differentiable Tensor operations. |
| [**torch.jit**](https://pytorch.org/docs/stable/jit.html)           | A compilation stack (TorchScript) for creating serializable and optimizable models.               |
| [**torch.nn**](https://pytorch.org/docs/stable/nn.html)             | A neural networks library tightly integrated with autograd, designed for maximum flexibility.        |
| [**torch.multiprocessing**](https://pytorch.org/docs/stable/multiprocessing.html) | Python multiprocessing with memory sharing for Tensors across processes (useful for data loading and Hogwild training). |
| [**torch.utils**](https://pytorch.org/docs/stable/data.html)        | DataLoader and other utility functions.                                                          |

PyTorch is typically used for:

*   Replacing NumPy to leverage the power of GPUs.
*   Developing deep learning research platforms with maximum flexibility and speed.

### A GPU-Ready Tensor Library

PyTorch provides Tensors, similar to NumPy's ndarrays, which can reside on either the CPU or GPU, significantly accelerating computations.

![Tensor illustration](./docs/source/_static/img/tensor_illustration.png)

It offers a wide array of tensor routines for scientific computation, including slicing, indexing, mathematical operations, linear algebra, and reductions, all optimized for speed.

### Dynamic Neural Networks: Tape-Based Autograd

PyTorch utilizes reverse-mode automatic differentiation, allowing you to dynamically change the behavior of your network without overhead.

Most frameworks, such as TensorFlow, Theano, Caffe, and CNTK, have a static view of the world.
One has to build a neural network and reuse the same structure again and again.
Changing the way the network behaves means that one has to start from scratch.

With PyTorch, we use a technique called reverse-mode auto-differentiation, which allows you to
change the way your network behaves arbitrarily with zero lag or overhead. Our inspiration comes
from several research papers on this topic, as well as current and past work such as
[torch-autograd](https://github.com/twitter/torch-autograd),
[autograd](https://github.com/HIPS/autograd),
[Chainer](https://chainer.org), etc.

While this technique is not unique to PyTorch, it's one of the fastest implementations of it to date.
You get the best of speed and flexibility for your crazy research.

![Dynamic graph](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/dynamic_graph.gif)

### Python First

PyTorch is built to be deeply integrated into Python.

You can use it naturally like you would use [NumPy](https://www.numpy.org/) / [SciPy](https://www.scipy.org/) / [scikit-learn](https://scikit-learn.org) etc.
You can write your new neural network layers in Python itself, using your favorite libraries
and use packages such as [Cython](https://cython.org/) and [Numba](http://numba.pydata.org/).
Our goal is to not reinvent the wheel where appropriate.

### Imperative Experiences

PyTorch is designed for intuitive and easy use.

When you execute a line of code, it gets executed. There isn't an asynchronous view of the world.
When you drop into a debugger or receive error messages and stack traces, understanding them is straightforward.
The stack trace points to exactly where your code was defined.
We hope you never spend hours debugging your code because of bad stack traces or asynchronous and opaque execution engines.

### Fast and Lean

PyTorch has minimal framework overhead and integrates with acceleration libraries like Intel MKL and NVIDIA cuDNN/NCCL.

Hence, PyTorch is quite fast — whether you run small or large neural networks.

The memory usage in PyTorch is extremely efficient compared to Torch or some of the alternatives.
We've written custom memory allocators for the GPU to make sure that
your deep learning models are maximally memory efficient.
This enables you to train bigger deep learning models than before.

### Extensions Without Pain

PyTorch is designed to make writing new neural network modules and interacting with the Tensor API easy.

You can write new neural network layers in Python using the torch API
[or your favorite NumPy-based libraries such as SciPy](https://pytorch.org/tutorials/advanced/numpy_extensions_tutorial.html).

If you want to write your layers in C/C++, we provide a convenient extension API that is efficient and with minimal boilerplate.
No wrapper code needs to be written. You can see [a tutorial here](https://pytorch.org/tutorials/advanced/cpp_extension.html) and [an example here](https://github.com/pytorch/extension-cpp).

## Installation

### Binaries

Install binaries via Conda or pip wheels from our website: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

#### NVIDIA Jetson Platforms

Pre-built Python wheels for NVIDIA Jetson platforms are available [here](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048), and the L4T container is published [here](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch). They require JetPack 4.2 and above.

### From Source

#### Prerequisites

If installing from source, you'll need:

*   Python 3.9 or later
*   A compiler supporting C++17 (e.g., clang or gcc, with gcc 9.4.0+ required on Linux)
*   Visual Studio or Visual Studio Build Tool (Windows only)

    \* PyTorch CI uses Visual C++ BuildTools, which come with Visual Studio Enterprise,
    Professional, or Community Editions. You can also install the build tools from
    https://visualstudio.microsoft.com/visual-cpp-build-tools/. The build tools *do not*
    come with Visual Studio Code by default.

*   Example environment setup:

    *   Linux:

    ```bash
    $ source <CONDA_INSTALL_DIR>/bin/activate
    $ conda create -y -n <CONDA_NAME>
    $ conda activate <CONDA_NAME>
    ```

    *   Windows:

    ```bash
    $ source <CONDA_INSTALL_DIR>\Scripts\activate.bat
    $ conda create -y -n <CONDA_NAME>
    $ conda activate <CONDA_NAME>
    $ call "C:\Program Files\Microsoft Visual Studio\<VERSION>\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
    ```

    A conda environment is not required.  You can also do a PyTorch build in a
    standard virtual environment, e.g., created with tools like `uv`, provided
    your system has installed all the necessary dependencies unavailable as pip
    packages (e.g., CUDA, MKL.)

##### NVIDIA CUDA Support

If compiling with CUDA support, install:

*   [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads) (select a supported version from the matrix: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/))
*   [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) v8.5 or above
*   Compiler compatible with CUDA

    Note: You could refer to the [cuDNN Support Matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/support-matrix.html) for cuDNN versions with the various supported CUDA, CUDA driver, and NVIDIA hardware.

To disable CUDA, set `USE_CUDA=0`.  If CUDA is in a non-standard location, set PATH accordingly.

If building for NVIDIA Jetson platforms, instructions are available [here](https://devtalk.nvidia.com/default/topic/1049071/jetson-nano/pytorch-for-jetson-nano/).

##### AMD ROCm Support

If compiling with ROCm support, install:

*   [AMD ROCm](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html) 4.0 and above (Linux only)

The build system expects ROCm in `/opt/rocm` by default; use `ROCM_PATH` if it's installed elsewhere.  Use `PYTORCH_ROCM_ARCH` to explicitly set the AMD GPU architecture.

To disable ROCm, set `USE_ROCM=0`.

##### Intel GPU Support

If compiling with Intel GPU support, follow:

*   [PyTorch Prerequisites for Intel GPUs](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpus.html)

Intel GPU is supported for Linux and Windows.

To disable Intel GPU support, set `USE_XPU=0`.

#### Get the PyTorch Source

```bash
git clone https://github.com/pytorch/pytorch
cd pytorch
# if you are updating an existing checkout
git submodule sync
git submodule update --init --recursive
```

#### Install Dependencies

**Common**

```bash
conda install cmake ninja
# Run this command from the PyTorch directory after cloning the source code using the “Get the PyTorch Source“ section below
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

If compiling for AMD ROCm, run this command first:

```bash
# Only run this if you're compiling for ROCm
python tools/amd_build/build_amd.py
```

Then install PyTorch:

```bash
export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation -v -e .
```

**On macOS**

```bash
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation -v -e .
```

**On Windows**

If you want to build legacy python code, please refer to [Building on legacy code and CUDA](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md#building-on-legacy-code-and-cuda)

**CPU-only builds**

```cmd
python -m pip install --no-build-isolation -v -e .
```

Note on OpenMP: The desired OpenMP implementation is Intel OpenMP (iomp). In order to link against iomp, you'll need to manually download the library and set up the building environment by tweaking `CMAKE_INCLUDE_PATH` and `LIB`. The instruction [here](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/windows.rst#building-from-source) is an example for setting up both MKL and Intel OpenMP. Without these configurations for CMake, Microsoft Visual C OpenMP runtime (vcomp) will be used.

**CUDA based build**

In this mode PyTorch computations will leverage your GPU via CUDA for faster number crunching

[NVTX](https://docs.nvidia.com/gameworks/content/gameworkslibrary/nvtx/nvidia_tools_extension_library_nvtx.htm) is needed to build Pytorch with CUDA.
NVTX is a part of CUDA distributive, where it is called "Nsight Compute". To install it onto an already installed CUDA run CUDA installation once again and check the corresponding checkbox.
Make sure that CUDA with Nsight Compute is installed after Visual Studio.

Currently, VS 2017 / 2019, and Ninja are supported as the generator of CMake. If `ninja.exe` is detected in `PATH`, then Ninja will be used as the default generator, otherwise, it will use VS 2017 / 2019.
<br/> If Ninja is selected as the generator, the latest MSVC will get selected as the underlying toolchain.

Additional libraries such as
[Magma](https://developer.nvidia.com/magma), [oneDNN, a.k.a. MKLDNN or DNNL](https://github.com/oneapi-src/oneDNN), and [Sccache](https://github.com/mozilla/sccache) are often needed. Please refer to the [installation-helper](https://github.com/pytorch/pytorch/tree/main/.ci/pytorch/win-test-helpers/installation-helpers) to install them.

You can refer to the [build_pytorch.bat](https://github.com/pytorch/pytorch/blob/main/.ci/pytorch/win-test-helpers/build_pytorch.bat) script for some other environment variables configurations

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

In this mode PyTorch with Intel GPU support will be built.

Please make sure [the common prerequisites](#prerequisites) as well as [the prerequisites for Intel GPU](#intel-gpu-support) are properly installed and the environment variables are configured prior to starting the build. For build tool support, `Visual Studio 2022` is required.

Then PyTorch can be built with the command:

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

You can adjust the configuration of cmake variables optionally (without building first), by doing
the following. For example, adjusting the pre-detected directories for CuDNN or BLAS can be done
with such a step.

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

Run pre-built Docker images from Docker Hub with:

```bash
docker run --gpus all --rm -ti --ipc=host pytorch/pytorch:latest
```

*   Note:  Use `--ipc=host` or `--shm-size` when using `torch.multiprocessing`.

#### Building the image yourself

**NOTE:**  Requires Docker version > 18.06

Build the Docker image with CUDA 11.1 support:

```bash
make -f docker.Makefile
# images are tagged as docker.io/${your_docker_username}/pytorch
```

*   You can also set `CMAKE_VARS="..."` to specify additional CMake variables.

### Building the Documentation

Build documentation with Sphinx and the pytorch_sphinx_theme2:

```bash
cd docs/
pip install -r requirements.txt
make html
make serve
```

Run `make` for a list of output formats.  Fix katex and numpy incompatibilities if errors appear.

When you make changes to the dependencies run by CI, edit the
`.ci/docker/requirements-docs.txt` file.

#### Building a PDF

Create a PDF of the documentation with:

1.  `make latexpdf`
2.  `make LATEXOPTS="-interaction=nonstopmode"` within the `build/latex` directory.

### Previous Versions

Find installation instructions and binaries for previous versions [on our website](https://pytorch.org/get-started/previous-versions).

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

*   [Forums](https://discuss.pytorch.org)
*   [GitHub Issues](https://github.com/pytorch/pytorch/issues)
*   [Slack](https://pytorch.slack.com/) (sign-up: https://goo.gl/forms/PP1AGvNHpSaJP8to1)
*   [Newsletter](https://eepurl.com/cbG0rv)
*   [Facebook Page](https://www.facebook.com/pytorch)
*   [Brand Guidelines](https://pytorch.org/)

## Releases and Contributing

PyTorch typically releases three minor versions a year. Report bugs via [GitHub Issues](https://github.com/pytorch/pytorch/issues).

We welcome contributions! For new features, open an issue to discuss them first.  See the [Contribution page](CONTRIBUTING.md) and [Release page](RELEASE.md) for more information.

## The Team

PyTorch is a community-driven project, maintained by [Soumith Chintala](http://soumith.ch), [Gregory Chanan](https://github.com/gchanan), [Dmytro Dzhulgakov](https://github.com/dzhulgakov), [Edward Yang](https://github.com/ezyang), and [Nikita Shulga](https://github.com/malfet) with contributions from many others.

## License

PyTorch is licensed under a BSD-style license, found in the [LICENSE](LICENSE) file.