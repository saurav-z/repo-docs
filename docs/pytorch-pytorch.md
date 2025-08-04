# PyTorch: The Open Source Deep Learning Framework 

[![PyTorch Logo](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/pytorch-logo-dark.png)](https://github.com/pytorch/pytorch)

**PyTorch is an open-source deep learning framework that empowers researchers and developers to build and train machine learning models with unparalleled flexibility and speed.**

## Key Features:

*   **GPU-Accelerated Tensor Computation:** Leverage the power of GPUs for fast tensor operations, mimicking NumPy but with significant performance boosts.
*   **Dynamic Neural Networks:** Build and modify neural networks on the fly with PyTorch's tape-based autograd system, offering unparalleled flexibility for research.
*   **Python-First Approach:** Seamlessly integrate with your existing Python workflow, including libraries like NumPy, SciPy, and Cython.
*   **Imperative Programming:** Enjoy an intuitive and easy-to-debug experience with PyTorch's imperative programming style, where code executes line by line.
*   **Fast and Lean:** Benefit from minimal framework overhead, optimized backends (Intel MKL, cuDNN, NCCL), and efficient memory usage for optimal performance.
*   **Simplified Extensions:** Easily write custom neural network modules and interface with PyTorch's tensor API using Python or C/C++, without cumbersome wrapper code.

[See the original repository on GitHub](https://github.com/pytorch/pytorch)

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

At its core, PyTorch is a library comprising several key components:

| Component | Description                                                                                                  |
| :-------- | :----------------------------------------------------------------------------------------------------------- |
| **torch**           | A Tensor library (similar to NumPy) with strong GPU support                                              |
| **torch.autograd**  | A tape-based automatic differentiation library that supports all differentiable Tensor operations in torch |
| **torch.jit**       | A compilation stack (TorchScript) to create serializable and optimizable models from PyTorch code  |
| **torch.nn**        | A neural networks library deeply integrated with autograd designed for maximum flexibility                 |
| **torch.multiprocessing** | Python multiprocessing, but with magical memory sharing of torch Tensors across processes. Useful for data loading and Hogwild training |
| **torch.utils**     | DataLoader and other utility functions for convenience                                                   |

Common use cases for PyTorch include:

*   Replacing NumPy to utilize GPU acceleration.
*   Serving as a flexible research platform for deep learning.

### A GPU-Ready Tensor Library

PyTorch provides Tensors, similar to NumPy's ndarrays, that can operate on both CPUs and GPUs, enabling significant acceleration.

![Tensor illustration](./docs/source/_static/img/tensor_illustration.png)

PyTorch offers a rich set of tensor routines for scientific computation, including slicing, indexing, mathematical operations, linear algebra, and reductions, all optimized for speed.

### Dynamic Neural Networks: Tape-Based Autograd

PyTorch uses reverse-mode auto-differentiation, allowing for dynamic network behavior changes with minimal overhead.  This contrasts with static frameworks, providing the flexibility needed for cutting-edge research.

![Dynamic graph](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/dynamic_graph.gif)

### Python First

PyTorch is deeply integrated with Python, allowing you to use it naturally within your existing Python environment.  You can easily incorporate familiar libraries like NumPy, SciPy, and Cython.

### Imperative Experiences

PyTorch's design promotes an intuitive, linear coding experience. Code executes immediately, simplifying debugging and error analysis.

### Fast and Lean

PyTorch minimizes framework overhead and utilizes optimized libraries such as Intel MKL, cuDNN, and NCCL to maximize speed. It offers excellent memory efficiency, allowing you to train larger models.

### Extensions Without Pain

PyTorch makes creating new neural network modules and working with its Tensor API straightforward. You can write layers in Python or leverage a convenient extension API for C/C++, minimizing boilerplate code.

## Installation

### Binaries

Install binaries using Conda or pip wheels.  Visit [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) for instructions.

#### NVIDIA Jetson Platforms

Pre-built Python wheels for NVIDIA Jetson platforms are available [here](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048), with L4T containers published [here](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch). They require JetPack 4.2 or later.

### From Source

#### Prerequisites

*   Python 3.9 or later
*   C++17 compliant compiler (gcc 9.4.0 or newer on Linux)
*   Visual Studio or Visual Studio Build Tool (Windows only)

\* PyTorch CI uses Visual C++ BuildTools, which come with Visual Studio Enterprise,
Professional, or Community Editions. You can also install the build tools from
https://visualstudio.microsoft.com/visual-cpp-build-tools/. The build tools *do not*
come with Visual Studio Code by default.

Example environment setup:

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

A conda environment is not required. You can also build in a standard virtual environment.

##### NVIDIA CUDA Support

To compile with CUDA support:

1.  [Select a supported CUDA version](https://pytorch.org/get-started/locally/).
2.  Install [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads), [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) v8.5 or above, and a compatible [compiler](https://gist.github.com/ax3l/9489132).

*Refer to the [cuDNN Support Matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/support-matrix.html) for compatible versions.*

Set `USE_CUDA=0` to disable CUDA support.

If you are building for NVIDIA's Jetson platforms (Jetson Nano, TX1, TX2, AGX Xavier), Instructions to install PyTorch for Jetson Nano are [available here](https://devtalk.nvidia.com/default/topic/1049071/jetson-nano/pytorch-for-jetson-nano/)

##### AMD ROCm Support

To compile with ROCm support:

1.  Install [AMD ROCm](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html) 4.0 or later.
2.  ROCm is supported only on Linux.

By default the build system expects ROCm to be installed in `/opt/rocm`. If ROCm is installed in a different directory, the `ROCM_PATH` environment variable must be set to the ROCm installation directory. The build system automatically detects the AMD GPU architecture. Optionally, the AMD GPU architecture can be explicitly set with the `PYTORCH_ROCM_ARCH` environment variable [AMD GPU architecture](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-gpus)

Set `USE_ROCM=0` to disable ROCm support.

##### Intel GPU Support

To compile with Intel GPU support:

1.  Follow the [PyTorch Prerequisites for Intel GPUs](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpus.html) instructions.
2.  Intel GPU is supported for Linux and Windows.

Set `USE_XPU=0` to disable Intel GPU support.

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

If you want to build legacy python code, please refer to [Building on legacy code and CUDA](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md#building-on-legacy-code-and-cuda)

**CPU-only builds**

In this mode PyTorch computations will run on your CPU, not your GPU.

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

Customize CMake variables optionally before building.

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

Run pre-built Docker images from Docker Hub (requires docker v19.03+):

```bash
docker run --gpus all --rm -ti --ipc=host pytorch/pytorch:latest
```

*Use `--ipc=host` or `--shm-size` for multiprocessing with `torch.multiprocessing`.*

#### Building the image yourself

*Build with Docker version > 18.06.*

```bash
make -f docker.Makefile
# images are tagged as docker.io/${your_docker_username}/pytorch
```

*   Pass `PYTHON_VERSION=x.y` to specify the Python version.
*   Use `CMAKE_VARS="..."` to pass additional CMake variables.

### Building the Documentation

You will need [Sphinx](http://www.sphinx-doc.org) and pytorch_sphinx_theme2 to build the documentation.

Make sure `torch` is installed in your environment.  For small fixes, you can install the
nightly version as described in [Getting Started](https://pytorch.org/get-started/locally/).
For more complex fixes, such as adding a new module and docstrings for
the new module, you might need to install torch [from source](#from-source).
See [Docstring Guidelines](https://github.com/pytorch/pytorch/wiki/Docstring-Guidelines)
for docstring conventions.

```bash
cd docs/
pip install -r requirements.txt
make html
make serve
```

Run `make` to see all output formats.

*   If you encounter a katex error run `npm install katex`. If it persists, try `npm install -g katex`
*   If you see a numpy incompatibility error, run `pip install 'numpy<2'`

When you make changes to the dependencies run by CI, edit the
`.ci/docker/requirements-docs.txt` file.

#### Building a PDF

Requires `texlive` and LaTeX.

1.  Run `make latexpdf`.
2.  Navigate to `build/latex` and run `make LATEXOPTS="-interaction=nonstopmode"`.
3.  Run this command one more time so that it generates the correct table
    of contents and index.

> [!NOTE]
> To view the Table of Contents, switch to the **Table of Contents**
> view in your PDF viewer.

### Previous Versions

Find installation instructions and binaries for previous PyTorch versions on [our website](https://pytorch.org/get-started/previous-versions).

## Getting Started

*   [Tutorials](https://pytorch.org/tutorials/): Learn the basics.
*   [Examples](https://github.com/pytorch/examples): Explore working code across various domains.
*   [API Reference](https://pytorch.org/docs/): Consult the comprehensive API documentation.
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

*   **Forums:** Discuss implementations and research at https://discuss.pytorch.org
*   **GitHub Issues:** Report bugs, request features, and discuss installation issues.
*   **Slack:** Connect with other PyTorch users and developers (request an invite: https://goo.gl/forms/PP1AGvNHpSaJP8to1)
*   **Newsletter:** Sign up for the PyTorch newsletter at https://eepurl.com/cbG0rv
*   **Facebook Page:** Stay updated on PyTorch announcements at https://www.facebook.com/pytorch
*   **Brand Guidelines:** Access brand guidelines on the PyTorch website at [pytorch.org](https://pytorch.org/)

## Releases and Contributing

PyTorch typically has three minor releases per year.  Please report bugs via [GitHub Issues](https://github.com/pytorch/pytorch/issues).

Contributions are welcome. For new features, open an issue to discuss them before submitting a pull request.  See the [Contribution page](CONTRIBUTING.md) and [Release page](RELEASE.md) for details.

## The Team

PyTorch is a community-driven project maintained by dedicated engineers, researchers, and contributors.  The current maintainers are [Soumith Chintala](http://soumith.ch), [Gregory Chanan](https://github.com/gchanan), [Dmytro Dzhulgakov](https://github.com/dzhulgakov), [Edward Yang](https://github.com/ezyang), and [Nikita Shulga](https://github.com/malfet), with significant contributions from numerous talented individuals.

*This project is unrelated to [hughperkins/pytorch](https://github.com/hughperkins/pytorch).*

## License

PyTorch is licensed under a BSD-style license, found in the [LICENSE](LICENSE) file.