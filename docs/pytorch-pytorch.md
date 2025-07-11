<img src="https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/pytorch-logo-dark.png" alt="PyTorch Logo" width="200">

# PyTorch: Deep Learning Framework for Research and Production

**PyTorch is a leading deep learning framework that empowers researchers and developers to build cutting-edge AI models.** [Learn more about PyTorch](https://github.com/pytorch/pytorch)

**Key Features:**

*   **Tensor Computation with GPU Acceleration:**  Leverage the power of GPUs for fast tensor operations, similar to NumPy.
*   **Dynamic Neural Networks with Autograd:** Build and modify neural networks with unparalleled flexibility using a tape-based autograd system.
*   **Python-First Approach:** Seamlessly integrate PyTorch with your existing Python workflow, utilizing familiar libraries like NumPy, SciPy, and Cython.
*   **Imperative and Intuitive:** Experience a straightforward and easy-to-debug development process with an imperative programming style.
*   **Fast and Lean:** Benefit from minimal framework overhead and optimized acceleration libraries like Intel MKL and cuDNN.
*   **Easy Extensions:** Create custom neural network modules and easily interface with PyTorch's Tensor API with minimal abstractions.

## Getting Started

Explore these resources to begin your PyTorch journey:

*   [**Tutorials**](https://pytorch.org/tutorials/) – Learn the fundamentals with comprehensive tutorials.
*   [**Examples**](https://github.com/pytorch/examples) – Discover practical PyTorch code examples across diverse domains.
*   [**API Reference**](https://pytorch.org/docs/) – Consult the official API documentation.
*   [**Glossary**](https://github.com/pytorch/pytorch/blob/main/GLOSSARY.md) – Understand key PyTorch terms and concepts.

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

At its core, PyTorch provides the following key components:

| Component               | Description                                                                                                         |
| :---------------------- | :------------------------------------------------------------------------------------------------------------------ |
| [`torch`](https://pytorch.org/docs/stable/torch.html)               | Tensor library like NumPy with strong GPU support.                                     |
| [`torch.autograd`](https://pytorch.org/docs/stable/autograd.html)   | Tape-based automatic differentiation for all differentiable Tensor operations.    |
| [`torch.jit`](https://pytorch.org/docs/stable/jit.html)           | Compilation stack (TorchScript) to create serializable and optimizable models.     |
| [`torch.nn`](https://pytorch.org/docs/stable/nn.html)                | Neural networks library deeply integrated with autograd for maximum flexibility. |
| [`torch.multiprocessing`](https://pytorch.org/docs/stable/multiprocessing.html) | Python multiprocessing with magic memory sharing of torch Tensors across processes. |
| [`torch.utils`](https://pytorch.org/docs/stable/data.html)             | DataLoader and other utility functions.                                             |

PyTorch is commonly used for:

*   Replacing NumPy to harness the power of GPUs.
*   Rapid prototyping and flexible deep learning research.

### A GPU-Ready Tensor Library

If you use NumPy, you are familiar with Tensors (a.k.a. ndarray).

![Tensor illustration](./docs/source/_static/img/tensor_illustration.png)

PyTorch offers Tensors that can reside on CPUs or GPUs, significantly accelerating computations. It provides a wide range of tensor routines for slicing, indexing, mathematical operations, and linear algebra, all designed for speed.

### Dynamic Neural Networks: Tape-Based Autograd

PyTorch builds neural networks using a unique tape recorder approach. Unlike frameworks with static graphs, PyTorch uses reverse-mode auto-differentiation, allowing you to modify your network's behavior without overhead.

![Dynamic graph](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/dynamic_graph.gif)

### Python First

PyTorch is deeply integrated into Python, making it feel natural to use. You can extend it using your favorite Python libraries and packages like Cython and Numba.

### Imperative Experiences

PyTorch offers an intuitive, linear-in-thought, and easy-to-use experience. Code executes immediately, and debugging is straightforward due to clear stack traces.

### Fast and Lean

PyTorch minimizes framework overhead and integrates acceleration libraries like Intel MKL and NVIDIA cuDNN/NCCL to maximize speed. Its CPU and GPU Tensor and neural network backends have been thoroughly tested.

PyTorch also has efficient memory usage compared to alternatives, with custom memory allocators for GPUs enabling training of larger models.

### Extensions Without Pain

Extending PyTorch is designed to be straightforward. You can write new neural network layers in Python using the torch API, or in C/C++ using a convenient extension API.

## Installation

The easiest way to get started is by installing pre-built binaries, using either Conda or pip. You can select your build configuration on the PyTorch website.

*   [**Binaries**](https://pytorch.org/get-started/locally/)

    *   [**NVIDIA Jetson Platforms**](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048)

*   [**From Source**](#from-source)

    *   [**Prerequisites**](#prerequisites)
        *   [NVIDIA CUDA Support](#nvidia-cuda-support)
        *   [AMD ROCm Support](#amd-rocm-support)
        *   [Intel GPU Support](#intel-gpu-support)
    *   [Get the PyTorch Source](#get-the-pytorch-source)
    *   [Install Dependencies](#install-dependencies)
    *   [Install PyTorch](#install-pytorch)
        *   [Adjust Build Options (Optional)](#adjust-build-options-optional)

*   [**Docker Image**](#docker-image)

    *   [Using pre-built images](#using-pre-built-images)
    *   [Building the image yourself](#building-the-image-yourself)

*   [**Building the Documentation**](#building-the-documentation)
    *   [Building a PDF](#building-a-pdf)

*   [**Previous Versions**](https://pytorch.org/get-started/previous-versions)

### Binaries

Commands to install binaries via Conda or pip wheels are on our website: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

#### NVIDIA Jetson Platforms

Python wheels for NVIDIA's Jetson Nano, Jetson TX1/TX2, Jetson Xavier NX/AGX, and Jetson AGX Orin are provided [here](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048) and the L4T container is published [here](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch)

They require JetPack 4.2 and above, and [@dusty-nv](https://github.com/dusty-nv) and [@ptrblck](https://github.com/ptrblck) are maintaining them.

### From Source

#### Prerequisites

Ensure you have the following:

*   Python 3.9 or later
*   A compiler with full C++17 support (e.g., clang or gcc).
*   Visual Studio or Visual Studio Build Tools (Windows only).

*PyTorch CI uses Visual C++ BuildTools, which come with Visual Studio Enterprise,
Professional, or Community Editions. You can also install the build tools from
https://visualstudio.microsoft.com/visual-cpp-build-tools/. The build tools *do not*
come with Visual Studio Code by default.*

*Example Environment Setup*

*Linux:*
```bash
$ source <CONDA_INSTALL_DIR>/bin/activate
$ conda create -y -n <CONDA_NAME>
$ conda activate <CONDA_NAME>
```
*Windows:*
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

To compile with CUDA support:

1.  **Supported CUDA Version:** [Select a supported version from the support matrix](https://pytorch.org/get-started/locally/)
2.  **Install:**
    *   [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads)
    *   [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) v8.5 or above
    *   [Compiler](https://gist.github.com/ax3l/9489132) compatible with CUDA

    Note: You could refer to the [cuDNN Support Matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/support-matrix.html) for cuDNN versions with the various supported CUDA, CUDA driver, and NVIDIA hardware.

3.  **Disable CUDA:**  Set `USE_CUDA=0` environment variable.
4.  **CUDA Path:** If CUDA is installed in a non-standard location, set the `PATH` environment variable to locate `nvcc`.
    e.g., `export PATH=/usr/local/cuda-12.8/bin:$PATH`

If you are building for NVIDIA's Jetson platforms (Jetson Nano, TX1, TX2, AGX Xavier), Instructions to install PyTorch for Jetson Nano are [available here](https://devtalk.nvidia.com/default/topic/1049071/jetson-nano/pytorch-for-jetson-nano/)

##### AMD ROCm Support

To compile with ROCm support:

1.  **Install:**
    *   [AMD ROCm](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html) 4.0 and above.  **ROCm is currently supported only for Linux.**
2.  **ROCM_PATH:** If ROCm is not installed in `/opt/rocm`, set the `ROCM_PATH` environment variable to the installation directory.
3.  **AMD GPU Architecture:**  The build system automatically detects the AMD GPU architecture. You can set the `PYTORCH_ROCM_ARCH` environment variable to specify it.  [AMD GPU architecture](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-gpus)
4.  **Disable ROCm:** Set `USE_ROCM=0` environment variable.

##### Intel GPU Support

To compile with Intel GPU support:

1.  **Follow Prerequisites:** [PyTorch Prerequisites for Intel GPUs](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpus.html)
2.  **Supported OS:** Intel GPU is supported for Linux and Windows.
3.  **Disable Intel GPU:** Set `USE_XPU=0` environment variable.

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

Optionally, configure cmake variables after building. For example, adjust pre-detected directories for CuDNN or BLAS:

*Linux and macOS*
```bash
export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"
CMAKE_ONLY=1 python setup.py build
ccmake build  # or cmake-gui build
```

### Docker Image

#### Using pre-built images

```bash
docker run --gpus all --rm -ti --ipc=host pytorch/pytorch:latest
```

#### Building the image yourself

Build with a Docker version > 18.06.

```bash
make -f docker.Makefile
# images are tagged as docker.io/${your_docker_username}/pytorch
```

### Building the Documentation

Documentation is built using Sphinx and the pytorch_sphinx_theme2.
Ensure torch is installed in your environment.

```bash
cd docs/
pip install -r requirements.txt
make html
make serve
```

Run `make` for a list of output formats.
If a katex error occurs run `npm install katex`.  If it persists, try
`npm install -g katex`

> [!NOTE]
> If you installed `nodejs` with a different package manager (e.g.,
> `conda`) then `npm` will probably install a version of `katex` that is not
> compatible with your version of `nodejs` and doc builds will fail.
> A combination of versions that is known to work is `node@6.13.1` and
> `katex@0.13.18`. To install the latter with `npm` you can run
> ```npm install -g katex@0.13.18```

> [!NOTE]
> If you see a numpy incompatibility error, run:
> ```
> pip install 'numpy<2'
> ```

When you make changes to the dependencies run by CI, edit the
`.ci/docker/requirements-docs.txt` file.

#### Building a PDF

To create the PDF:

1.  Run:
    ```
    make latexpdf
    ```
2.  Navigate to this directory and execute:
    ```
    make LATEXOPTS="-interaction=nonstopmode"
    ```
    Run this command one more time so that it generates the correct table
    of contents and index.

> [!NOTE]
> To view the Table of Contents, switch to the **Table of Contents**
> view in your PDF viewer.

### Previous Versions

Find installation instructions and binaries on [our website](https://pytorch.org/get-started/previous-versions).

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

*   **Forums:** Discuss implementations and research: [https://discuss.pytorch.org](https://discuss.pytorch.org)
*   **GitHub Issues:** Report bugs, request features, and discuss installation issues.
*   **Slack:** The [PyTorch Slack](https://pytorch.slack.com/) hosts a primary audience of moderate to experienced PyTorch users and developers for general chat, online discussions, collaboration, etc. If you are a beginner looking for help, the primary medium is [PyTorch Forums](https://discuss.pytorch.org). If you need a slack invite, please fill this form: https://goo.gl/forms/PP1AGvNHpSaJP8to1
*   **Newsletter:**  A one-way email newsletter with important announcements: https://eepurl.com/cbG0rv
*   **Facebook Page:** Important announcements: https://www.facebook.com/pytorch
*   **Brand Guidelines:**  Please visit our website at [pytorch.org](https://pytorch.org/)

## Releases and Contributing

PyTorch typically releases three minor versions per year. Report bugs via [GitHub Issues](https://github.com/pytorch/pytorch/issues).

We appreciate all contributions.  For bug fixes, submit PRs without further discussion. For new features, open an issue to discuss before submitting a PR.
See [Contribution page](CONTRIBUTING.md) for details.  See [Release page](RELEASE.md) for release information.

## The Team

PyTorch is a community-driven project. The project is maintained by [Soumith Chintala](http://soumith.ch), [Gregory Chanan](https://github.com/gchanan), [Dmytro Dzhulgakov](https://github.com/dzhulgakov), [Edward Yang](https://github.com/ezyang), and [Nikita Shulga](https://github.com/malfet) with major contributions from many talented individuals.

## License

PyTorch is licensed under a BSD-style license; see the [LICENSE](LICENSE) file.