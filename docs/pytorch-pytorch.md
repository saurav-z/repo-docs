<!-- PyTorch Logo -->
![PyTorch Logo](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/pytorch-logo-dark.png)

# PyTorch: The Deep Learning Framework of Choice

**PyTorch is a powerful and flexible open-source deep learning framework that empowers researchers and developers to build and deploy cutting-edge AI models.**  [Explore the PyTorch repository](https://github.com/pytorch/pytorch).

**Key Features:**

*   **Tensor Computation with GPU Acceleration:** Enables fast numerical computation with tensors, similar to NumPy, with seamless GPU support for significant speedups.
*   **Dynamic Neural Networks with Autograd:**  Offers a tape-based autograd system, providing unparalleled flexibility for creating and modifying neural network architectures on the fly.
*   **Python-First Development:** Deeply integrated with Python, allowing users to leverage existing Python tools and libraries like NumPy, SciPy, and Cython for a streamlined workflow.
*   **Imperative Programming:** Provides an intuitive and easy-to-debug imperative programming experience, enabling developers to write code that executes line by line with straightforward stack traces.
*   **Fast and Lean:** Optimized for performance, PyTorch integrates acceleration libraries like Intel MKL, cuDNN, and NCCL, with efficient memory usage for training large models.
*   **Extensible Ecosystem:** Offers a simple API for writing custom neural network modules and interfaces, enabling integration with other libraries and providing maximum flexibility.

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

PyTorch is comprised of the following core components:

| Component                 | Description                                                                              |
| :------------------------ | :--------------------------------------------------------------------------------------- |
| [**torch**](https://pytorch.org/docs/stable/torch.html)       | A Tensor library like NumPy, with strong GPU support                               |
| [**torch.autograd**](https://pytorch.org/docs/stable/autograd.html)  | A tape-based automatic differentiation library that supports all differentiable Tensor operations in torch |
| [**torch.jit**](https://pytorch.org/docs/stable/jit.html)    | A compilation stack (TorchScript) to create serializable and optimizable models from PyTorch code  |
| [**torch.nn**](https://pytorch.org/docs/stable/nn.html)       | A neural networks library deeply integrated with autograd designed for maximum flexibility |
| [**torch.multiprocessing**](https://pytorch.org/docs/stable/multiprocessing.html) | Python multiprocessing, but with magical memory sharing of torch Tensors across processes. Useful for data loading and Hogwild training |
| [**torch.utils**](https://pytorch.org/docs/stable/data.html)    | DataLoader and other utility functions for convenience                              |

PyTorch is commonly used for:

*   Replacing NumPy to leverage GPU power.
*   Deep learning research offering maximum flexibility and speed.

### A GPU-Ready Tensor Library

If you use NumPy, then you have used Tensors (a.k.a. ndarray).

![Tensor illustration](./docs/source/_static/img/tensor_illustration.png)

PyTorch provides Tensors that can live either on the CPU or the GPU, greatly accelerating computation. It provides a wide variety of tensor routines for scientific computation needs, such as slicing, indexing, mathematical operations, linear algebra, and reductions, and they are fast!

### Dynamic Neural Networks: Tape-Based Autograd

PyTorch's unique approach to building neural networks involves using and replaying a tape recorder. This reverse-mode auto-differentiation allows you to change your network's behavior arbitrarily with zero lag or overhead.

![Dynamic graph](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/dynamic_graph.gif)

### Python First

PyTorch is designed to seamlessly integrate with Python. Use it naturally alongside NumPy, SciPy, and scikit-learn, and write new neural network layers in Python, utilizing your favorite libraries and packages such as Cython and Numba.

### Imperative Experiences

PyTorch is built to be intuitive, linear, and easy to use. Code executes instantly, and debugging is straightforward, with clear stack traces pointing to the exact location of your code.

### Fast and Lean

PyTorch minimizes framework overhead, integrating acceleration libraries such as Intel MKL, cuDNN, and NCCL to maximize speed. With mature CPU and GPU Tensor and neural network backends, PyTorch is fast for both small and large neural networks.  PyTorch's memory usage is also very efficient, allowing for the training of larger deep learning models than previous options.

### Extensions Without Pain

Writing new neural network modules or interfacing with PyTorch's Tensor API is designed to be simple. Write new neural network layers in Python using the torch API or your favorite NumPy-based libraries like SciPy. For those wanting to write layers in C/C++, a convenient and efficient extension API is provided with minimal boilerplate.

## Installation

### Binaries

Commands to install binaries via Conda or pip wheels are on our website: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

#### NVIDIA Jetson Platforms

Python wheels for NVIDIA's Jetson Nano, Jetson TX1/TX2, Jetson Xavier NX/AGX, and Jetson AGX Orin are provided [here](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048) and the L4T container is published [here](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch)

They require JetPack 4.2 and above, and [@dusty-nv](https://github.com/dusty-nv) and [@ptrblck](https://github.com/ptrblck) are maintaining them.

### From Source

#### Prerequisites

If you are installing from source, you will need:

*   Python 3.9 or later
*   A compiler that fully supports C++17, such as clang or gcc (gcc 9.4.0 or newer is required, on Linux)
*   Visual Studio or Visual Studio Build Tool (Windows only)

\* PyTorch CI uses Visual C++ BuildTools, which come with Visual Studio Enterprise, Professional, or Community Editions. You can also install the build tools from [https://visualstudio.microsoft.com/visual-cpp-build-tools/](https://visualstudio.microsoft.com/visual-cpp-build-tools/). The build tools *do not* come with Visual Studio Code by default.

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

A conda environment is not required. You can also build PyTorch in a standard virtual environment, e.g., created with tools like `uv`, if your system has all necessary dependencies.

##### NVIDIA CUDA Support

To compile with CUDA support, [select a supported version of CUDA from our support matrix](https://pytorch.org/get-started/locally/), then install:

*   [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads)
*   [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) v8.5 or above
*   [Compiler](https://gist.github.com/ax3l/9489132) compatible with CUDA

Note: You could refer to the [cuDNN Support Matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/support-matrix.html) for cuDNN versions with the various supported CUDA, CUDA driver, and NVIDIA hardware.

If you want to disable CUDA support, export the environment variable `USE_CUDA=0`. Other potentially useful environment variables may be found in `setup.py`. If CUDA is installed in a non-standard location, set PATH so that the nvcc you want to use can be found.

If you are building for NVIDIA's Jetson platforms (Jetson Nano, TX1, TX2, AGX Xavier), Instructions to install PyTorch for Jetson Nano are [available here](https://devtalk.nvidia.com/default/topic/1049071/jetson-nano/pytorch-for-jetson-nano/)

##### AMD ROCm Support

To compile with ROCm support, install:

*   [AMD ROCm](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html) 4.0 and above installation
*   ROCm is currently supported only for Linux systems.

By default, the build system expects ROCm to be installed in `/opt/rocm`. If ROCm is installed in a different directory, the `ROCM_PATH` environment variable must be set to the ROCm installation directory. The build system automatically detects the AMD GPU architecture. Optionally, the AMD GPU architecture can be explicitly set with the `PYTORCH_ROCM_ARCH` environment variable [AMD GPU architecture](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-gpus)

If you want to disable ROCm support, export the environment variable `USE_ROCM=0`. Other potentially useful environment variables may be found in `setup.py`.

##### Intel GPU Support

To compile with Intel GPU support, follow the
*   [PyTorch Prerequisites for Intel GPUs](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpus.html) instructions.
*   Intel GPU is supported for Linux and Windows.

If you want to disable Intel GPU support, export the environment variable `USE_XPU=0`. Other potentially useful environment variables may be found in `setup.py`.

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

You can optionally adjust the configuration of cmake variables (without building first), by doing the following. For example, adjusting the pre-detected directories for CuDNN or BLAS can be done with such a step.

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

You can also pull a pre-built docker image from Docker Hub and run with docker v19.03+

```bash
docker run --gpus all --rm -ti --ipc=host pytorch/pytorch:latest
```

Please note that PyTorch uses shared memory to share data between processes, so if torch multiprocessing is used (e.g.
for multithreaded data loaders) the default shared memory segment size that container runs with is not enough, and you
should increase shared memory size either with `--ipc=host` or `--shm-size` command line options to `nvidia-docker run`.

#### Building the image yourself

**NOTE:** Must be built with a docker version > 18.06

The `Dockerfile` is supplied to build images with CUDA 11.1 support and cuDNN v8.
You can pass `PYTHON_VERSION=x.y` make variable to specify which Python version is to be used by Miniconda, or leave it
unset to use the default.

```bash
make -f docker.Makefile
# images are tagged as docker.io/${your_docker_username}/pytorch
```

You can also pass the `CMAKE_VARS="..."` environment variable to specify additional CMake variables to be passed to CMake during the build.
See [setup.py](./setup.py) for the list of available variables.

```bash
make -f docker.Makefile
```

### Building the Documentation

To build documentation in various formats, you will need [Sphinx](http://www.sphinx-doc.org)
and the pytorch_sphinx_theme2.

Before you build the documentation locally, ensure `torch` is
installed in your environment. For small fixes, you can install the
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

Run `make` to get a list of all available output formats.

If you get a katex error run `npm install katex`.  If it persists, try
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

To compile a PDF of all PyTorch documentation, ensure you have
`texlive` and LaTeX installed. On macOS, you can install them using:

```
brew install --cask mactex
```

To create the PDF:

1.  Run:

    ```
    make latexpdf
    ```

    This will generate the necessary files in the `build/latex` directory.

2.  Navigate to this directory and execute:

    ```
    make LATEXOPTS="-interaction=nonstopmode"
    ```

    This will produce a `pytorch.pdf` with the desired content. Run this
    command one more time so that it generates the correct table
    of contents and index.

> [!NOTE]
> To view the Table of Contents, switch to the **Table of Contents**
> view in your PDF viewer.

### Previous Versions

Installation instructions and binaries for previous PyTorch versions may be found on [our website](https://pytorch.org/get-started/previous-versions).

## Getting Started

Three pointers to get you started:

*   [Tutorials: get you started with understanding and using PyTorch](https://pytorch.org/tutorials/)
*   [Examples: easy to understand PyTorch code across all domains](https://github.com/pytorch/examples)
*   [The API Reference](https://pytorch.org/docs/)
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

*   Forums: Discuss implementations, research, etc. [https://discuss.pytorch.org](https://discuss.pytorch.org)
*   GitHub Issues: Bug reports, feature requests, install issues, RFCs, thoughts, etc.
*   Slack: The [PyTorch Slack](https://pytorch.slack.com/) hosts a primary audience of moderate to experienced PyTorch users and developers for general chat, online discussions, collaboration, etc. If you are a beginner looking for help, the primary medium is [PyTorch Forums](https://discuss.pytorch.org). If you need a slack invite, please fill this form: [https://goo.gl/forms/PP1AGvNHpSaJP8to1](https://goo.gl/forms/PP1AGvNHpSaJP8to1)
*   Newsletter: No-noise, a one-way email newsletter with important announcements about PyTorch. You can sign-up here: [https://eepurl.com/cbG0rv](https://eepurl.com/cbG0rv)
*   Facebook Page: Important announcements about PyTorch. [https://www.facebook.com/pytorch](https://www.facebook.com/pytorch)
*   For brand guidelines, please visit our website at [pytorch.org](https://pytorch.org/)

## Releases and Contributing

PyTorch typically has three minor releases a year. Please report bugs by [filing an issue](https://github.com/pytorch/pytorch/issues).

Contributions are welcome. Contribute bug fixes directly; for new features, discuss them with us by opening an issue.  See our [Contribution page](CONTRIBUTING.md) and [Release page](RELEASE.md).

## The Team

PyTorch is community-driven.  The project is currently maintained by [Soumith Chintala](http://soumith.ch), [Gregory Chanan](https://github.com/gchanan), [Dmytro Dzhulgakov](https://github.com/dzhulgakov), [Edward Yang](https://github.com/ezyang), and [Nikita Shulga](https://github.com/malfet) with major contributions from many talented individuals.

Note: This project is unrelated to [hughperkins/pytorch](https://github.com/hughperkins/pytorch).

## License

PyTorch is licensed under a BSD-style license, found in the [LICENSE](LICENSE) file.