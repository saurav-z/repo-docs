[![PyTorch Logo](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/pytorch-logo-dark.png)](https://github.com/pytorch/pytorch)

# PyTorch: The Flexible and Powerful Deep Learning Framework

**PyTorch empowers researchers and developers to build and deploy machine learning models with ease, offering flexibility, speed, and a Python-first approach.**  [Visit the original repository](https://github.com/pytorch/pytorch).

**Key Features:**

*   **GPU-Accelerated Tensor Computations:** Utilize powerful GPUs for fast numerical computations, mirroring the functionality of NumPy with added acceleration.
*   **Dynamic Neural Networks with Autograd:** Build and modify neural networks with unmatched flexibility using a tape-based automatic differentiation system for optimal speed.
*   **Python-First Development:** Seamlessly integrate with existing Python tools and libraries like NumPy, SciPy, and Cython for a natural and intuitive coding experience.
*   **Imperative and Intuitive:** Enjoy an imperative programming style that makes debugging and understanding your code straightforward.
*   **Fast and Lean:** Benefit from minimal overhead and optimized integrations with libraries like Intel MKL, cuDNN, and NCCL, ensuring high performance.
*   **Easy Extensibility:** Effortlessly write custom neural network modules and extend the framework using Python or C/C++.

**Key Components:**

*   **torch:** The tensor library, similar to NumPy but with GPU support.
*   **torch.autograd:**  The automatic differentiation engine.
*   **torch.jit:** The compilation stack for optimizing models.
*   **torch.nn:**  The neural network library with flexible design.
*   **torch.multiprocessing:**  Multiprocessing with shared tensors across processes.
*   **torch.utils:** Data loading and utility functions.

---
## Table of Contents
- [Installation](#installation)
  - [Binaries](#binaries)
    - [NVIDIA Jetson Platforms](#nvidia-jetson-platforms)
  - [From Source](#from-source)
    - [Prerequisites](#prerequisites)
      - [NVIDIA CUDA Support](#nvidia-cuda-support)
      - [AMD ROCm Support](#amd-rocm-support)
      - [Intel GPU Support](#intel-gpu-support)
    - [Get the PyTorch Source](#get-the-pytorch-source)
    - [Install Dependencies](#install-dependencies)
    - [Install PyTorch](#install-pytorch)
      - [Adjust Build Options (Optional)](#adjust-build-options-optional)
  - [Docker Image](#docker-image)
    - [Using pre-built images](#using-pre-built-images)
    - [Building the image yourself](#building-the-image-yourself)
  - [Building the Documentation](#building-the-documentation)
    - [Building a PDF](#building-a-pdf)
  - [Previous Versions](#previous-versions)
- [Getting Started](#getting-started)
- [Resources](#resources)
- [Communication](#communication)
- [Releases and Contributing](#releases-and-contributing)
- [The Team](#the-team)
- [License](#license)

## Installation

### Binaries
Commands to install binaries via Conda or pip wheels are on our website: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

#### NVIDIA Jetson Platforms

Python wheels for NVIDIA's Jetson Nano, Jetson TX1/TX2, Jetson Xavier NX/AGX, and Jetson AGX Orin are provided [here](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048) and the L4T container is published [here](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch)

They require JetPack 4.2 and above, and [@dusty-nv](https://github.com/dusty-nv) and [@ptrblck](https://github.com/ptrblck) are maintaining them.

### From Source

#### Prerequisites

*   Python 3.9 or later
*   A C++17 compliant compiler (e.g., GCC 9.4.0+ on Linux, or Visual Studio)
*   A build tool (e.g., Visual Studio Build Tools for Windows)

#### NVIDIA CUDA Support
If you want to compile with CUDA support, [select a supported version of CUDA from our support matrix](https://pytorch.org/get-started/locally/), then install the following:
-   [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads)
-   [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) v8.5 or above
-   [Compiler](https://gist.github.com/ax3l/9489132) compatible with CUDA

Note: You could refer to the [cuDNN Support Matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/support-matrix.html) for cuDNN versions with the various supported CUDA, CUDA driver, and NVIDIA hardware.

If you want to disable CUDA support, export the environment variable `USE_CUDA=0`.
Other potentially useful environment variables may be found in `setup.py`.  If
CUDA is installed in a non-standard location, set PATH so that the nvcc you
want to use can be found (e.g., `export PATH=/usr/local/cuda-12.8/bin:$PATH`).

If you are building for NVIDIA's Jetson platforms (Jetson Nano, TX1, TX2, AGX Xavier), Instructions to install PyTorch for Jetson Nano are [available here](https://devtalk.nvidia.com/default/topic/1049071/jetson-nano/pytorch-for-jetson-nano/)

#### AMD ROCm Support
If you want to compile with ROCm support, install
-   [AMD ROCm](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html) 4.0 and above installation
-   ROCm is currently supported only for Linux systems.

By default the build system expects ROCm to be installed in `/opt/rocm`. If ROCm is installed in a different directory, the `ROCM_PATH` environment variable must be set to the ROCm installation directory. The build system automatically detects the AMD GPU architecture. Optionally, the AMD GPU architecture can be explicitly set with the `PYTORCH_ROCM_ARCH` environment variable [AMD GPU architecture](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-gpus)

If you want to disable ROCm support, export the environment variable `USE_ROCM=0`.
Other potentially useful environment variables may be found in `setup.py`.

#### Intel GPU Support
If you want to compile with Intel GPU support, follow these
-   [PyTorch Prerequisites for Intel GPUs](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpus.html) instructions.
-   Intel GPU is supported for Linux and Windows.

If you want to disable Intel GPU support, export the environment variable `USE_XPU=0`.
Other potentially useful environment variables may be found in `setup.py`.

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
conda install -c conda-forge libuv
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

1. Run:

   ```
   make latexpdf
   ```

   This will generate the necessary files in the `build/latex` directory.

2. Navigate to this directory and execute:

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

Installation instructions and binaries for previous PyTorch versions may be found
on [our website](https://pytorch.org/get-started/previous-versions).

## Getting Started

Start exploring PyTorch with these resources:

*   [Tutorials](https://pytorch.org/tutorials/): Learn the fundamentals and build your skills.
*   [Examples](https://github.com/pytorch/examples): Explore code examples across diverse domains.
*   [API Reference](https://pytorch.org/docs/): Find detailed information on all available functions and classes.
*   [Glossary](https://github.com/pytorch/pytorch/blob/main/GLOSSARY.md): Understand key concepts and terminology.

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

*   **Forums:** Discuss implementations and research at https://discuss.pytorch.org.
*   **GitHub Issues:** Report bugs, request features, and discuss installations.
*   **Slack:** Join the [PyTorch Slack](https://pytorch.slack.com/) for general chat, collaboration, and discussions (primarily for experienced users). Beginners should use the [PyTorch Forums](https://discuss.pytorch.org).  Get a Slack invite via: https://goo.gl/forms/PP1AGvNHpSaJP8to1
*   **Newsletter:** Sign up for the no-noise, one-way email newsletter for important PyTorch announcements: https://eepurl.com/cbG0rv
*   **Facebook Page:** Stay up-to-date with announcements: https://www.facebook.com/pytorch
*   For brand guidelines, please visit our website at [pytorch.org](https://pytorch.org/)

## Releases and Contributing

PyTorch releases typically occur three times per year.  Report bugs by [filing an issue](https://github.com/pytorch/pytorch/issues).

We welcome contributions!  For bug fixes, contribute directly.  For new features, first discuss them in an issue.  See our [Contribution page](CONTRIBUTING.md) and [Release page](RELEASE.md) for more information.

## The Team

PyTorch is a community-driven project led by talented engineers and researchers.

The project is currently maintained by [Soumith Chintala](http://soumith.ch), [Gregory Chanan](https://github.com/gchanan), [Dmytro Dzhulgakov](https://github.com/dzhulgakov), [Edward Yang](https://github.com/ezyang), and [Nikita Shulga](https://github.com/malfet), with significant contributions from numerous individuals, including [Trevor Killeen](https://github.com/killeent), [Sasank Chilamkurthy](https://github.com/chsasank), [Sergey Zagoruyko](https://github.com/szagoruyko), [Adam Lerer](https://github.com/adamlerer), [Francisco Massa](https://github.com/fmassa), [Alykhan Tejani](https://github.com/alykhantejani), [Luca Antiga](https://github.com/lantiga), [Alban Desmaison](https://github.com/albanD), [Andreas Koepf](https://github.com/andreaskoepf), [James Bradbury](https://github.com/jekbradbury), [Zeming Lin](https://github.com/ebetica), [Yuandong Tian](https://github.com/yuandong-tian), [Guillaume Lample](https://github.com/glample), [Marat Dukhan](https://github.com/Maratyszcza), [Natalia Gimelshein](https://github.com/ngimel), [Christian Sarofeen](https://github.com/csarofeen), [Martin Raison](https://github.com/martinraison), [Edward Yang](https://github.com/ezyang), [Zachary Devito](https://github.com/zdevito). <!-- codespell:ignore -->

Note: This project is unrelated to [hughperkins/pytorch](https://github.com/hughperkins/pytorch) with the same name. Hugh is a valuable contributor to the Torch community and has helped with many things Torch and PyTorch.

## License

PyTorch is licensed under a BSD-style license, as found in the [LICENSE](LICENSE) file.