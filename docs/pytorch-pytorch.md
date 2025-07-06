<div align="center">
  <img src="https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/pytorch-logo-dark.png" alt="PyTorch Logo" width="200">
  <h1>PyTorch: Deep Learning with Flexibility and Speed</h1>
  <p><em>PyTorch is an open-source machine learning framework that accelerates the path from research prototyping to production deployment.</em></p>
</div>

---

## Key Features of PyTorch

*   **Tensor Computation with GPU Acceleration:**  Leverage the power of GPUs for blazing-fast tensor operations, similar to NumPy but with significant performance gains.
*   **Dynamic Neural Networks:** Build and modify neural networks with ease using a tape-based autograd system, enabling flexible and rapid experimentation.
*   **Python-First Approach:** Seamlessly integrate with Python, utilizing existing libraries like NumPy, SciPy, and Cython for extended functionality.
*   **Imperative Style:** Enjoy an intuitive and easy-to-debug coding experience with PyTorch's imperative execution model.
*   **Fast and Lean:** Benefit from minimal framework overhead and optimized backends for speed and efficiency, especially with Intel MKL and NVIDIA libraries (cuDNN, NCCL).
*   **Easy Extensions:** Extend PyTorch's capabilities by writing custom modules in Python or C/C++ with minimal boilerplate.

---

**[Explore the PyTorch Repository on GitHub](https://github.com/pytorch/pytorch)**

## Comprehensive Guide to PyTorch

### Introduction to PyTorch

[Learn the basics of PyTorch](https://pytorch.org/tutorials/beginner/basics/intro.html)

PyTorch is a versatile Python-based framework designed for deep learning and scientific computing.  It provides two primary high-level features:

*   **torch:** A powerful tensor library offering functionalities similar to NumPy, with built-in support for GPU acceleration.
*   **torch.autograd:** An automatic differentiation engine based on a tape-based system, enabling the construction of dynamic neural networks.

PyTorch offers maximum flexibility and speed. It is used as a replacement for NumPy to leverage the power of GPUs and a deep learning research platform.

### Core Components

PyTorch's core functionality is organized into several key components:

| Component                  | Description                                                                                                     |
| :------------------------- | :-------------------------------------------------------------------------------------------------------------- |
| [**torch**](https://pytorch.org/docs/stable/torch.html)         | The fundamental Tensor library, offering GPU support and NumPy-like functionality.  |
| [**torch.autograd**](https://pytorch.org/docs/stable/autograd.html) | Automatic differentiation for all differentiable Tensor operations in `torch`.            |
| [**torch.jit**](https://pytorch.org/docs/stable/jit.html)         | A compiler to create serializable and optimizable models from PyTorch code (TorchScript). |
| [**torch.nn**](https://pytorch.org/docs/stable/nn.html)          | The neural network library, deeply integrated with autograd for maximum flexibility.    |
| [**torch.multiprocessing**](https://pytorch.org/docs/stable/multiprocessing.html) | Provides memory sharing for Tensors across processes (for data loading, Hogwild training).    |
| [**torch.utils**](https://pytorch.org/docs/stable/data.html)       | Utility functions, including the `DataLoader`, for data handling and more.                  |

### Detailed Features

*   **GPU-Ready Tensor Library**: PyTorch's tensors can reside on CPUs or GPUs, accelerating computations significantly. It includes a wide variety of routines for operations like slicing, indexing, mathematical operations, and linear algebra.

    ![Tensor illustration](./docs/source/_static/img/tensor_illustration.png)

*   **Dynamic Neural Networks: Tape-Based Autograd**: PyTorch utilizes reverse-mode automatic differentiation, enabling users to modify network behavior without overhead. The dynamic graph feature provides flexibility and speed for research applications.

    ![Dynamic graph](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/dynamic_graph.gif)

*   **Python-First**: It deeply integrates with Python, supporting integration with libraries like NumPy, SciPy, and Cython, without reinventing the wheel.

*   **Imperative Experiences**: PyTorch's imperative design makes it easy to understand, debug, and step through your code linearly. Error messages and stack traces point directly to the source of the problem.

*   **Fast and Lean**: It integrates acceleration libraries such as Intel MKL and NVIDIA ([cuDNN](https://developer.nvidia.com/cudnn), [NCCL](https://developer.nvidia.com/nccl)) to maximize speed. The memory usage in PyTorch is optimized to ensure your deep learning models are memory efficient.

*   **Extensions Without Pain**: PyTorch makes writing new neural network modules and interacting with the Tensor API straightforward, using the torch API, the NumPy API and extension API. Tutorials on custom extensions are available in the PyTorch documentation.

## Installation

Installation methods:
*   Binaries
*   From Source
*   Docker Image

### Binaries

Install binaries via Conda or pip wheels are on our website: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

#### NVIDIA Jetson Platforms

*   Python wheels for NVIDIA's Jetson Nano, Jetson TX1/TX2, Jetson Xavier NX/AGX, and Jetson AGX Orin are provided [here](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048).
*   L4T container is published [here](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch)
*   They require JetPack 4.2 and above, and [@dusty-nv](https://github.com/dusty-nv) and [@ptrblck](https://github.com/ptrblck) are maintaining them.

### From Source

#### Prerequisites
*   Python 3.9 or later
*   A compiler that fully supports C++17, such as clang or gcc (gcc 9.4.0 or newer is required, on Linux)
*   Visual Studio or Visual Studio Build Tool (Windows only)

\* PyTorch CI uses Visual C++ BuildTools, which come with Visual Studio Enterprise,
Professional, or Community Editions. You can also install the build tools from
https://visualstudio.microsoft.com/visual-cpp-build-tools/. The build tools *do not*
come with Visual Studio Code by default.

#### NVIDIA CUDA Support
If you want to compile with CUDA support:
*   [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads)
*   [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) v8.5 or above
*   [Compiler](https://gist.github.com/ax3l/9489132) compatible with CUDA

If you want to disable CUDA support, export the environment variable `USE_CUDA=0`.
If you are building for NVIDIA's Jetson platforms, Instructions to install PyTorch for Jetson Nano are [available here](https://devtalk.nvidia.com/default/topic/1049071/jetson-nano/pytorch-for-jetson-nano/)

#### AMD ROCm Support
If you want to compile with ROCm support:
*   [AMD ROCm](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html) 4.0 and above installation
*   ROCm is currently supported only for Linux systems.

By default the build system expects ROCm to be installed in `/opt/rocm`. If ROCm is installed in a different directory, the `ROCM_PATH` environment variable must be set to the ROCm installation directory. The build system automatically detects the AMD GPU architecture. Optionally, the AMD GPU architecture can be explicitly set with the `PYTORCH_ROCM_ARCH` environment variable [AMD GPU architecture](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-gpus)

If you want to disable ROCm support, export the environment variable `USE_ROCM=0`.

#### Intel GPU Support
If you want to compile with Intel GPU support:
*   [PyTorch Prerequisites for Intel GPUs](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpus.html) instructions.
*   Intel GPU is supported for Linux and Windows.

If you want to disable Intel GPU support, export the environment variable `USE_XPU=0`.

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
python setup.py develop
```

**On macOS**

```bash
python3 setup.py develop
```

**On Windows**

If you want to build legacy python code, please refer to [Building on legacy code and CUDA](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md#building-on-legacy-code-and-cuda)

**CPU-only builds**

In this mode PyTorch computations will run on your CPU, not your GPU.

```cmd
python setup.py develop
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

python setup.py develop

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

python setup.py develop
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

```bash
docker run --gpus all --rm -ti --ipc=host pytorch/pytorch:latest
```

#### Building the image yourself

**NOTE:** Must be built with a docker version > 18.06

```bash
make -f docker.Makefile
# images are tagged as docker.io/${your_docker_username}/pytorch
```

### Building the Documentation

To build documentation in various formats, you will need [Sphinx](http://www.sphinx-doc.org)
and the pytorch_sphinx_theme2.

```bash
cd docs/
pip install -r requirements.txt
make html
make serve
```

#### Building a PDF

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

### Previous Versions

Find installation instructions and binaries for previous PyTorch versions on [our website](https://pytorch.org/get-started/previous-versions).

## Getting Started

*   [Tutorials](https://pytorch.org/tutorials/): Start with the tutorials to understand and use PyTorch.
*   [Examples](https://github.com/pytorch/examples): Explore easy-to-understand PyTorch code across various domains.
*   [The API Reference](https://pytorch.org/docs/): Consult the comprehensive API reference for detailed documentation.
*   [Glossary](https://github.com/pytorch/pytorch/blob/main/GLOSSARY.md): Learn the key terminology.

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

*   **Forums**: Discuss implementations, research, etc. [https://discuss.pytorch.org](https://discuss.pytorch.org)
*   **GitHub Issues**: Report bugs, request features, and discuss installations.
*   **Slack**: The [PyTorch Slack](https://pytorch.slack.com/) hosts a primary audience of moderate to experienced PyTorch users and developers for general chat, online discussions, collaboration, etc. If you are a beginner looking for help, the primary medium is [PyTorch Forums](https://discuss.pytorch.org). If you need a slack invite, please fill this form: https://goo.gl/forms/PP1AGvNHpSaJP8to1
*   **Newsletter**: Receive important announcements via email. [Sign-up here](https://eepurl.com/cbG0rv)
*   **Facebook Page**: Follow for important announcements. https://www.facebook.com/pytorch
*   For brand guidelines, please visit our website at [pytorch.org](https://pytorch.org/)

## Releases and Contributing

PyTorch typically releases three minor versions per year. Report bugs via [filing an issue](https://github.com/pytorch/pytorch/issues).

Contributions are welcome. If you're planning to contribute, please:

*   Report bugs directly with fixes.
*   Discuss new features or core extensions by opening an issue before creating a PR.

See the [Contribution page](CONTRIBUTING.md) and [Release page](RELEASE.md) for details.

## The Team

PyTorch is a community-driven project, currently maintained by [Soumith Chintala](http://soumith.ch), [Gregory Chanan](https://github.com/gchanan), [Dmytro Dzhulgakov](https://github.com/dzhulgakov), [Edward Yang](https://github.com/ezyang), and [Nikita Shulga](https://github.com/malfet) with contributions from many skilled engineers and researchers.

## License

PyTorch is licensed under a BSD-style license, found in the [LICENSE](LICENSE) file.