[![PyTorch Logo](https://github.com/pytorch/pytorch/blob/9708fcf92db88b80b9010c68662d634434da3106/docs/source/_static/img/pytorch-logo-dark.png)](https://github.com/pytorch/pytorch)

## PyTorch: Deep Learning with Speed and Flexibility

**PyTorch** is a powerful and versatile deep learning framework, offering both GPU-accelerated tensor computation and dynamic neural networks built on a tape-based autograd system.  It seamlessly integrates with Python and is designed for both research and production.  Get started with PyTorch: [https://github.com/pytorch/pytorch](https://github.com/pytorch/pytorch).

**Key Features:**

*   **Tensor Computation with GPU Acceleration:**  Leverage the power of GPUs for fast tensor operations, similar to NumPy but with significant performance gains.
*   **Dynamic Neural Networks with Autograd:**  Build and modify neural networks with unparalleled flexibility using reverse-mode auto-differentiation.
*   **Python-First Development:** Enjoy deep integration with Python, allowing you to reuse your favorite libraries like NumPy, SciPy, and Cython.
*   **Imperative Programming Style:** Benefit from an intuitive and easy-to-debug coding experience.
*   **Fast and Lean:** Experience minimal framework overhead and optimized performance with acceleration libraries like Intel MKL and NVIDIA cuDNN.
*   **Seamless Extension:** Easily write custom neural network modules in Python or C/C++ to expand your capabilities.

### Table of Contents

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

PyTorch is a library consisting of the following components:

| Component | Description |
| ---- | --- |
| [**torch**](https://pytorch.org/docs/stable/torch.html) | A Tensor library like NumPy, with strong GPU support |
| [**torch.autograd**](https://pytorch.org/docs/stable/autograd.html) | A tape-based automatic differentiation library that supports all differentiable Tensor operations in torch |
| [**torch.jit**](https://pytorch.org/docs/stable/jit.html) | A compilation stack (TorchScript) to create serializable and optimizable models from PyTorch code  |
| [**torch.nn**](https://pytorch.org/docs/stable/nn.html) | A neural networks library deeply integrated with autograd designed for maximum flexibility |
| [**torch.multiprocessing**](https://pytorch.org/docs/stable/multiprocessing.html) | Python multiprocessing, but with magical memory sharing of torch Tensors across processes. Useful for data loading and Hogwild training |
| [**torch.utils**](https://pytorch.org/docs/stable/data.html) | DataLoader and other utility functions for convenience |

PyTorch is commonly used as:

*   A high-performance replacement for NumPy, leveraging GPUs.
*   A flexible deep learning research platform.

### A GPU-Ready Tensor Library

PyTorch Tensors are similar to NumPy's ndarrays but can reside on either the CPU or GPU, drastically accelerating computations.

![Tensor illustration](https://github.com/pytorch/pytorch/blob/9708fcf92db88b80b9010c68662d634434da3106/docs/source/_static/img/tensor_illustration.png)

PyTorch offers various tensor operations like slicing, indexing, and mathematical operations optimized for speed.

### Dynamic Neural Networks: Tape-Based Autograd

PyTorch utilizes reverse-mode auto-differentiation (tape recording) for building neural networks, allowing for flexible network behavior changes without performance overhead. This approach offers a fast and flexible environment for research, compared to frameworks with a static graph structure.

![Dynamic graph](https://github.com/pytorch/pytorch/blob/9708fcf92db88b80b9010c68662d634434da3106/docs/source/_static/img/dynamic_graph.gif)

### Python First

PyTorch seamlessly integrates with Python and avoids being just a binding to a monolithic C++ framework.  It allows you to write custom layers in Python, use libraries like Cython, and leverage existing tools like NumPy, SciPy, and scikit-learn.

### Imperative Experiences

PyTorch is designed for an intuitive, easy-to-debug imperative programming style.  Stack traces are clear and point directly to your code, avoiding the frustrations of asynchronous execution engines.

### Fast and Lean

PyTorch minimizes framework overhead and integrates acceleration libraries such as Intel MKL and NVIDIA cuDNN/NCCL to maximize speed. The CPU and GPU tensor and neural network backends are well-tested and mature.

Additionally, memory usage is highly efficient, enabling the training of larger deep learning models.

### Extensions Without Pain

Creating new neural network modules or interfacing with PyTorch's Tensor API is straightforward. You can implement new layers in Python, leveraging the torch API or existing NumPy-based libraries. The extension API also provides efficient ways to write layers in C/C++ with minimal boilerplate.

## Installation

### Binaries

Install binary packages using Conda or pip: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

#### NVIDIA Jetson Platforms

Pre-built Python wheels for NVIDIA Jetson platforms (Nano, TX1/TX2, Xavier NX/AGX, and AGX Orin) are available: [https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048).  They require JetPack 4.2 and above.

### From Source

#### Prerequisites

*   Python 3.9 or later
*   A C++17-compliant compiler (e.g., clang or gcc; gcc 9.4.0 or newer on Linux)
*   Visual Studio or Visual Studio Build Tool (Windows)

\*  PyTorch CI uses Visual C++ BuildTools from Visual Studio Enterprise, Professional, or Community Editions.  The build tools *do not* come with Visual Studio Code by default.

Example setup:

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

#### NVIDIA CUDA Support

To compile with CUDA support:

*   Install [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads), [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) v8.5 or above, and a [compiler](https://gist.github.com/ax3l/9489132) compatible with CUDA.
*   Refer to the [cuDNN Support Matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/support-matrix.html).
*   Set the environment variable `USE_CUDA=0` to disable CUDA.

#### AMD ROCm Support

To compile with ROCm support:

*   Install [AMD ROCm](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html) 4.0 and above.
*   ROCm is currently only supported on Linux.
*   Set `ROCM_PATH` if ROCm is not installed in `/opt/rocm`.
*   Use `PYTORCH_ROCM_ARCH` to explicitly set the AMD GPU architecture.
*   Set the environment variable `USE_ROCM=0` to disable ROCm.

#### Intel GPU Support

To compile with Intel GPU support:

*   Follow the [PyTorch Prerequisites for Intel GPUs](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpus.html) instructions.
*   Intel GPU is supported on Linux and Windows.
*   Set the environment variable `USE_XPU=0` to disable Intel GPU support.

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
conda install -c conda-forge libuv
```

#### Install PyTorch

**On Linux**

```bash
python tools/amd_build/build_amd.py # Only run if compiling for AMD ROCm
export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"
python -m pip install --no-build-isolation -v -e .
```

**On macOS**

```bash
python -m pip install --no-build-isolation -v -e .
```

**On Windows**

For legacy python code builds, see [Building on legacy code and CUDA](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md#building-on-legacy-code-and-cuda).

**CPU-only builds**

```cmd
python -m pip install --no-build-isolation -v -e .
```

**CUDA based build**

Refer to the [build_pytorch.bat](https://github.com/pytorch/pytorch/blob/main/.ci/pytorch/win-test-helpers/build_pytorch.bat) script for more environment variable configurations

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

Instructions are on [our website](https://pytorch.org/get-started/previous-versions).

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

*   Forums: [https://discuss.pytorch.org](https://discuss.pytorch.org)
*   GitHub Issues
*   Slack: [PyTorch Slack](https://pytorch.slack.com/) - Get an invite:  [https://goo.gl/forms/PP1AGvNHpSaJP8to1](https://goo.gl/forms/PP1AGvNHpSaJP8to1)
*   Newsletter:  [https://eepurl.com/cbG0rv](https://eepurl.com/cbG0rv)
*   Facebook Page: [https://www.facebook.com/pytorch](https://www.facebook.com/pytorch)
*   Brand Guidelines: [pytorch.org](https://pytorch.org/)

## Releases and Contributing

Typically, PyTorch has three minor releases a year.  Report bugs via [filing an issue](https://github.com/pytorch/pytorch/issues).

We appreciate contributions.  Contribute bug fixes directly. For new features, discuss them first via an issue.  See our [Contribution page](CONTRIBUTING.md) and [Release page](RELEASE.md).

## The Team

Maintained by [Soumith Chintala](http://soumith.ch), [Gregory Chanan](https://github.com/gchanan), [Dmytro Dzhulgakov](https://github.com/dzhulgakov), [Edward Yang](https://github.com/ezyang), [Alban Desmaison](https://github.com/albanD), [Piotr Bialecki](https://github.com/ptrblck) and [Nikita Shulga](https://github.com/malfet) along with many contributors.

## License

PyTorch uses a BSD-style license, as found in the [LICENSE](LICENSE) file.