![PyTorch Logo](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/pytorch-logo-dark.png)

# PyTorch: The Deep Learning Framework for Rapid Prototyping and Production

PyTorch is a powerful and flexible open-source machine learning framework that empowers researchers and developers to build cutting-edge deep learning models. [Explore the PyTorch repository](https://github.com/pytorch/pytorch).

**Key Features:**

*   **Tensor Computation with GPU Acceleration:**  Leverage the power of GPUs for fast tensor computations, similar to NumPy, accelerating your projects.
*   **Dynamic Neural Networks:**  Build and modify neural networks on the fly using a tape-based autograd system, providing flexibility.
*   **Python-First Design:**  Seamlessly integrate with your existing Python workflow, including popular libraries like NumPy, SciPy, and Cython.
*   **Imperative Programming:**  Enjoy an intuitive and easy-to-debug imperative programming style.
*   **Fast and Lean:** Benefit from minimal framework overhead and optimized acceleration libraries.
*   **Extensible Architecture:**  Easily create custom neural network modules or extend PyTorch's Tensor API with minimal effort.

## Core Components

PyTorch is comprised of these key components:

| Component                     | Description                                                                                                                      |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| [`torch`](https://pytorch.org/docs/stable/torch.html)      | A Tensor library, similar to NumPy, with GPU support.                                                  |
| [`torch.autograd`](https://pytorch.org/docs/stable/autograd.html) | Automatic differentiation library, supporting all differentiable Tensor operations.                  |
| [`torch.jit`](https://pytorch.org/docs/stable/jit.html)     | Compilation stack (TorchScript) for serializing and optimizing PyTorch code.                        |
| [`torch.nn`](https://pytorch.org/docs/stable/nn.html)       | Neural network library, integrated with autograd for maximum flexibility.                          |
| [`torch.multiprocessing`](https://pytorch.org/docs/stable/multiprocessing.html) | Multiprocessing with memory sharing for Tensors across processes (useful for data loading and training). |
| [`torch.utils`](https://pytorch.org/docs/stable/data.html)    | Utility functions, including `DataLoader` for data management.                                   |

## Getting Started

[Learn the basics of PyTorch](https://pytorch.org/tutorials/beginner/basics/intro.html).

### A GPU-Ready Tensor Library

PyTorch provides Tensors that can run on CPU or GPU and significantly speeds up computation.  PyTorch offers extensive tensor operations for various scientific and mathematical needs, including slicing, indexing, and linear algebra, designed for speed.

### Dynamic Neural Networks: Tape-Based Autograd

PyTorch uses reverse-mode auto-differentiation, enabling flexible network changes without delays.  This approach provides both speed and flexibility for advanced research, as illustrated by this [dynamic graph](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/dynamic_graph.gif).

### Python-First Development

PyTorch is deeply integrated with Python, making it easy to use like NumPy, SciPy, and scikit-learn.  You can write custom neural network layers in Python, utilizing packages like Cython and Numba.

### Imperative Experience

PyTorch offers a clear and intuitive coding experience, making debugging straightforward.  Errors are easily traceable with clear stack traces.

### Fast and Lean Performance

PyTorch leverages acceleration libraries like Intel MKL and NVIDIA (cuDNN, NCCL) for optimal speed. It offers memory efficiency, allowing you to train large deep learning models, with custom memory allocators for GPUs.

### Easy Extension

Creating new modules and interfacing with the Tensor API is simple and intuitive.  You can write layers in Python or use the convenient extension API for C/C++, minimizing the boilerplate.

## Installation

Installation instructions are available on our website: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/).

### Binaries

Install binaries using Conda or pip wheels.

#### NVIDIA Jetson Platforms

Pre-built wheels for NVIDIA's Jetson Nano, Jetson TX1/TX2, Jetson Xavier NX/AGX, and Jetson AGX Orin: [here](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048) and the L4T container is published [here](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch)

### From Source

#### Prerequisites

Install prerequisites:
*   Python 3.9 or later
*   C++17 compliant compiler (e.g., clang or gcc 9.4.0 or newer)
*   Visual Studio or Visual Studio Build Tool (Windows)

#### NVIDIA CUDA Support
*   NVIDIA CUDA (refer to our support matrix:  [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)).
*   NVIDIA cuDNN v8.5 or above
*   Compatible Compiler

#### AMD ROCm Support
*   [AMD ROCm](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html) 4.0 and above installation
*   ROCm is currently supported only for Linux systems.

#### Intel GPU Support
*   Follow these
*   [PyTorch Prerequisites for Intel GPUs](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpus.html) instructions.
*   Intel GPU is supported for Linux and Windows.

#### Get the PyTorch Source

```bash
git clone https://github.com/pytorch/pytorch
cd pytorch
git submodule sync
git submodule update --init --recursive
```

#### Install Dependencies

```bash
conda install cmake ninja
pip install -r requirements.txt
```

**On Linux**

```bash
pip install mkl-static mkl-include
.ci/docker/common/install_magma_conda.sh 12.4
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
conda install -c conda-forge libuv
```

#### Install PyTorch
```bash
export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"
python -m pip install --no-build-isolation -v -e .
```

**Build for ROCm Support**

```bash
python tools/amd_build/build_amd.py
```

**CPU-only builds**

```cmd
python -m pip install --no-build-isolation -v -e .
```

**CUDA based build**

```cmd
:: Set the environment variables after you have downloaded and unzipped the mkl package,
:: else CMake would throw an error as `Could NOT find OpenMP`.
set CMAKE_INCLUDE_PATH={Your directory}\mkl\include
set LIB={Your directory}\mkl\lib;%LIB%
set CMAKE_GENERATOR_TOOLSET_VERSION=14.27
set DISTUTILS_USE_SDK=1
for /f "usebackq tokens=*" %i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -version [15^,17^) -products * -latest -property installationPath`) do call "%i\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=%CMAKE_GENERATOR_TOOLSET_VERSION%
set CUDAHOSTCXX=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.27.29110\bin\HostX64\x64\cl.exe

python -m pip install --no-build-isolation -v -e .
```

**Intel GPU builds**
```cmd
if defined CMAKE_PREFIX_PATH (
    set "CMAKE_PREFIX_PATH=%CONDA_PREFIX%\Library;%CMAKE_PREFIX_PATH%"
) else (
    set "CMAKE_PREFIX_PATH=%CONDA_PREFIX%\Library"
)

python -m pip install --no-build-isolation -v -e .
```

#### Adjust Build Options (Optional)
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

Find installation instructions and binaries: [https://pytorch.org/get-started/previous-versions](https://pytorch.org/get-started/previous-versions)

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
*   Slack: [PyTorch Slack](https://pytorch.slack.com/)
*   Newsletter: [https://eepurl.com/cbG0rv](https://eepurl.com/cbG0rv)
*   Facebook Page: [https://www.facebook.com/pytorch](https://www.facebook.com/pytorch)

## Releases and Contributing

We welcome contributions!  Please report bugs or request features via [filing an issue](https://github.com/pytorch/pytorch/issues).

See our [Contribution page](CONTRIBUTING.md) and [Release page](RELEASE.md).

## The Team

Maintained by [Soumith Chintala](http://soumith.ch), [Gregory Chanan](https://github.com/gchanan), [Dmytro Dzhulgakov](https://github.com/dzhulgakov), [Edward Yang](https://github.com/ezyang), and [Nikita Shulga](https://github.com/malfet) with contributions from many others.

## License

PyTorch is available under a BSD-style license, as found in the [LICENSE](LICENSE) file.