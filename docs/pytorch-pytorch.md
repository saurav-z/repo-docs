[![PyTorch Logo](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/pytorch-logo-dark.png)](https://github.com/pytorch/pytorch)

# PyTorch: The Flexible and Fast Deep Learning Framework

**PyTorch is a leading open-source machine learning framework, providing a seamless path from research prototyping to production deployment.** Explore the official repository [here](https://github.com/pytorch/pytorch).

**Key Features:**

*   **Tensor Computation with GPU Acceleration:**  Leverage the power of GPUs for fast numerical computation, similar to NumPy but with significant performance gains.
*   **Dynamic Neural Networks:** Build and modify neural networks with unparalleled flexibility, thanks to a tape-based autograd system.
*   **Python-First Design:** Integrate PyTorch effortlessly into your existing Python workflows, with full support for NumPy, SciPy, Cython, and more.
*   **Imperative Programming:**  Enjoy an intuitive, easy-to-debug development experience with an imperative programming style.
*   **Fast and Lean:** Benefit from minimal framework overhead, optimized backends, and efficient memory management.
*   **Easy Extension:**  Create custom neural network modules and easily interface with PyTorch's Tensor API, or utilize existing NumPy-based libraries.

## More About PyTorch

[Learn the basics of PyTorch](https://pytorch.org/tutorials/beginner/basics/intro.html)

PyTorch offers a comprehensive set of tools for deep learning, organized around the following key components:

| Component                   | Description                                                                                                     |
| :-------------------------- | :-------------------------------------------------------------------------------------------------------------- |
| [**torch**](https://pytorch.org/docs/stable/torch.html) | Tensor library, similar to NumPy, with robust GPU support.                                     |
| [**torch.autograd**](https://pytorch.org/docs/stable/autograd.html) | Automatic differentiation engine supporting all differentiable Tensor operations.                      |
| [**torch.jit**](https://pytorch.org/docs/stable/jit.html)  | Compilation stack (TorchScript) for creating serializable and optimizable models.                |
| [**torch.nn**](https://pytorch.org/docs/stable/nn.html) | Neural networks library, tightly integrated with autograd, designed for flexibility.            |
| [**torch.multiprocessing**](https://pytorch.org/docs/stable/multiprocessing.html) | Python multiprocessing with memory sharing of torch Tensors across processes, useful for data loading and Hogwild training |
| [**torch.utils**](https://pytorch.org/docs/stable/data.html)  | DataLoader and other utility functions for data loading and manipulation.                         |

PyTorch is commonly used in two primary ways:

*   **GPU-accelerated NumPy replacement:**  Utilize PyTorch tensors on GPUs for significantly faster computations.
*   **Research platform for deep learning:**  Leverage PyTorch's flexibility and speed for cutting-edge research projects.

### A GPU-Ready Tensor Library

PyTorch introduces Tensors, which function similarly to NumPy's ndarrays, but can reside on either the CPU or the GPU. This allows for efficient acceleration of computations.

![Tensor illustration](./docs/source/_static/img/tensor_illustration.png)

PyTorch provides a vast array of tensor operations, including slicing, indexing, mathematical calculations, linear algebra operations, and reduction functions, all optimized for speed.

### Dynamic Neural Networks: Tape-Based Autograd

PyTorch uses reverse-mode automatic differentiation, allowing for unparalleled flexibility in building and modifying neural networks.

Unlike static graph frameworks, PyTorch's dynamic graphs enable modifications without delays or overhead.

![Dynamic graph](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/dynamic_graph.gif)

### Python First

PyTorch is designed to be deeply integrated into Python, allowing you to use it in a natural way. Easily integrate with NumPy, SciPy, and scikit-learn, create custom layers with Python, Cython, and Numba.

### Imperative Experiences

PyTorch is built to be intuitive, linear in thought, and easy to use. It offers straightforward debugging with clear stack traces and transparent execution.

### Fast and Lean

PyTorch is engineered for speed and efficiency, utilizing libraries like Intel MKL and NVIDIA's cuDNN and NCCL for acceleration. It's designed to minimize overhead and maximize memory efficiency, enabling the training of larger deep learning models.

### Extensions Without Pain

PyTorch simplifies the process of creating new neural network modules. You can write new layers in Python or utilize existing NumPy-based libraries, or write them in C/C++ with the efficient extension API.

## Installation

Detailed installation instructions, including binary installations (Conda and pip) and instructions for installing from source are on our website: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

### Binaries

Commands to install binaries via Conda or pip wheels are on our website: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

#### NVIDIA Jetson Platforms

Python wheels for NVIDIA's Jetson Nano, Jetson TX1/TX2, Jetson Xavier NX/AGX, and Jetson AGX Orin are provided [here](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048) and the L4T container is published [here](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch)

They require JetPack 4.2 and above, and [@dusty-nv](https://github.com/dusty-nv) and [@ptrblck](https://github.com/ptrblck) are maintaining them.

### From Source

#### Prerequisites

*   Python 3.9 or later
*   A compiler with full C++17 support (e.g., clang or gcc 9.4.0+)
*   Visual Studio or Visual Studio Build Tool (Windows only)

    \* PyTorch CI uses Visual C++ BuildTools, which come with Visual Studio Enterprise,
    Professional, or Community Editions. You can also install the build tools from
    https://visualstudio.microsoft.com/visual-cpp-build-tools/. The build tools *do not*
    come with Visual Studio Code by default.

#### NVIDIA CUDA Support

*   Install [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads), [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) v8.5 or above, and a [compatible compiler](https://gist.github.com/ax3l/9489132).

#### AMD ROCm Support

*   Install [AMD ROCm](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html) 4.0 and above.

#### Intel GPU Support

*   Follow the [PyTorch Prerequisites for Intel GPUs](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpus.html) instructions.

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
conda install -c conda-forge libuv=1.39
```

#### Install PyTorch

**On Linux**

If you're compiling for AMD ROCm then first run this command:

```bash
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

```cmd
python -m pip install --no-build-isolation -v -e .
```

**CUDA based build**

[NVTX](https://docs.nvidia.com/gameworks/content/gameworkslibrary/nvtx/nvidia_tools_extension_library_nvtx.htm) is needed to build Pytorch with CUDA.
NVTX is a part of CUDA distributive, where it is called "Nsight Compute". To install it onto an already installed CUDA run CUDA installation once again and check the corresponding checkbox.
Make sure that CUDA with Nsight Compute is installed after Visual Studio.

```cmd
cmd
set CMAKE_INCLUDE_PATH={Your directory}\mkl\include
set LIB={Your directory}\mkl\lib;%LIB%

set CMAKE_GENERATOR_TOOLSET_VERSION=14.27
set DISTUTILS_USE_SDK=1
for /f "usebackq tokens=*" %i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -version [15^,17^) -products * -latest -property installationPath`) do call "%i\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=%CMAKE_GENERATOR_TOOLSET_VERSION%
set CUDAHOSTCXX=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.27.29110\bin\HostX64\x64\cl.exe
python -m pip install --no-build-isolation -v -e .
```

**Intel GPU builds**

Please make sure [the common prerequisites](#prerequisites) as well as [the prerequisites for Intel GPU](#intel-gpu-support) are properly installed and the environment variables are configured prior to starting the build. For build tool support, `Visual Studio 2022` is required.

```cmd
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

```bash
docker run --gpus all --rm -ti --ipc=host pytorch/pytorch:latest
```

#### Building the image yourself

```bash
make -f docker.Makefile
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

```bash
make latexpdf
make LATEXOPTS="-interaction=nonstopmode"
```

### Previous Versions

Installation instructions and binaries for previous PyTorch versions may be found
on [our website](https://pytorch.org/get-started/previous-versions).

## Getting Started

-   [Tutorials](https://pytorch.org/tutorials/)
-   [Examples](https://github.com/pytorch/examples)
-   [API Reference](https://pytorch.org/docs/)
-   [Glossary](https://github.com/pytorch/pytorch/blob/main/GLOSSARY.md)

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

*   [Forums](https://discuss.pytorch.org): Discuss implementations, research, etc.
*   GitHub Issues: Bug reports, feature requests, install issues, RFCs.
*   [Slack](https://pytorch.slack.com/): General chat, discussions, and collaboration.  [Slack invite form](https://goo.gl/forms/PP1AGvNHpSaJP8to1)
*   [Newsletter](https://eepurl.com/cbG0rv): One-way email newsletter with announcements.
*   Facebook Page: Important announcements about PyTorch. https://www.facebook.com/pytorch
*   For brand guidelines, please visit our website at [pytorch.org](https://pytorch.org/)

## Releases and Contributing

Typically, PyTorch releases three minor versions a year.  Report bugs via [GitHub Issues](https://github.com/pytorch/pytorch/issues).  Contribute new features by [following guidelines](CONTRIBUTING.md).

## The Team

PyTorch is a community-driven project maintained by a dedicated team and many contributors.

[Team names and links provided in the original README]

## License

PyTorch is available under a BSD-style license, found in the [LICENSE](LICENSE) file.