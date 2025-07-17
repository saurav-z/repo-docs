![PyTorch Logo](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/pytorch-logo-dark.png)

# PyTorch: Open Source Deep Learning Framework

**PyTorch is a powerful Python-based deep learning framework that provides flexibility, speed, and ease of use for researchers and developers.** Find the original repository [here](https://github.com/pytorch/pytorch).

**Key Features:**

*   **Tensor Computation with GPU Acceleration:** Leverage the power of GPUs for fast numerical computation, similar to NumPy.
*   **Dynamic Neural Networks:** Build and modify neural networks on the fly with a tape-based autograd system.
*   **Python-First Design:** Seamlessly integrates with Python, using familiar libraries like NumPy and SciPy.
*   **Imperative Programming Style:** Intuitive and easy-to-debug code execution with clear stack traces.
*   **Fast and Lean:** Optimized for speed and memory efficiency, utilizing libraries like Intel MKL, cuDNN, and NCCL.
*   **Extensible:** Easy to write custom neural network modules and integrate with existing Python and C/C++ code.

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

PyTorch offers a modular structure with the following key components:

| Component                  | Description                                                                                              |
| -------------------------- | -------------------------------------------------------------------------------------------------------- |
| [**torch**](https://pytorch.org/docs/stable/torch.html)           | Tensor library with NumPy-like functionality and GPU support.                     |
| [**torch.autograd**](https://pytorch.org/docs/stable/autograd.html)   | Automatic differentiation library for all differentiable Tensor operations in torch. |
| [**torch.jit**](https://pytorch.org/docs/stable/jit.html)        | Compilation stack (TorchScript) to create serializable and optimizable models.      |
| [**torch.nn**](https://pytorch.org/docs/stable/nn.html)             | Neural network library integrated with autograd for flexibility.                      |
| [**torch.multiprocessing**](https://pytorch.org/docs/stable/multiprocessing.html) | Python multiprocessing with magical memory sharing of torch Tensors across processes |
| [**torch.utils**](https://pytorch.org/docs/stable/data.html)          | DataLoader and other utility functions for convenience.                        |

PyTorch is commonly used for:

*   Replacing NumPy to harness the power of GPUs.
*   Building a research platform offering maximum flexibility and speed in deep learning.

### A GPU-Ready Tensor Library

PyTorch provides tensors (similar to NumPy's ndarrays) that can reside on CPUs or GPUs, significantly accelerating computations.  It provides a rich set of tensor operations for all your scientific computing needs, including slicing, indexing, and linear algebra, with a focus on speed.

![Tensor illustration](./docs/source/_static/img/tensor_illustration.png)

### Dynamic Neural Networks: Tape-Based Autograd

PyTorch uses a tape recorder to build neural networks, enabling dynamic behavior. Unlike static graph frameworks, PyTorch's reverse-mode auto-differentiation allows for flexible network design and modifications without performance overhead.

![Dynamic graph](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/dynamic_graph.gif)

### Python First

PyTorch is deeply integrated with Python, allowing you to leverage familiar tools like NumPy, SciPy, and scikit-learn. It's designed to feel natural for Python users.

### Imperative Experiences

PyTorch offers an intuitive, imperative programming style, where code executes line by line. This makes debugging straightforward, with clear error messages and stack traces.

### Fast and Lean

PyTorch minimizes framework overhead and integrates acceleration libraries like Intel MKL, cuDNN, and NCCL.  Its mature CPU and GPU backends provide high performance for both small and large networks, and it uses custom GPU memory allocators for maximum efficiency.

### Extensions Without Pain

PyTorch allows easy creation of new neural network modules and interaction with its Tensor API.  You can create new layers using the `torch` API or NumPy-based libraries or utilize an extension API for C/C++ layers with minimal boilerplate.

## Installation

### Binaries

Install binaries easily via Conda or pip wheels: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

#### NVIDIA Jetson Platforms

Pre-built wheels are available for NVIDIA Jetson Nano, TX1/TX2, Xavier NX/AGX, and AGX Orin platforms: [here](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048) and the L4T container is published [here](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch). Requires JetPack 4.2+.

### From Source

#### Prerequisites

*   Python 3.9 or later
*   C++17 compatible compiler (gcc 9.4.0+ on Linux)
*   Visual Studio or Visual Studio Build Tool (Windows)

\* PyTorch CI uses Visual C++ BuildTools. You can install the build tools from
https://visualstudio.microsoft.com/visual-cpp-build-tools/.

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

A conda environment is not required. You can also do a PyTorch build in a standard virtual environment.

##### NVIDIA CUDA Support

1.  Select a supported CUDA version from our support matrix: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/).
2.  Install:
    *   [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads)
    *   [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) v8.5 or above
    *   [Compiler](https://gist.github.com/ax3l/9489132) compatible with CUDA

Note: Refer to the [cuDNN Support Matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/support-matrix.html) for compatibility details.

To disable CUDA, set `USE_CUDA=0`.

##### AMD ROCm Support

1.  Install:
    *   [AMD ROCm](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html) 4.0 and above
    *   ROCm is supported only for Linux systems.

By default, the build system expects ROCm to be installed in `/opt/rocm`. Set `ROCM_PATH` if installed elsewhere. Set `PYTORCH_ROCM_ARCH` to specify the AMD GPU architecture.

To disable ROCm, set `USE_ROCM=0`.

##### Intel GPU Support

1.  Follow [PyTorch Prerequisites for Intel GPUs](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpus.html) instructions.
2.  Intel GPU is supported for Linux and Windows.

To disable Intel GPU, set `USE_XPU=0`.

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

If compiling for AMD ROCm, run:

```bash
python tools/amd_build/build_amd.py
```

Install PyTorch:

```bash
export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"
python -m pip install -r requirements-build.txt
python -m pip install --no-build-isolation -v -e .
```

**On macOS**

```bash
python -m pip install -r requirements-build.txt
python -m pip install --no-build-isolation -v -e .
```

**On Windows**

If building legacy python code, refer to [Building on legacy code and CUDA](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md#building-on-legacy-code-and-cuda)

**CPU-only builds**

```cmd
python -m pip install --no-build-isolation -v -e .
```

Note on OpenMP: iomp is the desired OpenMP implementation. To link against iomp, you need to manually download the library and set up the building environment by tweaking `CMAKE_INCLUDE_PATH` and `LIB`. Without these configurations for CMake, Microsoft Visual C OpenMP runtime (vcomp) will be used.

**CUDA based build**

[NVTX](https://docs.nvidia.com/gameworks/content/gameworkslibrary/nvtx/nvidia_tools_extension_library_nvtx.htm) is needed to build Pytorch with CUDA.  NVTX is a part of CUDA distributive, where it is called "Nsight Compute".
Make sure that CUDA with Nsight Compute is installed after Visual Studio.

Additional libraries such as [Magma](https://developer.nvidia.com/magma), [oneDNN, a.k.a. MKLDNN or DNNL](https://github.com/oneapi-src/oneDNN), and [Sccache](https://github.com/mozilla/sccache) are often needed. Refer to the [installation-helper](https://github.com/pytorch/pytorch/tree/main/.ci/pytorch/win-test-helpers/installation-helpers) to install them.

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

Make sure [the common prerequisites](#prerequisites) and [the prerequisites for Intel GPU](#intel-gpu-support) are installed and the environment variables are configured. For build tool support, `Visual Studio 2022` is required.

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

Adjust CMake variables with these steps:

**On Linux**

```bash
export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"
CMAKE_ONLY=1 python setup.py build
ccmake build  # or cmake-gui build
```

**On macOS**

```bash
export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"
MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ CMAKE_ONLY=1 python setup.py build
ccmake build  # or cmake-gui build
```

### Docker Image

#### Using pre-built images

Pull pre-built images from Docker Hub:

```bash
docker run --gpus all --rm -ti --ipc=host pytorch/pytorch:latest
```

Increase shared memory with `--ipc=host` or `--shm-size` for multiprocessing.

#### Building the image yourself

**NOTE:** Must be built with a docker version > 18.06

```bash
make -f docker.Makefile
# images are tagged as docker.io/${your_docker_username}/pytorch
```

Use `CMAKE_VARS="..."` to pass additional CMake variables. See [setup.py](./setup.py).

### Building the Documentation

Install dependencies and build HTML documentation:

```bash
cd docs/
pip install -r requirements.txt
make html
make serve
```

Run `make` for available output formats.

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

Install `texlive` and LaTeX.

1.  Run: `make latexpdf` (generates files in `build/latex`).
2.  Navigate to `build/latex` and execute: `make LATEXOPTS="-interaction=nonstopmode"`

This produces `pytorch.pdf`. Run this command again for the correct table of contents and index.

> [!NOTE]
> To view the Table of Contents, switch to the **Table of Contents**
> view in your PDF viewer.

### Previous Versions

Find installation instructions and binaries for previous PyTorch versions [here](https://pytorch.org/get-started/previous-versions).

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

*   Forums: https://discuss.pytorch.org
*   GitHub Issues
*   Slack: [PyTorch Slack](https://pytorch.slack.com/) (request invite: https://goo.gl/forms/PP1AGvNHpSaJP8to1)
*   Newsletter: https://eepurl.com/cbG0rv
*   Facebook Page: https://www.facebook.com/pytorch
*   Brand guidelines: [pytorch.org](https://pytorch.org/)

## Releases and Contributing

Typically, PyTorch has three minor releases a year. Report bugs via [filing an issue](https://github.com/pytorch/pytorch/issues).

Contributions are welcome!  Discuss new features by opening an issue first. See [Contribution page](CONTRIBUTING.md) and [Release page](RELEASE.md) for details.

## The Team

PyTorch is a community-driven project.

Maintained by [Soumith Chintala](http://soumith.ch), [Gregory Chanan](https://github.com/gchanan), [Dmytro Dzhulgakov](https://github.com/dzhulgakov), [Edward Yang](https://github.com/ezyang), and [Nikita Shulga](https://github.com/malfet), with significant contributions from many others.

## License

PyTorch is licensed under a BSD-style license, as found in the [LICENSE](LICENSE) file.