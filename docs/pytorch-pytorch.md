[![PyTorch Logo](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/pytorch-logo-dark.png)](https://github.com/pytorch/pytorch)

# PyTorch: The Open Source Deep Learning Platform

**PyTorch empowers researchers and developers with a flexible and efficient platform for building and deploying cutting-edge machine learning models.**  Explore the power of tensors, automatic differentiation, and dynamic neural networks with PyTorch!  Get started at the [original PyTorch repository](https://github.com/pytorch/pytorch).

**Key Features:**

*   **GPU-Accelerated Tensors:**  Like NumPy, but with GPU support for blazing-fast computation.
*   **Dynamic Neural Networks with Autograd:**  Build and modify networks on-the-fly for unparalleled flexibility in research.
*   **Python-First Approach:** Seamless integration with Python, leveraging your favorite libraries.
*   **Imperative Design:** Intuitive and easy-to-debug code execution.
*   **Fast and Lean:** Optimized for speed and efficiency, with minimal framework overhead.
*   **Extensible:**  Easily create custom modules in Python or C/C++.

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

PyTorch provides a comprehensive suite of tools for deep learning, built around these core components:

| Component | Description |
| ---- | --- |
| [**torch**](https://pytorch.org/docs/stable/torch.html) | A Tensor library like NumPy, with strong GPU support |
| [**torch.autograd**](https://pytorch.org/docs/stable/autograd.html) | A tape-based automatic differentiation library that supports all differentiable Tensor operations in torch |
| [**torch.jit**](https://pytorch.org/docs/stable/jit.html) | A compilation stack (TorchScript) to create serializable and optimizable models from PyTorch code  |
| [**torch.nn**](https://pytorch.org/docs/stable/nn.html) | A neural networks library deeply integrated with autograd designed for maximum flexibility |
| [**torch.multiprocessing**](https://pytorch.org/docs/stable/multiprocessing.html) | Python multiprocessing, but with magical memory sharing of torch Tensors across processes. Useful for data loading and Hogwild training |
| [**torch.utils**](https://pytorch.org/docs/stable/data.html) | DataLoader and other utility functions for convenience |

Use PyTorch as:
*   A GPU-accelerated alternative to NumPy.
*   A flexible and fast deep learning research platform.

### A GPU-Ready Tensor Library

PyTorch uses tensors, similar to NumPy's ndarrays.

![Tensor illustration](./docs/source/_static/img/tensor_illustration.png)

PyTorch tensors can reside on the CPU or GPU, speeding up computations significantly.  It provides a wide range of tensor operations including slicing, indexing, math, and linear algebra.

### Dynamic Neural Networks: Tape-Based Autograd

PyTorch uses a "tape recorder" approach to build neural networks.

Most frameworks, such as TensorFlow, Theano, Caffe, and CNTK, utilize static graphs that are fixed at the beginning. PyTorch offers reverse-mode auto-differentiation enabling flexibility to alter the network's behavior arbitrarily. Our inspiration comes from several research papers on this topic.

![Dynamic graph](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/dynamic_graph.gif)

### Python First

PyTorch is deeply integrated into Python, offering a natural experience similar to NumPy, SciPy, and scikit-learn. It seamlessly integrates with popular libraries, including NumPy, SciPy, Cython, and Numba.

### Imperative Experiences

PyTorch's design emphasizes intuitive and linear execution. Code executes immediately, making debugging straightforward. Stack traces point directly to the source of errors.

### Fast and Lean

PyTorch minimizes framework overhead and integrates acceleration libraries (Intel MKL, cuDNN, NCCL). Its CPU and GPU backends have been tested over years.

PyTorch is fast, memory efficient, and allows the training of large models.

### Extensions Without Pain

PyTorch makes it easy to build new neural network modules and interact with its Tensor API. You can write new neural network layers in Python, or use C/C++ via a convenient extension API.

## Installation

Comprehensive installation instructions can be found at [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/).

### Binaries

Install binaries via Conda or pip wheels:  [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

#### NVIDIA Jetson Platforms

Pre-built Python wheels are available for NVIDIA Jetson Nano, TX1/TX2, Xavier NX/AGX, and AGX Orin.  Find the wheels [here](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048) and the L4T container [here](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch).  These require JetPack 4.2 or later.

### From Source

Build PyTorch from source for more customization.

#### Prerequisites

*   Python 3.9 or later
*   C++17-compliant compiler (e.g., GCC 9.4.0 or newer on Linux)
*   Visual Studio or Visual Studio Build Tool (Windows)

\*  PyTorch CI uses Visual C++ BuildTools. You can install the build tools from
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

##### NVIDIA CUDA Support

To compile with CUDA:

*   Install [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads).
*   Install [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) v8.5 or above.
*   Install a [Compiler](https://gist.github.com/ax3l/9489132) compatible with CUDA.

If CUDA is installed in a non-standard location, set the `PATH` variable.  To disable CUDA support, set `USE_CUDA=0`.

##### AMD ROCm Support

To compile with ROCm:

*   Install [AMD ROCm](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html) 4.0 and above.

Set `ROCM_PATH` if ROCm is installed in a non-default location.  Optionally, set `PYTORCH_ROCM_ARCH` for the AMD GPU architecture. To disable ROCm support, set `USE_ROCM=0`.

##### Intel GPU Support

To compile with Intel GPU support, follow the instructions at:
*   [PyTorch Prerequisites for Intel GPUs](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpus.html)
*   Intel GPU support is available for Linux and Windows.
To disable Intel GPU support, set `USE_XPU=0`.

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

If compiling for AMD ROCm, run:

```bash
python tools/amd_build/build_amd.py
```

Install:

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

Refer to [Building on legacy code and CUDA](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md#building-on-legacy-code-and-cuda) for building with legacy python code.

**CPU-only builds**

```cmd
python -m pip install --no-build-isolation -v -e .
```

Note on OpenMP: The desired OpenMP implementation is Intel OpenMP (iomp). In order to link against iomp, you'll need to manually download the library and set up the building environment by tweaking `CMAKE_INCLUDE_PATH` and `LIB`. The instruction [here](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/windows.rst#building-from-source) is an example for setting up both MKL and Intel OpenMP. Without these configurations for CMake, Microsoft Visual C OpenMP runtime (vcomp) will be used.

**CUDA based build**

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

Make sure [the common prerequisites](#prerequisites) and [the prerequisites for Intel GPU](#intel-gpu-support) are installed. `Visual Studio 2022` is required.

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

Adjust CMake variables:

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

Increase shared memory if using multiprocessing with `torch.multiprocessing`:  `--ipc=host` or `--shm-size`

#### Building the image yourself

Build Docker images with CUDA 11.1 support and cuDNN v8.

```bash
make -f docker.Makefile
```

### Building the Documentation

Install [Sphinx](http://www.sphinx-doc.org) and pytorch\_sphinx\_theme2.

Ensure `torch` is installed in your environment.

```bash
cd docs/
pip install -r requirements.txt
make html
make serve
```

Run `make` to see available output formats.

If katex error run `npm install katex`.  If it persists, try
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

```
brew install --cask mactex
```

1.  Run:

    ```
    make latexpdf
    ```

2.  Navigate to `build/latex` and execute:

    ```
    make LATEXOPTS="-interaction=nonstopmode"
    ```

    This produces a `pytorch.pdf`.
    Run this command one more time so that it generates the correct table
    of contents and index.

> [!NOTE]
> To view the Table of Contents, switch to the **Table of Contents**
> view in your PDF viewer.

### Previous Versions

Find installation instructions and binaries for previous PyTorch versions on [our website](https://pytorch.org/get-started/previous-versions).

## Getting Started

Get started with these resources:
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

*   Forums: https://discuss.pytorch.org
*   GitHub Issues
*   Slack: [PyTorch Slack](https://pytorch.slack.com/).  Request an invite:  https://goo.gl/forms/PP1AGvNHpSaJP8to1
*   Newsletter: Sign-up: https://eepurl.com/cbG0rv
*   Facebook Page: https://www.facebook.com/pytorch
*   Brand guidelines:  [pytorch.org](https://pytorch.org/)

## Releases and Contributing

PyTorch typically releases three minor versions per year.  Report bugs via [filing an issue](https://github.com/pytorch/pytorch/issues).

We welcome contributions!  Discuss new features or extensions with us by opening an issue.  See [Contribution page](CONTRIBUTING.md) for more information.  For more about PyTorch releases, see [Release page](RELEASE.md).

## The Team

PyTorch is a community-driven project.  Maintainers include Soumith Chintala, Gregory Chanan, Dmytro Dzhulgakov, Edward Yang, and Nikita Shulga, with major contributions from many individuals.

## License

PyTorch is licensed under a BSD-style license, as found in the [LICENSE](LICENSE) file.