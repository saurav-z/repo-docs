<div align="center">
  <a href="https://github.com/pytorch/pytorch">
    <img src="https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/pytorch-logo-dark.png" alt="PyTorch Logo" width="200">
  </a>
  <h1>PyTorch: Deep Learning with GPU Acceleration</h1>
</div>

PyTorch is a powerful and flexible open-source deep learning framework that provides seamless GPU acceleration and a dynamic computational graph, making it ideal for research and production. **[Explore the PyTorch GitHub Repository](https://github.com/pytorch/pytorch) to get started!**

**Key Features:**

*   🚀 **GPU-Accelerated Tensors:**  Leverage the power of GPUs for blazing-fast tensor computations, similar to NumPy but optimized for acceleration.
*   🧠 **Dynamic Neural Networks:** Build and modify neural networks with unparalleled flexibility using a tape-based autograd system.
*   🐍 **Pythonic Design:** Seamlessly integrate PyTorch into your existing Python workflows with intuitive APIs and extensive library compatibility (NumPy, SciPy, etc.).
*   ⚙️ **Imperative Style:** Benefit from an imperative programming style, simplifying debugging and providing clear error messages with direct stack traces.
*   ⚡️ **Fast and Lean:** Experience minimal overhead, optimized performance with libraries like Intel MKL and NVIDIA cuDNN, and efficient memory usage.
*   ➕ **Easy Extensions:** Easily customize and extend PyTorch with custom layers, functions, and integration with C/C++ for enhanced performance.

## Table of Contents

1.  [More About PyTorch](#more-about-pytorch)
    1.  [A GPU-Ready Tensor Library](#a-gpu-ready-tensor-library)
    2.  [Dynamic Neural Networks: Tape-Based Autograd](#dynamic-neural-networks-tape-based-autograd)
    3.  [Python First](#python-first)
    4.  [Imperative Experiences](#imperative-experiences)
    5.  [Fast and Lean](#fast-and-lean)
    6.  [Extensions Without Pain](#extensions-without-pain)
2.  [Installation](#installation)
    1.  [Binaries](#binaries)
        1.  [NVIDIA Jetson Platforms](#nvidia-jetson-platforms)
    2.  [From Source](#from-source)
        1.  [Prerequisites](#prerequisites)
            1.  [NVIDIA CUDA Support](#nvidia-cuda-support)
            2.  [AMD ROCm Support](#amd-rocm-support)
            3.  [Intel GPU Support](#intel-gpu-support)
        2.  [Get the PyTorch Source](#get-the-pytorch-source)
        3.  [Install Dependencies](#install-dependencies)
        4.  [Install PyTorch](#install-pytorch)
            1.  [Adjust Build Options (Optional)](#adjust-build-options-optional)
    3.  [Docker Image](#docker-image)
        1.  [Using pre-built images](#using-pre-built-images)
        2.  [Building the image yourself](#building-the-image-yourself)
    4.  [Building the Documentation](#building-the-documentation)
        1.  [Building a PDF](#building-a-pdf)
    5.  [Previous Versions](#previous-versions)
3.  [Getting Started](#getting-started)
4.  [Resources](#resources)
5.  [Communication](#communication)
6.  [Releases and Contributing](#releases-and-contributing)
7.  [The Team](#the-team)
8.  [License](#license)

## More About PyTorch

[Learn the basics of PyTorch](https://pytorch.org/tutorials/beginner/basics/intro.html)

PyTorch is comprised of several core components:

| Component                  | Description                                                                                                               |
| :-------------------------- | :------------------------------------------------------------------------------------------------------------------------ |
| [**torch**](https://pytorch.org/docs/stable/torch.html) | Tensor library similar to NumPy, with powerful GPU support.                                                     |
| [**torch.autograd**](https://pytorch.org/docs/stable/autograd.html) | Tape-based automatic differentiation library that supports all differentiable Tensor operations in torch.   |
| [**torch.jit**](https://pytorch.org/docs/stable/jit.html)     | Compilation stack (TorchScript) for serializable and optimizable models from PyTorch code.                     |
| [**torch.nn**](https://pytorch.org/docs/stable/nn.html)      | Neural networks library deeply integrated with autograd, designed for maximum flexibility.                    |
| [**torch.multiprocessing**](https://pytorch.org/docs/stable/multiprocessing.html) | Python multiprocessing with magical memory sharing of torch Tensors across processes. Useful for data loading and Hogwild training. |
| [**torch.utils**](https://pytorch.org/docs/stable/data.html)  | DataLoader and other utility functions for convenience.                                                        |

PyTorch is typically used for:

*   Replacing NumPy to harness the power of GPUs.
*   A deep learning research platform offering flexibility and speed.

### A GPU-Ready Tensor Library

PyTorch uses Tensors (similar to NumPy's ndarray).

![Tensor illustration](./docs/source/_static/img/tensor_illustration.png)

These Tensors can live on the CPU or GPU, significantly accelerating computations.

PyTorch provides a range of tensor operations for scientific computing, including slicing, indexing, mathematical operations, linear algebra, and reductions, all optimized for speed.

### Dynamic Neural Networks: Tape-Based Autograd

PyTorch uses a "tape recorder" approach for building neural networks.

Most frameworks like TensorFlow use static graphs, requiring rebuilding the network for any structural changes.  PyTorch uses reverse-mode automatic differentiation, allowing for dynamic network behavior changes with zero overhead.

![Dynamic graph](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/dynamic_graph.gif)

This approach is known to be one of the fastest implementations for speed and flexibility in deep learning research.

### Python First

PyTorch is deeply integrated with Python, like NumPy, SciPy, and scikit-learn. You can use Python features, your favorite libraries, and packages such as [Cython](https://cython.org/) and [Numba](http://numba.pydata.org/) to extend PyTorch.

### Imperative Experiences

PyTorch is designed to be intuitive. When you execute a line of code, it's executed immediately without asynchronous behavior. Understanding debugging and error messages is straightforward. Stack traces point directly to the defined code.

### Fast and Lean

PyTorch minimizes framework overhead, using acceleration libraries like Intel MKL and NVIDIA cuDNN/NCCL. The CPU and GPU Tensor and neural network backends are mature and tested for years.  This results in high-speed performance.

PyTorch offers efficient memory usage, especially on the GPU, allowing training larger models. Custom memory allocators optimize deep learning models for maximum memory efficiency.

### Extensions Without Pain

Extending PyTorch with new neural network modules is easy and provides minimal abstraction.  You can write new layers in Python using the torch API or using NumPy-based libraries.  For C/C++ layers, a convenient and efficient extension API with minimal boilerplate is available.

## Installation

### Binaries

Install binaries via Conda or pip wheels on our website: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

#### NVIDIA Jetson Platforms

Python wheels for NVIDIA's Jetson Nano, Jetson TX1/TX2, Jetson Xavier NX/AGX, and Jetson AGX Orin are provided [here](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048) and the L4T container is published [here](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch)

They require JetPack 4.2 and above, and [@dusty-nv](https://github.com/dusty-nv) and [@ptrblck](https://github.com/ptrblck) are maintaining them.

### From Source

#### Prerequisites

If installing from source, you'll need:

*   Python 3.9 or later
*   A C++17 compatible compiler (e.g., GCC 9.4.0 or newer on Linux, clang)
*   Visual Studio or Visual Studio Build Tool (Windows)

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

A conda environment is not required. You can also build in a
standard virtual environment.

##### NVIDIA CUDA Support

To compile with CUDA support, [select a supported version of CUDA from our support matrix](https://pytorch.org/get-started/locally/), and install:

*   [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads)
*   [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) v8.5 or above
*   [Compiler](https://gist.github.com/ax3l/9489132) compatible with CUDA

Note: You could refer to the [cuDNN Support Matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/support-matrix.html) for cuDNN versions with the various supported CUDA, CUDA driver and NVIDIA hardware

To disable CUDA, use the environment variable `USE_CUDA=0`.  Other environment variables are in `setup.py`. Set PATH if CUDA is in a non-standard location.

Instructions for NVIDIA's Jetson platforms: [here](https://devtalk.nvidia.com/default/topic/1049071/jetson-nano/pytorch-for-jetson-nano/)

##### AMD ROCm Support

To compile with ROCm support, install:

*   [AMD ROCm](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html) 4.0 and above.
*   ROCm is currently supported only for Linux systems.

By default, ROCm is expected in `/opt/rocm`. Use the `ROCM_PATH` environment variable if it's installed elsewhere. Optionally, set `PYTORCH_ROCM_ARCH` to specify the AMD GPU architecture [AMD GPU architecture](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-gpus)

To disable ROCm, use the environment variable `USE_ROCM=0`.  Other environment variables are in `setup.py`.

##### Intel GPU Support

To compile with Intel GPU support, follow the [PyTorch Prerequisites for Intel GPUs](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpus.html) instructions.
* Intel GPU is supported for Linux and Windows.

To disable Intel GPU, use the environment variable `USE_XPU=0`. Other environment variables are in `setup.py`.

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

```cmd
python setup.py develop
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

python setup.py develop

```

**Intel GPU builds**

Make sure [the common prerequisites](#prerequisites) and [the prerequisites for Intel GPU](#intel-gpu-support) are installed, and environment variables are configured.  `Visual Studio 2022` is required.

Build with:

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

You can adjust CMake variables (without building first) by running:

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

Run a pre-built image from Docker Hub:

```bash
docker run --gpus all --rm -ti --ipc=host pytorch/pytorch:latest
```

Use `--ipc=host` or `--shm-size` for shared memory with torch multiprocessing.

#### Building the image yourself

**NOTE:** Must be built with a docker version > 18.06

The `Dockerfile` builds images with CUDA 11.1 support and cuDNN v8.
Specify the Python version with `PYTHON_VERSION=x.y`.

```bash
make -f docker.Makefile
# images are tagged as docker.io/${your_docker_username}/pytorch
```

Use `CMAKE_VARS="..."` to specify additional CMake variables.

```bash
make -f docker.Makefile
```

### Building the Documentation

You'll need [Sphinx](http://www.sphinx-doc.org) and the pytorch_sphinx_theme2.

Ensure `torch` is installed in your environment. Use the nightly version as described in [Getting Started](https://pytorch.org/get-started/locally/) or install [from source](#from-source). See [Docstring Guidelines](https://github.com/pytorch/pytorch/wiki/Docstring-Guidelines) for docstring conventions.

```bash
cd docs/
pip install -r requirements.txt
make html
make serve
```

Run `make` to see available output formats.

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

When you change dependencies, edit `.ci/docker/requirements-docs.txt`.

#### Building a PDF

Install `texlive` and LaTeX:

```
brew install --cask mactex
```

To create the PDF:

1.  Run:

    ```
    make latexpdf
    ```

    This creates files in `build/latex`.

2.  Go to the directory and run:

    ```
    make LATEXOPTS="-interaction=nonstopmode"
    ```

    This generates the `pytorch.pdf`.

3. Run this command one more time so that it generates the correct table
    of contents and index.

> [!NOTE]
> To view the Table of Contents, switch to the **Table of Contents**
> view in your PDF viewer.

### Previous Versions

Find installation instructions and binaries for previous PyTorch versions on [our website](https://pytorch.org/get-started/previous-versions).

## Getting Started

Key resources:

*   [Tutorials](https://pytorch.org/tutorials/)
*   [Examples](https://github.com/pytorch/examples)
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

*   Forums: https://discuss.pytorch.org
*   GitHub Issues: Bug reports, feature requests, etc.
*   Slack: The [PyTorch Slack](https://pytorch.slack.com/) hosts a primary audience of moderate to experienced PyTorch users and developers for general chat, online discussions, collaboration, etc. If you are a beginner looking for help, the primary medium is [PyTorch Forums](https://discuss.pytorch.org). If you need a slack invite, please fill this form: https://goo.gl/forms/PP1AGvNHpSaJP8to1
*   Newsletter: https://eepurl.com/cbG0rv
*   Facebook Page: https://www.facebook.com/pytorch
*   For brand guidelines, please visit our website at [pytorch.org](https://pytorch.org/)

## Releases and Contributing

PyTorch typically releases three minor versions annually.  Report bugs by [filing an issue](https://github.com/pytorch/pytorch/issues).

We welcome contributions.  For bug fixes, submit PRs. For new features, open an issue to discuss them.  See the [Contribution page](CONTRIBUTING.md) and [Release page](RELEASE.md).

## The Team

PyTorch is community-driven. Maintained by [Soumith Chintala](http://soumith.ch), [Gregory Chanan](https://github.com/gchanan), [Dmytro Dzhulgakov](https://github.com/dzhulgakov), [Edward Yang](https://github.com/ezyang), and [Nikita Shulga](https://github.com/malfet), with contributions from many talented individuals.  A non-exhaustive list of contributors includes [Trevor Killeen](https://github.com/killeent), [Sasank Chilamkurthy](https://github.com/chsasank), [Sergey Zagoruyko](https://github.com/szagoruyko), [Adam Lerer](https://github.com/adamlerer), [Francisco Massa](https://github.com/fmassa), [Alykhan Tejani](https://github.com/alykhantejani), [Luca Antiga](https://github.com/lantiga), [Alban Desmaison](https://github.com/albanD), [Andreas Koepf](https://github.com/andreaskoepf), [James Bradbury](https://github.com/jekbradbury), [Zeming Lin](https://github.com/ebetica), [Yuandong Tian](https://github.com/yuandong-tian), [Guillaume Lample](https://github.com/glample), [Marat Dukhan](https://github.com/Maratyszcza), [Natalia Gimelshein](https://github.com/ngimel), [Christian Sarofeen](https://github.com/csarofeen), [Martin Raison](https://github.com/martinraison), [Edward Yang](https://github.com/ezyang), [Zachary Devito](https://github.com/zdevito).

Note: This project is unrelated to [hughperkins/pytorch](https://github.com/hughperkins/pytorch) with the same name. Hugh is a valuable contributor to the Torch community and has helped with many things Torch and PyTorch.

## License

PyTorch is licensed under a BSD-style license, as found in the [LICENSE](LICENSE) file.