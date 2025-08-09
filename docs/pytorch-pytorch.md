<div align="center">
  <a href="https://github.com/pytorch/pytorch">
    <img src="https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/pytorch-logo-dark.png" alt="PyTorch Logo" width="200">
  </a>
  <h1>PyTorch: Deep Learning with Python</h1>
  <p><em>Power your AI projects with PyTorch, a flexible and efficient open-source deep learning framework.</em></p>
</div>

---

**PyTorch** is a leading open-source machine learning framework, offering powerful tools for tensor computation, deep neural network development, and GPU acceleration.  Built for flexibility and ease of use, it allows researchers and developers to bring their AI ideas to life. Access the original repo [here](https://github.com/pytorch/pytorch).

**Key Features:**

*   **Tensor Computation with GPU Acceleration:**  Leverage the power of GPUs for fast and efficient numerical computation, similar to NumPy but with enhanced performance.
*   **Dynamic Neural Networks:** Build flexible neural networks using a tape-based autograd system, enabling dynamic behavior and rapid prototyping.
*   **Python-First Approach:** Integrate seamlessly with the Python ecosystem, utilizing familiar libraries like NumPy, SciPy, and Cython.
*   **Imperative and Intuitive:** Experience a straightforward and easy-to-debug development process with an imperative style.
*   **Fast and Lean:** Benefit from minimal framework overhead and optimized performance through integration with acceleration libraries.
*   **Extensible Architecture:** Easily create custom neural network modules and interface with PyTorch's Tensor API using Python or C/C++.

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

---

## More About PyTorch

[Learn the basics of PyTorch](https://pytorch.org/tutorials/beginner/basics/intro.html)

PyTorch is composed of the following key components:

| Component                        | Description                                                                                                                                     |
| :------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------ |
| [**torch**](https://pytorch.org/docs/stable/torch.html)             | Tensor library with GPU support, similar to NumPy.                                                                            |
| [**torch.autograd**](https://pytorch.org/docs/stable/autograd.html) | Automatic differentiation library based on a tape, supporting all differentiable Tensor operations.                                     |
| [**torch.jit**](https://pytorch.org/docs/stable/jit.html)        | Compilation stack (TorchScript) to create serializable and optimizable models from PyTorch code.                                        |
| [**torch.nn**](https://pytorch.org/docs/stable/nn.html)         | Neural networks library deeply integrated with autograd, designed for maximum flexibility.                                                        |
| [**torch.multiprocessing**](https://pytorch.org/docs/stable/multiprocessing.html)  | Python multiprocessing with magical memory sharing of torch Tensors across processes, beneficial for data loading and Hogwild training.     |
| [**torch.utils**](https://pytorch.org/docs/stable/data.html)    | DataLoader and other utility functions.                                                                                                  |

PyTorch is typically used for:

*   Replacing NumPy to utilize the power of GPUs.
*   Serving as a deep learning research platform that provides maximum flexibility and speed.

### A GPU-Ready Tensor Library

PyTorch's `torch` module provides tensors (similar to NumPy's `ndarray`), allowing you to perform complex calculations efficiently, with the option of running them on either the CPU or the GPU. This results in significant acceleration.

![Tensor illustration](./docs/source/_static/img/tensor_illustration.png)

PyTorch provides a comprehensive set of tensor operations to accelerate your scientific computation needs, including slicing, indexing, mathematical operations, linear algebra, and reductions. These operations are optimized for speed.

### Dynamic Neural Networks: Tape-Based Autograd

PyTorch employs a unique approach to constructing neural networks, using a tape recorder-like system.  This enables dynamic computation graphs, allowing you to change the network's behavior during runtime with minimal overhead.

Unlike frameworks with static graph designs, PyTorch's reverse-mode auto-differentiation allows for dynamic behavior, adapting to research needs without requiring restructuring from scratch.

![Dynamic graph](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/dynamic_graph.gif)

### Python First

PyTorch is seamlessly integrated into Python. It does not simply offer Python bindings to a C++ framework. You can use PyTorch as naturally as you would use NumPy, SciPy, or scikit-learn. Write custom neural network layers directly in Python, integrating your favorite libraries, and utilizing tools like Cython and Numba.

### Imperative Experiences

PyTorch emphasizes an intuitive, linear, and user-friendly experience. Code executes line by line, facilitating debugging and error analysis. Stack traces point directly to the source code, simplifying debugging and ensuring transparency in execution.

### Fast and Lean

PyTorch minimizes framework overhead and integrates acceleration libraries like Intel MKL and NVIDIA (cuDNN, NCCL) to maximize speed. The core CPU and GPU tensor and neural network backends are mature and have been tested extensively.

PyTorch's memory usage is highly efficient, with custom GPU memory allocators to maximize memory efficiency in deep learning models. This allows the training of larger models.

### Extensions Without Pain

Building custom neural network modules or interfacing with PyTorch's Tensor API is designed to be straightforward.

Write new neural network layers in Python using the torch API, or integrate your favorite NumPy-based libraries, such as SciPy.  If you need to write layers in C/C++, an extension API is provided that is efficient and easy to use, eliminating boilerplate.

## Installation

Find installation instructions and binaries on our website: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

### Binaries

Installation via Conda or pip wheels.

#### NVIDIA Jetson Platforms

Pre-built Python wheels for NVIDIA's Jetson Nano, Jetson TX1/TX2, Jetson Xavier NX/AGX, and Jetson AGX Orin are available [here](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048), and the L4T container is published [here](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch).

These require JetPack 4.2 and above and are maintained by [@dusty-nv](https://github.com/dusty-nv) and [@ptrblck](https://github.com/ptrblck).

### From Source

#### Prerequisites

*   Python 3.9 or later
*   A C++17 compliant compiler (e.g., clang or gcc, with gcc 9.4.0 or newer on Linux)
*   Visual Studio or Visual Studio Build Tool (Windows only)

\*  PyTorch CI uses Visual C++ BuildTools (available with Visual Studio Community Edition). You can also install from https://visualstudio.microsoft.com/visual-cpp-build-tools/. Build tools *do not* come with Visual Studio Code by default.

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

A conda environment is not required. You can also build in a standard virtual environment, using tools like `uv`, assuming all necessary dependencies are installed (e.g., CUDA, MKL.)

##### NVIDIA CUDA Support

If building with CUDA, install the following:

*   [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads) (select a supported version from our support matrix: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/))
*   [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) v8.5 or above
*   [Compiler](https://gist.github.com/ax3l/9489132) compatible with CUDA

Note: Check the [cuDNN Support Matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/support-matrix.html) for cuDNN and CUDA version compatibility.

To disable CUDA support, set `USE_CUDA=0`.  Other useful environment variables can be found in `setup.py`. If CUDA is in a non-standard location, set the `PATH` accordingly (e.g., `export PATH=/usr/local/cuda-12.8/bin:$PATH`).

For Jetson platforms, instructions are available [here](https://devtalk.nvidia.com/default/topic/1049071/jetson-nano/pytorch-for-jetson-nano/).

##### AMD ROCm Support

If compiling with ROCm, install:

*   [AMD ROCm](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html) 4.0 or higher.

ROCm is currently supported only on Linux.  The build system looks for ROCm in `/opt/rocm`. If installed elsewhere, set the `ROCM_PATH` environment variable. The `PYTORCH_ROCM_ARCH` variable can be used to explicitly set the AMD GPU architecture.

Disable ROCm with `USE_ROCM=0`. Other helpful variables are in `setup.py`.

##### Intel GPU Support

If compiling with Intel GPU support, refer to:

*   [PyTorch Prerequisites for Intel GPUs](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpus.html).
*   Intel GPU support is available on Linux and Windows.

Disable Intel GPU with `USE_XPU=0`. Environment variables can be found in `setup.py`.

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

If compiling for AMD ROCm, first run:

```bash
# Only run this if you're compiling for ROCm
python tools/amd_build/build_amd.py
```

Install PyTorch:

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

```cmd
python -m pip install --no-build-isolation -v -e .
```

Note on OpenMP:  Intel OpenMP (iomp) is the preferred implementation. To link against iomp, you must manually download the library and set up the building environment by tweaking `CMAKE_INCLUDE_PATH` and `LIB`. The [Windows building instructions](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/windows.rst#building-from-source) provide an example for setting up both MKL and Intel OpenMP. Without these configurations for CMake, Microsoft Visual C OpenMP runtime (vcomp) will be used.

**CUDA based build**

CUDA-based PyTorch leverages your GPU via CUDA for faster computation.

[NVTX](https://docs.nvidia.com/gameworks/content/gameworkslibrary/nvtx/nvidia_tools_extension_library_nvtx.htm) is needed to build Pytorch with CUDA.
NVTX is a part of CUDA distributive, where it is called "Nsight Compute". To install it onto an already installed CUDA run CUDA installation once again and check the corresponding checkbox.
Make sure that CUDA with Nsight Compute is installed after Visual Studio.

VS 2017 / 2019, and Ninja are supported as the generator of CMake. If `ninja.exe` is detected in `PATH`, then Ninja will be used as the default generator, otherwise, it will use VS 2017 / 2019.
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

To build PyTorch with Intel GPU support.

Ensure you have met the [prerequisites](#prerequisites) and [Intel GPU prerequisites](#intel-gpu-support), and the environment variables are correctly configured. `Visual Studio 2022` is required.

Build PyTorch:

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

Adjust CMake variables using the following steps:

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

Use Docker Hub pre-built images with docker v19.03+:

```bash
docker run --gpus all --rm -ti --ipc=host pytorch/pytorch:latest
```

Shared memory is required for PyTorch multiprocessing. Increase the shared memory size using `--ipc=host` or `--shm-size` when running with `nvidia-docker run`.

#### Building the image yourself

**NOTE:** Requires Docker > 18.06

The `Dockerfile` creates images with CUDA 11.1 and cuDNN v8 support.  Specify `PYTHON_VERSION=x.y` to set the Python version.

```bash
make -f docker.Makefile
# images are tagged as docker.io/${your_docker_username}/pytorch
```

The `CMAKE_VARS="..."` environment variable allows you to pass additional CMake variables.

```bash
make -f docker.Makefile
```

### Building the Documentation

Requires [Sphinx](http://www.sphinx-doc.org) and the pytorch_sphinx_theme2.

Ensure `torch` is installed in your environment before building the documentation locally. For advanced fixes, install from source [as described above](#from-source). See [Docstring Guidelines](https://github.com/pytorch/pytorch/wiki/Docstring-Guidelines) for proper docstring conventions.

```bash
cd docs/
pip install -r requirements.txt
make html
make serve
```

Run `make` for a list of output formats.

If you get a katex error, run `npm install katex` (or `npm install -g katex`).

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

When you change CI dependencies, edit `.ci/docker/requirements-docs.txt`.

#### Building a PDF

Requires `texlive` and LaTeX. On macOS:

```
brew install --cask mactex
```

To create the PDF:

1.  Run:

    ```
    make latexpdf
    ```

    This generates files in `build/latex`.
2.  Navigate to the `build/latex` directory and execute:

    ```
    make LATEXOPTS="-interaction=nonstopmode"
    ```

    This generates `pytorch.pdf`. Run it once more to generate the correct table of contents and index.

> [!NOTE]
> To view the Table of Contents, switch to the **Table of Contents**
> view in your PDF viewer.

### Previous Versions

Installation instructions and binaries for previous PyTorch versions are available [on our website](https://pytorch.org/get-started/previous-versions).

## Getting Started

Get started using PyTorch with the following resources:

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

*   **Forums:** Discuss implementations, research, and more.  [https://discuss.pytorch.org](https://discuss.pytorch.org)
*   **GitHub Issues:** Report bugs, request features, report install issues, suggest RFCs, etc.
*   **Slack:**  Join the [PyTorch Slack](https://pytorch.slack.com/) for discussions.  If you need a slack invite, please fill this form: https://goo.gl/forms/PP1AGvNHpSaJP8to1.  Beginners are encouraged to use the forums for help.
*   **Newsletter:** Sign up for the PyTorch newsletter for announcements. [https://eepurl.com/cbG0rv](https://eepurl.com/cbG0rv)
*   **Facebook Page:** Stay updated on announcements.  [https://www.facebook.com/pytorch](https://www.facebook.com/pytorch)
*   For brand guidelines, visit [pytorch.org](https://pytorch.org/)

## Releases and Contributing

PyTorch typically releases three minor versions per year.  Report bugs by [filing an issue](https://github.com/pytorch/pytorch/issues).

We welcome contributions!  For bug fixes, feel free to submit a PR.  For new features, utility functions, or extensions, please open an issue to discuss them before submitting a PR.

Learn more about contributing to PyTorch on our [Contribution page](CONTRIBUTING.md) and about releases on our [Release page](RELEASE.md).

## The Team

PyTorch is a community-driven project. It is maintained by [Soumith Chintala](http://soumith.ch), [Gregory Chanan](https://github.com/gchanan), [Dmytro Dzhulgakov](https://github.com/dzhulgakov), [Edward Yang](https://github.com/ezyang), and [Nikita Shulga](https://github.com/malfet), with major contributions from hundreds of talented individuals. See the list of contributors in the original README for more names.

Note: This project is unrelated to [hughperkins/pytorch](https://github.com/hughperkins/pytorch).

## License

PyTorch is licensed under a BSD-style license, as found in the [LICENSE](LICENSE) file.