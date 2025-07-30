<!-- PyTorch Logo -->
![PyTorch Logo](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/pytorch-logo-dark.png)

# PyTorch: The Open Source Deep Learning Platform

**PyTorch empowers researchers and developers with a flexible and intuitive platform for building and deploying cutting-edge machine learning models.**  Learn more and contribute at the [original PyTorch repository](https://github.com/pytorch/pytorch).

## Key Features

*   **Tensor Computation with GPU Acceleration:** Enjoy fast tensor operations similar to NumPy, with seamless GPU support for accelerated computation.
*   **Dynamic Neural Networks with Autograd:** Build and modify neural networks with unparalleled flexibility using tape-based automatic differentiation.
*   **Python-First Approach:** Integrate effortlessly with existing Python libraries and workflows, leveraging the power of NumPy, SciPy, and Cython.
*   **Imperative Programming:** Benefit from an intuitive, imperative style for easy debugging and a clear understanding of your code's execution.
*   **Fast and Lean:** Utilize optimized acceleration libraries like Intel MKL, cuDNN, and NCCL for top performance and memory efficiency.
*   **Easy Extensions:** Create custom neural network modules and seamlessly interface with PyTorch's Tensor API with minimal effort.

## Installation

Install PyTorch using binaries or build from source.

### Binaries

Easy-to-use Conda or pip packages are available: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

#### NVIDIA Jetson Platforms

Pre-built wheels and L4T containers are provided for NVIDIA Jetson platforms: [NVIDIA Jetson PyTorch](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048).

### From Source

#### Prerequisites

*   Python 3.9 or later
*   C++17 compatible compiler (gcc 9.4.0 or newer on Linux)
*   Visual Studio or Visual Studio Build Tool (Windows)

#### NVIDIA CUDA Support

1.  Install supported CUDA and cuDNN versions as detailed in the [PyTorch CUDA Support Matrix](https://pytorch.org/get-started/locally/).
2.  CUDA, cuDNN and NVIDIA Driver version must be compatible.
3.  Set the PATH to locate the nvcc compiler.

#### AMD ROCm Support

1.  Install AMD ROCm 4.0 and above.
2.  Set the `ROCM_PATH` environment variable if ROCm is installed in a non-default directory.
3.  Optionally, use the `PYTORCH_ROCM_ARCH` environment variable.
4.  Disable ROCm support using `USE_ROCM=0`.

#### Intel GPU Support

1.  Follow the [PyTorch Prerequisites for Intel GPUs](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpus.html).

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

Install [Sphinx](http://www.sphinx-doc.org) and the pytorch_sphinx_theme2 to build the docs.

```bash
cd docs/
pip install -r requirements.txt
make html
make serve
```

#### Building a PDF

Install `texlive` and LaTeX.

```bash
make latexpdf
make LATEXOPTS="-interaction=nonstopmode"
```

### Previous Versions

Find installation instructions and binaries at [PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions).

## Getting Started

Explore tutorials, examples, the API Reference, and the Glossary to begin using PyTorch.
* [Tutorials](https://pytorch.org/tutorials/)
* [Examples](https://github.com/pytorch/examples)
* [API Reference](https://pytorch.org/docs/)
* [Glossary](https://github.com/pytorch/pytorch/blob/main/GLOSSARY.md)

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

*   **Forums:** [Discuss PyTorch](https://discuss.pytorch.org)
*   **GitHub Issues:** Bug reports, feature requests, etc.
*   **Slack:** [PyTorch Slack](https://pytorch.slack.com/) (request an invite)
*   **Newsletter:** Sign-up for important announcements: [PyTorch Newsletter](https://eepurl.com/cbG0rv)
*   **Facebook Page:** [PyTorch Facebook](https://www.facebook.com/pytorch)
*   **Brand Guidelines:** [PyTorch Brand](https://pytorch.org/)

## Releases and Contributing

PyTorch releases three minor versions per year; report bugs via [GitHub Issues](https://github.com/pytorch/pytorch/issues). Contribute bug fixes and new features by following the [Contribution Guidelines](CONTRIBUTING.md).  See the [Release Page](RELEASE.md) for release information.

## The Team

PyTorch is a community-driven project with many contributors.  Currently maintained by Soumith Chintala, Gregory Chanan, Dmytro Dzhulgakov, Edward Yang, and Nikita Shulga, with major contributions from hundreds of individuals.

## License

PyTorch is released under a BSD-style license, available in the [LICENSE](LICENSE) file.