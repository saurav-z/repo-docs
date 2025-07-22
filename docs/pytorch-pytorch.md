![PyTorch Logo](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/pytorch-logo-dark.png)

# PyTorch: The Deep Learning Framework for Research and Production

**PyTorch empowers you to build and train cutting-edge deep learning models with flexibility and speed.** This repository contains the core source code for PyTorch, a popular open-source machine learning framework. [Visit the original PyTorch repository](https://github.com/pytorch/pytorch) to learn more.

## Key Features of PyTorch

*   **GPU-Accelerated Tensor Computation:** Perform tensor operations on CPUs and GPUs, offering significant performance boosts for your scientific computing needs.
*   **Dynamic Neural Networks with Autograd:** Build and modify neural networks with unparalleled flexibility using PyTorch's tape-based automatic differentiation system.
*   **Python-First Development:** Seamlessly integrate PyTorch with your existing Python workflow, leveraging libraries like NumPy, SciPy, and Cython.
*   **Imperative Programming Style:** Experience intuitive debugging and straightforward code execution for easier development and understanding.
*   **Fast and Lean:** Benefit from minimal framework overhead and optimized integrations with acceleration libraries like Intel MKL, cuDNN, and NCCL.
*   **Effortless Extensions:** Extend PyTorch with custom neural network modules and easily interface with its tensor API in Python or C/C++.

## Installation

Get started quickly by following these steps for installing PyTorch on your system.

### Binaries

Install pre-built binaries using either Conda or pip wheels, as outlined on the [PyTorch website](https://pytorch.org/get-started/locally/).

#### NVIDIA Jetson Platforms

Pre-built Python wheels for NVIDIA Jetson platforms (Nano, TX1/TX2, Xavier NX/AGX, and AGX Orin) are available. For more information, see the [NVIDIA forums](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048) and the [L4T container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch).

### From Source

Build PyTorch from source for customization or to address specific needs.

#### Prerequisites

*   Python 3.9 or later
*   A C++17-compliant compiler (e.g., GCC 9.4.0+)
*   Visual Studio or Visual Studio Build Tool (Windows)

#### NVIDIA CUDA Support

*   Install the supported version of CUDA from the [PyTorch support matrix](https://pytorch.org/get-started/locally/).
*   Install NVIDIA CUDA and cuDNN v8.5 or above.
*   Make sure to have a compiler that is compatible with CUDA.

To disable CUDA support, set the `USE_CUDA=0` environment variable.

#### AMD ROCm Support

*   Install AMD ROCm 4.0 and above.

Set the `ROCM_PATH` environment variable to the ROCm installation directory if not in the default location. Use `PYTORCH_ROCM_ARCH` to specify the AMD GPU architecture.  To disable ROCm support, set `USE_ROCM=0`.

#### Intel GPU Support

*   Follow the [PyTorch Prerequisites for Intel GPUs](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpus.html) instructions.

To disable Intel GPU support, set `USE_XPU=0`.

#### Get the PyTorch Source

```bash
git clone https://github.com/pytorch/pytorch
cd pytorch
git submodule sync
git submodule update --init --recursive
```

#### Install Dependencies

```bash
conda install cmake ninja  # or use pip install if using a venv
pip install -r requirements.txt
```

**Optional Linux Dependencies:**

```bash
pip install mkl-static mkl-include
# CUDA only:
# magma installation (run from an active conda environment)
.ci/docker/common/install_magma_conda.sh 12.4
# (optional) Triton installation
make triton
```

**Optional macOS Dependencies:**

```bash
pip install mkl-static mkl-include
conda install pkg-config libuv
```

**Optional Windows Dependencies:**

```bash
pip install mkl-static mkl-include
conda install -c conda-forge libuv=1.39
```

#### Install PyTorch

**Linux:**

```bash
# Only run this if you're compiling for ROCm
python tools/amd_build/build_amd.py
export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"
python -m pip install --no-build-isolation -v -e .
```

**macOS:**

```bash
export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"
python -m pip install --no-build-isolation -v -e .
```

**Windows:** See the [CONTRIBUTING.md](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md#building-on-legacy-code-and-cuda) documentation.

**CPU-only builds:**

```cmd
python -m pip install --no-build-isolation -v -e .
```

**CUDA based builds:**

```cmd
# Set up variables
set CMAKE_INCLUDE_PATH={Your directory}\mkl\include
set LIB={Your directory}\mkl\lib;%LIB%
set CMAKE_GENERATOR_TOOLSET_VERSION=14.27
set DISTUTILS_USE_SDK=1
for /f "usebackq tokens=*" %i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -version [15^,17^) -products * -latest -property installationPath`) do call "%i\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=%CMAKE_GENERATOR_TOOLSET_VERSION%
set CUDAHOSTCXX=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.27.29110\bin\HostX64\x64\cl.exe
python -m pip install --no-build-isolation -v -e .
```

**Intel GPU builds:**

```cmd
# Set CMAKE_PREFIX_PATH (after conda activation)
if defined CMAKE_PREFIX_PATH (
    set "CMAKE_PREFIX_PATH=%CONDA_PREFIX%\Library;%CMAKE_PREFIX_PATH%"
) else (
    set "CMAKE_PREFIX_PATH=%CONDA_PREFIX%\Library"
)
python -m pip install --no-build-isolation -v -e .
```

##### Adjust Build Options (Optional)

Configure cmake variables before building by running:

**Linux/macOS:**

```bash
export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"
CMAKE_ONLY=1 python setup.py build
ccmake build  # or cmake-gui build
```

### Docker Image

Leverage pre-built or custom Docker images for a streamlined development environment.

#### Using pre-built images

```bash
docker run --gpus all --rm -ti --ipc=host pytorch/pytorch:latest
```

Use `--ipc=host` or `--shm-size` for multiprocessing.

#### Building the image yourself

```bash
make -f docker.Makefile
# images are tagged as docker.io/${your_docker_username}/pytorch
```

Use `CMAKE_VARS="..."` to specify additional CMake variables.

### Building the Documentation

Generate documentation in various formats using Sphinx.

```bash
cd docs/
pip install -r requirements.txt
make html
make serve
```

For PDF generation, install `texlive` and LaTeX and then:

```bash
make latexpdf
cd build/latex
make LATEXOPTS="-interaction=nonstopmode"
make LATEXOPTS="-interaction=nonstopmode"
```

### Previous Versions

Find installation instructions and binaries for previous PyTorch versions on [the PyTorch website](https://pytorch.org/get-started/previous-versions).

## Getting Started

Explore these resources to jumpstart your PyTorch journey:
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
*   Slack: [PyTorch Slack](https://pytorch.slack.com/) (requires an invite: https://goo.gl/forms/PP1AGvNHpSaJP8to1)
*   Newsletter: https://eepurl.com/cbG0rv
*   Facebook Page: https://www.facebook.com/pytorch

## Releases and Contributing

PyTorch has three minor releases per year. Please submit bug reports via [GitHub Issues](https://github.com/pytorch/pytorch/issues). Review the [Contribution page](CONTRIBUTING.md) and [Release page](RELEASE.md).

## The Team

PyTorch is a community-driven project. The core maintainers are: [Soumith Chintala](http://soumith.ch), [Gregory Chanan](https://github.com/gchanan), [Dmytro Dzhulgakov](https://github.com/dzhulgakov), [Edward Yang](https://github.com/ezyang), and [Nikita Shulga](https://github.com/malfet).

## License

PyTorch is licensed under a BSD-style license; see the [LICENSE](LICENSE) file.