[![PyTorch Logo](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/pytorch-logo-dark.png)](https://github.com/pytorch/pytorch)

# PyTorch: Deep Learning with GPU Acceleration

**PyTorch is a powerful and flexible open-source deep learning framework, providing GPU-accelerated tensor computation and dynamic neural networks for cutting-edge research and development.** Learn more and contribute to the project at the [original PyTorch repository](https://github.com/pytorch/pytorch).

## Key Features

*   **GPU-Accelerated Tensors:** Leverage the power of GPUs for lightning-fast tensor operations, mirroring NumPy functionality.
*   **Dynamic Neural Networks with Autograd:** Build and modify neural networks with unparalleled flexibility using tape-based autograd for dynamic computation graphs.
*   **Python-First Approach:** Seamlessly integrate PyTorch with your existing Python workflows, utilizing familiar libraries like NumPy, SciPy, and Cython.
*   **Imperative and Intuitive:** Experience easy debugging and clear stack traces with PyTorch's imperative programming style.
*   **Fast and Lean:** Benefit from optimized acceleration libraries and custom memory allocators for efficient performance and memory usage.
*   **Extensible and Customizable:** Easily create custom neural network modules and integrate with PyTorch's tensor API using Python or C/C++.

## Installation

Get started quickly by installing pre-built binaries or building from source, with options for various platforms.

### Binaries

Install PyTorch using `conda` or `pip`, details on the [PyTorch website](https://pytorch.org/get-started/locally/).

#### NVIDIA Jetson Platforms

Pre-built wheels are available for NVIDIA Jetson platforms: [Jetson PyTorch Wheels](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048) and [L4T container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch).

### From Source

Build PyTorch from source to customize your installation.

#### Prerequisites

*   Python 3.9 or later
*   A C++17-compliant compiler (e.g., GCC 9.4.0 or newer)
*   Visual Studio or Visual Studio Build Tool (Windows)

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
- [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads)
- [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) v8.5 or above
- [Compiler](https://gist.github.com/ax3l/9489132) compatible with CUDA

##### AMD ROCm Support
- [AMD ROCm](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html) 4.0 and above installation

##### Intel GPU Support
- [PyTorch Prerequisites for Intel GPUs](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpus.html) instructions.

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

```bash
python tools/amd_build/build_amd.py  # ROCm only
export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"
python -m pip install --no-build-isolation -v -e .
```

**On macOS**

```bash
python -m pip install --no-build-isolation -v -e .
```

**On Windows**

```cmd
:: CPU only
python -m pip install --no-build-isolation -v -e .
```

**CUDA based build**

```cmd
:: Optional Ninja and CUDA configurations
set CMAKE_GENERATOR_TOOLSET_VERSION=14.27
set DISTUTILS_USE_SDK=1
for /f "usebackq tokens=*" %i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -version [15^,17^) -products * -latest -property installationPath`) do call "%i\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=%CMAKE_GENERATOR_TOOLSET_VERSION%

:: Optional CUDA host compiler
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

### Previous Versions

See [Previous Versions](https://pytorch.org/get-started/previous-versions) for previous versions.

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
*   [Udacity Deep Learning with PyTorch](https://www.udacity.com/course/deep-learning-pytorch--ud188)
*   [Coursera Deep Neural Networks with PyTorch](https://www.coursera.org/learn/deep-neural-networks-with-pytorch)
*   [PyTorch Twitter](https://twitter.com/PyTorch)
*   [PyTorch Blog](https://pytorch.org/blog/)
*   [PyTorch YouTube](https://www.youtube.com/channel/UCWXI5YeOsh03QvJ59PMaXFw)

## Communication

*   [Forums](https://discuss.pytorch.org)
*   [GitHub Issues](https://github.com/pytorch/pytorch/issues)
*   [Slack](https://pytorch.slack.com/)
*   [Newsletter](https://eepurl.com/cbG0rv)
*   [Facebook](https://www.facebook.com/pytorch)

## Releases and Contributing

*   [File an Issue](https://github.com/pytorch/pytorch/issues)
*   [Contribution Page](CONTRIBUTING.md)
*   [Release Page](RELEASE.md)

## The Team

Maintained by Soumith Chintala, Gregory Chanan, Dmytro Dzhulgakov, Edward Yang, and Nikita Shulga.
Includes contributions from many talented individuals.

## License

PyTorch is licensed under a [BSD-style license](LICENSE).