[![PyTorch Logo](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/pytorch-logo-dark.png)](https://github.com/pytorch/pytorch)

# PyTorch: Deep Learning Made Flexible and Fast

**PyTorch is a powerful and versatile open-source deep learning framework, offering unparalleled flexibility and speed for research and production.**

## Key Features

*   **Tensor Computation with GPU Acceleration:**  Leverage the power of GPUs for lightning-fast tensor operations, similar to NumPy.
*   **Dynamic Neural Networks with Tape-Based Autograd:** Build and modify neural networks on the fly with PyTorch's unique reverse-mode auto-differentiation, providing unmatched flexibility.
*   **Python-First Approach:** Seamlessly integrate with the Python ecosystem, using your favorite libraries like NumPy, SciPy, and Cython.
*   **Imperative Design:** Experience an intuitive, straightforward programming style, simplifying debugging and understanding.
*   **Fast and Lean:** Optimized for speed, with minimal framework overhead and efficient memory management.
*   **Easy Extensions:** Extend PyTorch with custom layers and operations written in Python or C/C++, minimizing boilerplate.

## Quickstart

Install prebuilt binaries from [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

## Installation

Choose your preferred installation method:

### Binaries
Commands to install binaries via Conda or pip wheels are on our website: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

#### NVIDIA Jetson Platforms

Python wheels for NVIDIA's Jetson Nano, Jetson TX1/TX2, Jetson Xavier NX/AGX, and Jetson AGX Orin are provided [here](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048) and the L4T container is published [here](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch)

They require JetPack 4.2 and above, and [@dusty-nv](https://github.com/dusty-nv) and [@ptrblck](https://github.com/ptrblck) are maintaining them.

### From Source

#### Prerequisites

*   Python 3.9 or later
*   A C++17-compatible compiler (e.g., clang or gcc 9.4.0+)
*   Visual Studio or Visual Studio Build Tool (Windows)

    *   PyTorch CI uses Visual C++ BuildTools, which come with Visual Studio Enterprise,
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

##### NVIDIA CUDA Support

1.  Install a [supported version of CUDA](https://pytorch.org/get-started/locally/).
2.  Install [cuDNN](https://developer.nvidia.com/cudnn) v8.5 or above.
3.  Ensure a compiler compatible with CUDA is available.
    *   Set `USE_CUDA=0` to disable CUDA.

##### AMD ROCm Support

*   Install [AMD ROCm](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html) 4.0 and above.
    *   ROCm is only supported on Linux.
    *   Set `ROCM_PATH` if ROCm is installed outside the default `/opt/rocm` directory.
    *   Optionally, set `PYTORCH_ROCM_ARCH`.
    *   Set `USE_ROCM=0` to disable ROCm.

##### Intel GPU Support

*   Follow the [PyTorch Prerequisites for Intel GPUs](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpus.html).
    *   Intel GPU is supported for Linux and Windows.
    *   Set `USE_XPU=0` to disable Intel GPU support.

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

[NVTX](https://docs.nvidia.com/gameworks/content/gameworkslibrary/nvtx/nvidia_tools_extension_library_nvtx/nvtx.htm) is needed to build Pytorch with CUDA.
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

Use `CMAKE_ONLY=1 python setup.py build` followed by `ccmake build` or `cmake-gui build` to adjust CMake variables.

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
cd build/latex
make LATEXOPTS="-interaction=nonstopmode"
make LATEXOPTS="-interaction=nonstopmode"  # Run again for ToC & index
```

### Previous Versions

Find installation instructions for previous versions [on our website](https://pytorch.org/get-started/previous-versions).

## Getting Started

*   [Tutorials](https://pytorch.org/tutorials/)
*   [Examples](https://github.com/pytorch/examples)
*   [API Reference](https://pytorch.org/docs/)
*   [Glossary](https://github.com/pytorch/pytorch/blob/main/GLOSSARY.md)

## Components

*   [**torch**](https://pytorch.org/docs/stable/torch.html): Tensor library with GPU support.
*   [**torch.autograd**](https://pytorch.org/docs/stable/autograd.html): Automatic differentiation.
*   [**torch.jit**](https://pytorch.org/docs/stable/jit.html): Model compilation.
*   [**torch.nn**](https://pytorch.org/docs/stable/nn.html): Neural network library.
*   [**torch.multiprocessing**](https://pytorch.org/docs/stable/multiprocessing.html): Multiprocessing with tensor sharing.
*   [**torch.utils**](https://pytorch.org/docs/stable/data.html): Data loading utilities.

## Resources

*   [PyTorch.org](https://pytorch.org/)
*   [PyTorch Tutorials](https://pytorch.org/tutorials/)
*   [PyTorch Examples](https://github.com/pytorch/examples)
*   [PyTorch Models](https://pytorch.org/hub/)
*   ... (other links from the original) ...

## Communication

*   [Forums](https://discuss.pytorch.org)
*   [GitHub Issues](https://github.com/pytorch/pytorch/issues)
*   [Slack](https://pytorch.slack.com/)
*   [Newsletter](https://eepurl.com/cbG0rv)
*   [Facebook](https://www.facebook.com/pytorch)

## Releases and Contributing

[Contribute](CONTRIBUTING.md)  and learn more about releases at [RELEASE.md](RELEASE.md)
*   [File an issue](https://github.com/pytorch/pytorch/issues) to report bugs.

## The Team

The PyTorch community is a global and collaborative effort.