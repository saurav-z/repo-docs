![PyTorch Logo](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/pytorch-logo-dark.png)

# PyTorch: The Deep Learning Framework of Choice

PyTorch is an open-source machine learning framework that provides a flexible platform for research and production, offering both tensor computation and deep neural networks. **Unlock the power of GPU acceleration and dynamic neural networks with PyTorch, the leading deep learning framework.** Learn more and get started at the [original PyTorch repository](https://github.com/pytorch/pytorch).

## Key Features of PyTorch

*   **GPU-Accelerated Tensor Computation:** Experience lightning-fast tensor operations, similar to NumPy, with seamless GPU support for accelerated computations.
*   **Dynamic Neural Networks with Autograd:** Benefit from a tape-based autograd system that allows for flexible and dynamic neural network design, enabling rapid prototyping and experimentation.
*   **Python-First Design:** Seamlessly integrate PyTorch into your Python workflows, leveraging familiar tools like NumPy, SciPy, and Cython for a streamlined development process.
*   **Imperative Style & Intuitive Debugging:** Enjoy an imperative coding style that is easy to understand and debug, with clear stack traces that pinpoint the source of errors.
*   **Fast and Lean Performance:** Utilize optimized libraries like Intel MKL and NVIDIA cuDNN/NCCL to maximize speed and efficiency, whether running small or large neural networks.
*   **Effortless Extensions:** Easily create custom neural network modules and interface with PyTorch's Tensor API.

## Getting Started

### Installation

Install PyTorch using either binaries or building from source:

*   **Binaries:** Refer to [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) for detailed installation commands using Conda or pip.
*   **From Source:** Build PyTorch from source for advanced customization. Prerequisites include:

    *   Python 3.9 or later
    *   A C++17-compatible compiler (e.g., gcc 9.4.0 or newer)
    *   Visual Studio or Visual Studio Build Tool (Windows only)

    For CUDA, ROCm, or Intel GPU support, refer to the detailed instructions below and the PyTorch documentation.

#### NVIDIA CUDA Support

If compiling with CUDA, ensure you have:

*   [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads)
*   [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) v8.5 or above
*   A CUDA-compatible compiler ([Compiler](https://gist.github.com/ax3l/9489132))

#### AMD ROCm Support

For ROCm support, install:

*   [AMD ROCm](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html) 4.0 and above.
*   This is supported for Linux systems.

#### Intel GPU Support

For Intel GPU builds, follow the [PyTorch Prerequisites for Intel GPUs](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpus.html).

#### Installation Commands

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

**CPU-only builds**

In this mode PyTorch computations will run on your CPU, not your GPU.

```cmd
python -m pip install --no-build-isolation -v -e .
```

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

### Docker Image

*   **Pre-built Images:** Pull pre-built images from Docker Hub:

    ```bash
    docker run --gpus all --rm -ti --ipc=host pytorch/pytorch:latest
    ```

*   **Building Your Own Image:** Build a custom Docker image using the provided `Dockerfile`.

### Building Documentation

*   Install Sphinx and pytorch\_sphinx\_theme2.
*   Run `make html` to build the HTML documentation.

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

*   **Forums:** [https://discuss.pytorch.org](https://discuss.pytorch.org)
*   **GitHub Issues:** Bug reports, feature requests, etc.
*   **Slack:** [PyTorch Slack](https://pytorch.slack.com/) (request an invite via the provided form)
*   **Newsletter:** Subscribe to the no-noise email newsletter: [https://eepurl.com/cbG0rv](https://eepurl.com/cbG0rv)
*   **Facebook Page:** [https://www.facebook.com/pytorch](https://www.facebook.com/pytorch)

## Releases and Contributing

PyTorch has regular releases.  Please report bugs by [filing an issue](https://github.com/pytorch/pytorch/issues).

Contributions are welcome! Follow the [Contribution page](CONTRIBUTING.md) for instructions.  See also [Release page](RELEASE.md).

## The Team

PyTorch is a community-driven project. The core maintainers include Soumith Chintala, Gregory Chanan, Dmytro Dzhulgakov, Edward Yang, and Nikita Shulga, with contributions from numerous talented individuals.

## License

PyTorch is licensed under a BSD-style license; see the [LICENSE](LICENSE) file for details.