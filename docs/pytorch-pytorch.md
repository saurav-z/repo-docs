[![PyTorch Logo](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/pytorch-logo-dark.png)](https://github.com/pytorch/pytorch)

# PyTorch: Deep Learning with Python and GPU Acceleration

**PyTorch is a powerful and flexible open-source machine learning framework that accelerates the path from research prototyping to production deployment.** Access the original repo [here](https://github.com/pytorch/pytorch).

**Key Features:**

*   **Tensor Computation with GPU Acceleration:**  Leverage the power of GPUs for fast tensor operations, similar to NumPy, accelerating your machine learning workflows.
*   **Dynamic Neural Networks with Tape-Based Autograd:** Build and modify neural networks on the fly with PyTorch's flexible tape-based autograd system, perfect for cutting-edge research.
*   **Python-First Design:**  Enjoy seamless integration with your existing Python ecosystem, including NumPy, SciPy, and Cython.
*   **Imperative Programming:**  Experience intuitive and easy-to-debug code execution, with clear stack traces and straightforward debugging.
*   **Fast and Lean:** Benefit from minimal framework overhead and optimized performance, thanks to integration with acceleration libraries like Intel MKL, cuDNN, and NCCL.
*   **Extensible and Customizable:**  Easily write custom neural network modules and extend the functionality of PyTorch using Python or C/C++.

## Quick Links

*   [Getting Started](https://pytorch.org/get-started/locally/)
*   [Tutorials](https://pytorch.org/tutorials/)
*   [Examples](https://github.com/pytorch/examples)
*   [API Reference](https://pytorch.org/docs/)
*   [PyTorch.org](https://pytorch.org/)
*   [Resources](#resources)
*   [Communication](#communication)
*   [Releases and Contributing](#releases-and-contributing)

## Key Components of PyTorch

PyTorch is a comprehensive library with key components for all your deep learning needs:

| Component                | Description                                                                                                                                                                             |
| :----------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **torch**                | A Tensor library like NumPy, with strong GPU support. ([torch documentation](https://pytorch.org/docs/stable/torch.html))                                                                                               |
| **torch.autograd**       | A tape-based automatic differentiation library that supports all differentiable Tensor operations in torch. ([torch.autograd documentation](https://pytorch.org/docs/stable/autograd.html))                                |
| **torch.jit**            | A compilation stack (TorchScript) to create serializable and optimizable models from PyTorch code. ([torch.jit documentation](https://pytorch.org/docs/stable/jit.html))                                                  |
| **torch.nn**             | A neural networks library deeply integrated with autograd designed for maximum flexibility. ([torch.nn documentation](https://pytorch.org/docs/stable/nn.html))                                           |
| **torch.multiprocessing** | Python multiprocessing, but with magical memory sharing of torch Tensors across processes. Useful for data loading and Hogwild training.  ([torch.multiprocessing documentation](https://pytorch.org/docs/stable/multiprocessing.html)) |
| **torch.utils**          | DataLoader and other utility functions for convenience. ([torch.utils documentation](https://pytorch.org/docs/stable/data.html))                                                  |

## Installation

Get started with PyTorch quickly with these installation options:

### Binaries

Install pre-built binaries via Conda or pip wheels.  Follow the instructions on our website: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

#### NVIDIA Jetson Platforms

Pre-built wheels for NVIDIA Jetson Nano, TX1/TX2, Xavier NX/AGX, and AGX Orin are available [here](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048). The L4T container is published [here](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch).

### From Source

Build PyTorch from source for more customization.

#### Prerequisites

Ensure you have the following:

*   Python 3.9 or later
*   A C++17 compliant compiler (e.g., clang or gcc, gcc 9.4.0+ required on Linux)
*   Visual Studio or Visual Studio Build Tool (Windows)

    *   Visual Studio BuildTools are available at [https://visualstudio.microsoft.com/visual-cpp-build-tools/](https://visualstudio.microsoft.com/visual-cpp-build-tools/).  They do *not* come with Visual Studio Code.

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

A conda environment is not required; standard virtual environments (e.g. with `uv`) are sufficient.

##### NVIDIA CUDA Support

Compile with CUDA support by installing:

*   [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads) (select a supported version from our support matrix: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)).
*   [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) v8.5 or above
*   A [CUDA-compatible compiler](https://gist.github.com/ax3l/9489132)

To disable CUDA, set `USE_CUDA=0`.

##### AMD ROCm Support

Compile with ROCm support by installing:

*   [AMD ROCm](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html) 4.0 and above (Linux only).

Set `ROCM_PATH` if ROCm is installed in a non-default directory, and `PYTORCH_ROCM_ARCH` for AMD GPU architecture (optional).  To disable ROCm, set `USE_ROCM=0`.

##### Intel GPU Support

Compile with Intel GPU support following these [PyTorch Prerequisites for Intel GPUs](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpus.html) instructions.  Supported on Linux and Windows. Set `USE_XPU=0` to disable.

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
.ci/docker/common/install_magma_conda.sh 12.4 # CUDA only
make triton # (optional, for torch.compile with inductor/triton)
```

**On MacOS**

```bash
pip install mkl-static mkl-include
conda install pkg-config libuv  # for torch.distributed
```

**On Windows**

```bash
pip install mkl-static mkl-include
conda install -c conda-forge libuv=1.39 # for torch.distributed
```

#### Install PyTorch

**On Linux (AMD ROCm)**

```bash
python tools/amd_build/build_amd.py # if compiling for ROCm
export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"
python setup.py develop
```

**On macOS**

```bash
python3 setup.py develop
```

**On Windows**

Follow instructions at [Building on legacy code and CUDA](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md#building-on-legacy-code-and-cuda) if building legacy code.

**CPU-only Builds**

```cmd
python setup.py develop
```

**CUDA Based Build**

*   Requires Nsight Compute (part of CUDA distribution).
*   Requires libraries like [Magma](https://developer.nvidia.com/magma), [oneDNN](https://github.com/oneapi-src/oneDNN), and [Sccache](https://github.com/mozilla/sccache). Use the [installation-helper](https://github.com/pytorch/pytorch/tree/main/.ci/pytorch/win-test-helpers/installation-helpers) for assistance.

Refer to [.ci/pytorch/win-test-helpers/build_pytorch.bat](https://github.com/pytorch/pytorch/blob/main/.ci/pytorch/win-test-helpers/build_pytorch.bat) for environment variable configurations.

```cmd
cmd
set CMAKE_INCLUDE_PATH={Your directory}\mkl\include
set LIB={Your directory}\mkl\lib;%LIB%
set CMAKE_GENERATOR_TOOLSET_VERSION=14.27
set DISTUTILS_USE_SDK=1
for /f "usebackq tokens=*" %i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -version [15^,17^) -products * -latest -property installationPath`) do call "%i\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=%CMAKE_GENERATOR_TOOLSET_VERSION%
set CUDAHOSTCXX=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.27.29110\bin\HostX64\x64\cl.exe
python setup.py develop
```

**Intel GPU Builds**

*   Follow [common and Intel GPU prerequisites](#prerequisites).
*   Requires Visual Studio 2022.

```cmd
if defined CMAKE_PREFIX_PATH (
    set "CMAKE_PREFIX_PATH=%CONDA_PREFIX%\Library;%CMAKE_PREFIX_PATH%"
) else (
    set "CMAKE_PREFIX_PATH=%CONDA_PREFIX%\Library"
)
python setup.py develop
```

##### Adjust Build Options (Optional)

Adjust CMake variables for CuDNN/BLAS/etc.

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

Pull a pre-built Docker image from Docker Hub:

```bash
docker run --gpus all --rm -ti --ipc=host pytorch/pytorch:latest
```

#### Building the image yourself

**NOTE:**  Requires Docker > 18.06.

```bash
make -f docker.Makefile
# images are tagged as docker.io/${your_docker_username}/pytorch
```

### Building the Documentation

Build documentation using Sphinx and pytorch_sphinx_theme2:

```bash
cd docs/
pip install -r requirements.txt
make html
make serve
```

To build a PDF:

```bash
make latexpdf
make LATEXOPTS="-interaction=nonstopmode"
```

### Previous Versions

Find installation instructions and binaries for previous PyTorch versions on [our website](https://pytorch.org/get-started/previous-versions).

## Getting Started Resources

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

*   **Forums:** Discuss implementations, research, etc. https://discuss.pytorch.org
*   **GitHub Issues:** Bug reports, feature requests, installation issues, RFCs, etc.
*   **Slack:** [PyTorch Slack](https://pytorch.slack.com/) for experienced users and developers. Get an invite: https://goo.gl/forms/PP1AGvNHpSaJP8to1 (beginners should use the PyTorch Forums).
*   **Newsletter:** Sign up for the PyTorch newsletter: https://eepurl.com/cbG0rv
*   **Facebook Page:** https://www.facebook.com/pytorch
*   **Brand Guidelines:**  Visit [pytorch.org](https://pytorch.org/)

## Releases and Contributing

PyTorch typically releases three minor versions per year. Report bugs via [GitHub Issues](https://github.com/pytorch/pytorch/issues).

Contribute bug fixes directly.  Discuss new features and extensions by opening an issue before submitting a PR.  See [CONTRIBUTING.md](CONTRIBUTING.md) and [RELEASE.md](RELEASE.md).

## The Team

PyTorch is a community-driven project maintained by [Soumith Chintala](http://soumith.ch), [Gregory Chanan](https://github.com/gchanan), [Dmytro Dzhulgakov](https://github.com/dzhulgakov), [Edward Yang](https://github.com/ezyang), and [Nikita Shulga](https://github.com/malfet), with contributions from hundreds of talented individuals.  See the original repo for a non-exhaustive list.

## License

PyTorch is licensed under a BSD-style license; see the [LICENSE](LICENSE) file.