![PyTorch Logo](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/pytorch-logo-dark.png)

# PyTorch: The Open Source Deep Learning Platform

**PyTorch is a leading open-source deep learning framework, designed for flexibility and speed in research and production.**  [Visit the original PyTorch repository](https://github.com/pytorch/pytorch).

## Key Features of PyTorch

*   **GPU-Accelerated Tensor Computation:**  Provides tensor operations (like NumPy) with seamless GPU acceleration, enabling fast numerical computation.
*   **Dynamic Neural Networks with Autograd:**  Built on a tape-based autograd system, allowing for flexible and dynamic neural network architectures.
*   **Pythonic Design:** Deeply integrated with Python, enabling intuitive use of existing libraries like NumPy, SciPy, and Cython.
*   **Imperative Programming:** Offers an imperative programming experience that is easy to debug and understand, with clear stack traces.
*   **Fast and Lean:** Optimized for speed and efficiency, integrating with acceleration libraries like Intel MKL, cuDNN, and NCCL.
*   **Extensible:** Offers a straightforward API for creating new neural network modules and custom operations.

## Core Components

PyTorch is built upon a set of key components:

*   [**torch**](https://pytorch.org/docs/stable/torch.html): The core Tensor library with GPU support, akin to NumPy.
*   [**torch.autograd**](https://pytorch.org/docs/stable/autograd.html): Enables automatic differentiation for all differentiable tensor operations.
*   [**torch.jit**](https://pytorch.org/docs/stable/jit.html): Facilitates the creation of serializable and optimized models through TorchScript.
*   [**torch.nn**](https://pytorch.org/docs/stable/nn.html): A neural network library deeply integrated with autograd, built for maximum flexibility.
*   [**torch.multiprocessing**](https://pytorch.org/docs/stable/multiprocessing.html): Provides Python multiprocessing with efficient memory sharing for tensors across processes.
*   [**torch.utils**](https://pytorch.org/docs/stable/data.html): Contains DataLoader and other utility functions to improve convenience.

## Installation

Install PyTorch to leverage the power of deep learning for your projects.

### Install binaries

*   **Via Conda:** [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
*   **Via Pip:** [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

#### NVIDIA Jetson Platforms
*   **Jetson Nano, TX1/TX2, Xavier NX/AGX, and AGX Orin:** [Installation instructions](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048)
*   **L4T Container:** [L4T container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch)

### Install from Source

Build PyTorch from source to take advantage of advanced features and performance improvements.

#### Prerequisites

*   Python 3.9 or later
*   C++17 compliant compiler (e.g., gcc 9.4.0+)
*   Visual Studio or Visual Studio Build Tool (Windows only)

#### Dependencies

*   `conda install cmake ninja`
*   `pip install -r requirements.txt`
*   **Linux:** `pip install mkl-static mkl-include`
*   **macOS:** `pip install mkl-static mkl-include`
*   **Windows:** `pip install mkl-static mkl-include`

#### Steps

1.  Clone the repository:
    ```bash
    git clone https://github.com/pytorch/pytorch
    cd pytorch
    git submodule sync
    git submodule update --init --recursive
    ```

2.  Install Dependencies (common + OS specific) :
    ```bash
    conda install cmake ninja
    pip install -r requirements.txt
    ```
    **(Linux Specific)**
     ```bash
    pip install mkl-static mkl-include
    # CUDA Only: Add LAPACK support for the GPU if needed
    # magma installation: run with active conda environment. specify CUDA version to install
    .ci/docker/common/install_magma_conda.sh 12.4

    # (optional) If using torch.compile with inductor/triton, install the matching version of triton
    # Run from the pytorch directory after cloning
    # For Intel GPU support, please explicitly `export USE_XPU=1` before running command.
    make triton
     ```
    **(macOS Specific)**
     ```bash
    # Add this package on intel x86 processor machines only
    pip install mkl-static mkl-include
    # Add these packages if torch.distributed is needed
    conda install pkg-config libuv
     ```
    **(Windows Specific)**
    ```bash
    pip install mkl-static mkl-include
    # Add these packages if torch.distributed is needed.
    # Distributed package support on Windows is a prototype feature and is subject to changes.
    conda install -c conda-forge libuv=1.39
    ```

3.  Install PyTorch. **(Linux, macOS and Windows)**
    ```bash
    # AMD ROCm build (Linux):
    python tools/amd_build/build_amd.py
    # (Linux and macOS only)
    export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"
    python setup.py develop
    ```
    **(Windows Only)**
    ```bash
    python setup.py develop
    ```

#### CUDA, ROCm, and Intel GPU Support

*   **CUDA:**
    *   [Install NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads), cuDNN v8.5+, and a compatible compiler.
    *   Set `USE_CUDA=0` to disable CUDA support.
    *   Use the [cuDNN Support Matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/support-matrix.html) for information on cuDNN/CUDA driver compatibility.
*   **ROCm:**
    *   Install [AMD ROCm](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html) 4.0+.
    *   Set `USE_ROCM=0` to disable ROCm support.
    *   Set `ROCM_PATH` if ROCm is not installed in `/opt/rocm`.
    *   Optionally set `PYTORCH_ROCM_ARCH`.
*   **Intel GPU:**
    *   Follow the [PyTorch Prerequisites for Intel GPUs](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpus.html) instructions.
    *   Set `USE_XPU=0` to disable Intel GPU support.

### Docker Image

*   **Pre-built images:** Pull from Docker Hub: `docker run --gpus all --rm -ti --ipc=host pytorch/pytorch:latest`
*   **Build your own image:** Use the provided `Dockerfile`.

### Building the Documentation

*   Install Sphinx and the pytorch_sphinx_theme2: `pip install -r requirements.txt`
*   Build the documentation: `cd docs/` and then `make html` or `make latexpdf`
*   To build the PDF, run `make latexpdf` and then `make LATEXOPTS="-interaction=nonstopmode"` in the `build/latex` directory.

### Previous Versions

Instructions and binaries for previous versions are available [on our website](https://pytorch.org/get-started/previous-versions).

## Getting Started

*   [Tutorials](https://pytorch.org/tutorials/): Introductory guides to learning PyTorch.
*   [Examples](https://github.com/pytorch/examples): Ready-to-use PyTorch code across various domains.
*   [API Reference](https://pytorch.org/docs/): Comprehensive documentation of PyTorch's API.
*   [Glossary](https://github.com/pytorch/pytorch/blob/main/GLOSSARY.md): Definitions of PyTorch terminology.

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

*   [Forums](https://discuss.pytorch.org): Discuss implementations and research.
*   [GitHub Issues](https://github.com/pytorch/pytorch/issues): Report bugs, request features.
*   [Slack](https://pytorch.slack.com/): Join the PyTorch Slack for general chat and collaboration (fill out [this form](https://goo.gl/forms/PP1AGvNHpSaJP8to1) for an invite).
*   [Newsletter](https://eepurl.com/cbG0rv): Sign up for important announcements.
*   [Facebook Page](https://www.facebook.com/pytorch): Follow for announcements.
*   For brand guidelines, please visit [pytorch.org](https://pytorch.org/).

## Releases and Contributing

PyTorch typically releases three minor versions per year. [File an issue](https://github.com/pytorch/pytorch/issues) to report bugs.

We welcome contributions. For new features, first open an issue to discuss them.  See the [Contribution page](CONTRIBUTING.md) and [Release page](RELEASE.md) for details.

## The Team

PyTorch is a community-driven project, currently maintained by [Soumith Chintala](http://soumith.ch), [Gregory Chanan](https://github.com/gchanan), [Dmytro Dzhulgakov](https://github.com/dzhulgakov), [Edward Yang](https://github.com/ezyang), and [Nikita Shulga](https://github.com/malfet), with significant contributions from a wide range of talented individuals. A non-exhaustive list is provided for reference.

Note: This project is unrelated to [hughperkins/pytorch](https://github.com/hughperkins/pytorch).

## License

PyTorch is licensed under a BSD-style license, found in the [LICENSE](LICENSE) file.