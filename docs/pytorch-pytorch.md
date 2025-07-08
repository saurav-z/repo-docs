[![PyTorch Logo](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/pytorch-logo-dark.png)](https://github.com/pytorch/pytorch)

# PyTorch: Deep Learning with Ease and Power

**PyTorch is a leading open-source deep learning framework, providing a flexible and efficient platform for building and training machine learning models.** Get started and contribute to the future of deep learning on the [official PyTorch repository](https://github.com/pytorch/pytorch).

## Key Features

*   **Tensor Computation with GPU Acceleration:** Offers tensor operations similar to NumPy, but with robust support for GPU acceleration, enabling faster computation.
*   **Dynamic Neural Networks:** Built on a tape-based autograd system, allowing for flexible and dynamic neural network architectures.
*   **Python-First Design:** Deeply integrated with Python, allowing for seamless integration with existing Python libraries like NumPy, SciPy, and Cython.
*   **Imperative Programming:** Designed for intuitive use with an imperative programming style, making debugging and development easier.
*   **Fast and Lean:** Optimized for speed with minimal framework overhead, incorporating acceleration libraries and efficient memory management.
*   **Easy Extensions:** Provides straightforward APIs for creating new neural network modules and interfacing with PyTorch's tensor API, including easy C/C++ extensions.

## Core Components

PyTorch is built on several key components:

*   **torch:** Tensor library (similar to NumPy) with GPU support ([torch documentation](https://pytorch.org/docs/stable/torch.html))
*   **torch.autograd:** Automatic differentiation library ([torch.autograd documentation](https://pytorch.org/docs/stable/autograd.html))
*   **torch.jit:** Compilation stack (TorchScript) for serializable and optimizable models ([torch.jit documentation](https://pytorch.org/docs/stable/jit.html))
*   **torch.nn:** Neural networks library integrated with autograd ([torch.nn documentation](https://pytorch.org/docs/stable/nn.html))
*   **torch.multiprocessing:** Python multiprocessing with shared memory for tensors ([torch.multiprocessing documentation](https://pytorch.org/docs/stable/multiprocessing.html))
*   **torch.utils:** Data loading utilities and more ([torch.utils documentation](https://pytorch.org/docs/stable/data.html))

## Installation

For detailed installation instructions, including instructions for CUDA, ROCm, and Intel GPU support, visit the [PyTorch Installation Guide](https://pytorch.org/get-started/locally/).

###  Binaries
Install using Conda or pip wheels: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

#### NVIDIA Jetson Platforms
Prebuilt wheels are available for Jetson platforms:  [here](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048) and  [here](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch)

### From Source

#### Prerequisites
Requires Python 3.9+, C++17 compatible compiler, and (Windows) Visual Studio or Build Tool.

##### NVIDIA CUDA Support
Install NVIDIA CUDA and cuDNN, and compatible compiler.
See [CUDA Support Matrix](https://pytorch.org/get-started/locally/) for compatibility.

##### AMD ROCm Support
Install AMD ROCm 4.0+ on Linux.
Set `ROCM_PATH` environment variable if necessary.

##### Intel GPU Support
Follow the prerequisites and install instructions at:
[PyTorch Prerequisites for Intel GPUs](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpus.html)

#### Get the PyTorch Source

```bash
git clone https://github.com/pytorch/pytorch
cd pytorch
git submodule sync
git submodule update --init --recursive
```

#### Install Dependencies
(See original README for commands - consolidated for brevity)
```bash
conda install cmake ninja # Linux/Mac
pip install -r requirements.txt
# Linux: pip install mkl-static mkl-include
# macOS: conda install pkg-config libuv
# Windows: conda install -c conda-forge libuv=1.39
# Install magma
.ci/docker/common/install_magma_conda.sh 12.4  # CUDA only
# (Optional) Triton install
make triton # After cloning, Intel GPU, ROCm builds
```
#### Install PyTorch
*   **Linux (AMD ROCm):**  `python tools/amd_build/build_amd.py` then `python setup.py develop`
*   **Linux/macOS:** `export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"` then  `python setup.py develop`
*   **Windows:** See original README for setup & commands - includes CUDA-specific and OpenMP considerations.

#### Adjust Build Options (Optional)

```bash
# Linux/macOS
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
# images are tagged as docker.io/${your_docker_username}/pytorch
```

### Building the Documentation

```bash
cd docs/
pip install -r requirements.txt
make html
make serve
```

### Previous Versions

Find installation instructions and binaries for previous PyTorch versions on our [website](https://pytorch.org/get-started/previous-versions).

## Getting Started & Resources

*   [PyTorch Tutorials](https://pytorch.org/tutorials/)
*   [PyTorch Examples](https://github.com/pytorch/examples)
*   [PyTorch Models](https://pytorch.org/hub/)
*   [The API Reference](https://pytorch.org/docs/)
*   [Glossary](https://github.com/pytorch/pytorch/blob/main/GLOSSARY.md)

## Communication

*   Forums: [Discuss PyTorch](https://discuss.pytorch.org)
*   GitHub Issues: [Report Bugs & Feature Requests](https://github.com/pytorch/pytorch/issues)
*   Slack: [PyTorch Slack](https://pytorch.slack.com/) - for experienced users & developers (request invite)
*   Newsletter: [PyTorch Newsletter](https://eepurl.com/cbG0rv)
*   Facebook Page: [PyTorch Facebook Page](https://www.facebook.com/pytorch)

## Releases and Contributing

PyTorch typically releases three minor versions per year. [File an issue](https://github.com/pytorch/pytorch/issues) for bug reports.

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.
Learn more about PyTorch releases at [RELEASE.md](RELEASE.md).

## The Team

PyTorch is a community-driven project, maintained by [Soumith Chintala](http://soumith.ch), [Gregory Chanan](https://github.com/gchanan), [Dmytro Dzhulgakov](https://github.com/dzhulgakov), [Edward Yang](https://github.com/ezyang), and [Nikita Shulga](https://github.com/malfet), with contributions from many others.

## License

PyTorch is available under a [BSD-style license](LICENSE).