[![PyTorch Logo](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/pytorch-logo-dark.png)](https://github.com/pytorch/pytorch)

# PyTorch: The Flexible and Fast Deep Learning Framework

**PyTorch is a leading open-source machine learning framework that accelerates the path from research prototyping to production deployment.**

## Key Features

*   **GPU-Accelerated Tensor Computation:**  Perform tensor operations, similar to NumPy, with powerful GPU support for rapid calculations.
*   **Dynamic Neural Networks with Autograd:** Build and modify neural networks with unparalleled flexibility using a tape-based autograd system for dynamic computation graphs.
*   **Python-First Design:** Enjoy deep Python integration, leveraging your existing scientific computing ecosystem and libraries like NumPy and SciPy.
*   **Imperative Programming Style:** Write and debug your code with an intuitive and straightforward imperative programming experience.
*   **Fast and Lean:** Benefit from minimal framework overhead and optimized acceleration libraries like Intel MKL and NVIDIA cuDNN, ensuring speed and efficiency.
*   **Extensible:** Easily create custom neural network modules or integrate with PyTorch's Tensor API using Python or C++.

## Core Components

PyTorch is built upon key components:

| Component            | Description                                                                                                |
| -------------------- | ---------------------------------------------------------------------------------------------------------- |
| **torch**            | Tensor library with GPU acceleration.                                                                       |
| **torch.autograd**   | Automatic differentiation for all differentiable tensor operations.                                           |
| **torch.jit**        | TorchScript for creating serializable and optimizable models.                                               |
| **torch.nn**         | Neural networks library deeply integrated with autograd for maximum flexibility.                                  |
| **torch.multiprocessing** |  Python multiprocessing with shared memory for tensors.  |
| **torch.utils**      | DataLoader and other utility functions for convenience.                                                      |

## Installation

Choose the installation method that best suits your needs. For detailed commands and instructions, visit the official PyTorch website:  [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

### Binaries

*   **Conda:**  Install using Conda package manager.
*   **Pip:** Install pre-built packages using pip.

#### NVIDIA Jetson Platforms

*   Specific wheel files are available for NVIDIA Jetson Nano, TX1/TX2, Xavier NX/AGX, and AGX Orin platforms. [See here](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048) and [here](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch)

### From Source

Installing from source provides the most flexibility, but requires more steps and specific prerequisites.

#### Prerequisites

*   Python 3.9 or later
*   A C++17-compatible compiler (gcc 9.4.0+ on Linux)
*   Visual Studio or Visual Studio Build Tool (Windows)

#### NVIDIA CUDA Support

1.  Install a [supported CUDA version](https://pytorch.org/get-started/locally/).
2.  Install [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads), [cuDNN](https://developer.nvidia.com/cudnn), and a compatible compiler.

#### AMD ROCm Support

*   Install [AMD ROCm](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html) 4.0 and above.

#### Intel GPU Support

*   Follow the [PyTorch Prerequisites for Intel GPUs](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpus.html) instructions.

#### Get the PyTorch Source

```bash
git clone https://github.com/pytorch/pytorch
cd pytorch
git submodule sync
git submodule update --init --recursive
```

#### Install Dependencies

*   **Common:**  `conda install cmake ninja`; `pip install -r requirements.txt`

*   **Linux:** `pip install mkl-static mkl-include`; CUDA-specific instructions in original README.

*   **macOS:** `pip install mkl-static mkl-include`; `conda install pkg-config libuv`

*   **Windows:** `pip install mkl-static mkl-include`; `conda install -c conda-forge libuv=1.39`

#### Install PyTorch

*   **Linux (with ROCm):**  `python tools/amd_build/build_amd.py`; `python setup.py develop`

*   **Linux/macOS:**  `python setup.py develop` (or see notes for optional CMAKE variables)

*   **macOS:**  `python3 setup.py develop`

*   **Windows:** CPU and CUDA build instructions detailed in the original README, including environment variables.

##### Adjust Build Options (Optional)

Customize build configurations using `CMAKE_PREFIX_PATH` and the `CMAKE_ONLY` flag:
```bash
# On Linux/macOS:
export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"
CMAKE_ONLY=1 python setup.py build
ccmake build  # or cmake-gui build
```

### Docker Image

#### Using pre-built images

Quickly start with a pre-built image from Docker Hub:

```bash
docker run --gpus all --rm -ti --ipc=host pytorch/pytorch:latest
```

#### Building the image yourself

Build your own Docker image with CUDA 11.1 support:

```bash
make -f docker.Makefile
```

### Previous Versions

Find installation instructions and binaries for older versions on [our website](https://pytorch.org/get-started/previous-versions).

## Getting Started

Explore the following resources:

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

*   **Forums:**  [https://discuss.pytorch.org](https://discuss.pytorch.org)
*   **GitHub Issues:**  Bug reports, feature requests, installation issues, etc.
*   **Slack:**  [PyTorch Slack](https://pytorch.slack.com/) - Moderate to experienced users. Request an invite via [this form](https://goo.gl/forms/PP1AGvNHpSaJP8to1)
*   **Newsletter:**  [Sign-up here](https://eepurl.com/cbG0rv)
*   **Facebook Page:** [https://www.facebook.com/pytorch](https://www.facebook.com/pytorch)
*   For brand guidelines, please visit our website at [pytorch.org](https://pytorch.org/)

## Releases and Contributing

*   PyTorch typically has three minor releases per year.
*   Report bugs via [GitHub Issues](https://github.com/pytorch/pytorch/issues).
*   Contributions are welcome!  Follow the [Contribution page](CONTRIBUTING.md) and [Release page](RELEASE.md).

## The Team

PyTorch is a community-driven project with contributions from numerous engineers and researchers.  [See original README for list of contributors.]

## License

PyTorch is licensed under a BSD-style license, found in the [LICENSE](LICENSE) file.