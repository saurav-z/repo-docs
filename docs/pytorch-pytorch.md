![PyTorch Logo](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/pytorch-logo-dark.png)

# PyTorch: Deep Learning with GPU Acceleration

**PyTorch is an open-source machine learning framework that accelerates the path from research prototyping to production deployment.** ([Original Repo](https://github.com/pytorch/pytorch))

## Key Features

*   **GPU-Accelerated Tensor Computations:**  Leverage the power of GPUs for fast, efficient numerical computations, similar to NumPy but optimized for deep learning.
*   **Dynamic Neural Networks:**  Build and modify neural networks with unparalleled flexibility using a tape-based autograd system for dynamic computation graphs.
*   **Python-First Design:**  Seamlessly integrate with the Python ecosystem, utilizing familiar libraries like NumPy, SciPy, and Cython.
*   **Imperative Programming:**  Benefit from an intuitive and easy-to-debug imperative programming style.
*   **Fast and Lean:**  Maximize performance with minimal overhead, integrating optimized libraries like Intel MKL, cuDNN, and NCCL.
*   **Extensible Architecture:**  Easily extend PyTorch with custom modules and integrate with existing codebases through a flexible API.

## Getting Started

*   **Tutorials:** [https://pytorch.org/tutorials/](https://pytorch.org/tutorials/)
*   **Examples:** [https://github.com/pytorch/examples](https://github.com/pytorch/examples)
*   **API Reference:** [https://pytorch.org/docs/](https://pytorch.org/docs/)
*   **Glossary:** [https://github.com/pytorch/pytorch/blob/main/GLOSSARY.md](https://github.com/pytorch/pytorch/blob/main/GLOSSARY.md)

## What is PyTorch?

PyTorch is a versatile and powerful deep learning framework offering both tensor computations and a dynamic neural network library. Its core components provide the building blocks for creating and training sophisticated machine learning models.

### Key Components

*   **torch:**  A Tensor library like NumPy, with strong GPU support
*   **torch.autograd:**  A tape-based automatic differentiation library.
*   **torch.jit:**  A compilation stack (TorchScript) for serializable and optimizable models.
*   **torch.nn:** A neural networks library.
*   **torch.multiprocessing:** Python multiprocessing with memory sharing of torch Tensors.
*   **torch.utils:** DataLoader and utility functions.

### More About PyTorch

*   **A GPU-Ready Tensor Library:** Experience GPU-accelerated tensor operations for significant performance gains in your scientific computing tasks.
    ![Tensor illustration](./docs/source/_static/img/tensor_illustration.png)

*   **Dynamic Neural Networks: Tape-Based Autograd:**  Benefit from reverse-mode auto-differentiation, allowing for flexible network architectures and zero-overhead dynamic behavior.
    ![Dynamic graph](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/dynamic_graph.gif)

*   **Python First:** Leverage PyTorch's deep integration with Python, allowing you to leverage all of your favorite Python libraries and tools.

*   **Imperative Experiences:**  PyTorch is designed to be intuitive, easy to debug, and understand.

*   **Fast and Lean:**  PyTorch has minimal framework overhead, and integrates with acceleration libraries such as Intel MKL, cuDNN, and NCCL, to maximize speed.

*   **Extensions Without Pain:** You can write new neural network layers in Python using the torch API or your favorite NumPy-based libraries such as SciPy.

## Installation

Installation methods via Conda or pip wheels are available at [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

### Binaries

Instructions for installing binaries are found on the official PyTorch website.

#### NVIDIA Jetson Platforms

Pre-built wheels for NVIDIA Jetson platforms are available [here](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048) and the L4T container is published [here](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch).

### From Source

#### Prerequisites

Ensure you meet these prerequisites when installing from source:

*   Python 3.9 or later
*   A C++17 compatible compiler (e.g., gcc 9.4.0+)
*   Visual Studio or Visual Studio Build Tool (Windows)

**Optional Dependencies (Based on your build choices):**

*   **NVIDIA CUDA Support:** Requires CUDA, cuDNN v8.5+, and a compatible compiler.
*   **AMD ROCm Support:** Requires AMD ROCm 4.0+ on Linux systems.
*   **Intel GPU Support:** Follow Intel's instructions for building with Intel GPUs.

#### Build Process

1.  **Get the Source:**
    ```bash
    git clone https://github.com/pytorch/pytorch
    cd pytorch
    git submodule sync
    git submodule update --init --recursive
    ```

2.  **Install Dependencies:**
    ```bash
    conda install cmake ninja  # or `pip install cmake ninja`
    pip install -r requirements.txt
    ```
    Install CUDA and AMD ROCm dependencies as described in the original README, and consider the following:

    *   On Linux, consider `pip install mkl-static mkl-include`.
    *   CUDA specific instructions are described in the original README.
    *   AMD ROCm specific instructions are described in the original README.

3.  **Install PyTorch:**

    *   If building with ROCm:
        ```bash
        python tools/amd_build/build_amd.py
        ```
    *   Regardless of the above, continue with
        ```bash
        export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"
        python -m pip install --no-build-isolation -v -e .
        ```

### Docker Image

Leverage pre-built Docker images for convenient deployment or build your own.

#### Using pre-built images

```bash
docker run --gpus all --rm -ti --ipc=host pytorch/pytorch:latest
```

#### Building the image yourself

1.  Use the provided `Dockerfile`.
2.  Specify the desired Python version with `PYTHON_VERSION=x.y`.
3.  Set additional CMake variables via `CMAKE_VARS="..."`.
4.  Build: `make -f docker.Makefile`.

### Building the Documentation

Ensure you have Sphinx and the pytorch\_sphinx\_theme2 installed. Install `torch` first.

```bash
cd docs/
pip install -r requirements.txt
make html
make serve
```

### Previous Versions

Find older releases and binaries on [our website](https://pytorch.org/get-started/previous-versions).

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

*   **Forums:** https://discuss.pytorch.org
*   **GitHub Issues:** Report bugs, request features.
*   **Slack:** [PyTorch Slack](https://pytorch.slack.com/) (request invite via form).
*   **Newsletter:** https://eepurl.com/cbG0rv
*   **Facebook:** https://www.facebook.com/pytorch
*   **Brand Guidelines:** [pytorch.org](https://pytorch.org/)

## Releases and Contributing

PyTorch typically releases three minor versions per year. Report bugs via [GitHub Issues](https://github.com/pytorch/pytorch/issues).  Contribute bug fixes directly.  For new features, open an issue for discussion.  See the [Contribution page](CONTRIBUTING.md) and [Release page](RELEASE.md).

## The Team

PyTorch is a community-driven project, maintained by a core team with contributions from a large community.

## License

PyTorch is licensed under a BSD-style license, as found in the [LICENSE](LICENSE) file.