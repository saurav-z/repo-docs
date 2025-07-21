![PyTorch Logo](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/pytorch-logo-dark.png)

# PyTorch: The Open Source Deep Learning Framework

**PyTorch is a leading open-source machine learning framework, providing a flexible and efficient platform for deep learning research and development.**  [Explore the PyTorch Repository](https://github.com/pytorch/pytorch)

**Key Features:**

*   **Tensor Computation with GPU Acceleration:** Enables fast numerical computation, similar to NumPy, with seamless GPU integration.
*   **Dynamic Neural Networks with Autograd:**  Build and modify neural networks with unparalleled flexibility using a tape-based automatic differentiation system.
*   **Python-First Development:** Integrates seamlessly with the Python ecosystem, allowing you to leverage your favorite libraries like NumPy and SciPy.
*   **Imperative Programming Style:** Offers an intuitive and easy-to-debug development experience with code that executes line by line.
*   **Fast and Lean:** Optimized for speed and memory efficiency, integrating acceleration libraries like Intel MKL and NVIDIA cuDNN for peak performance.
*   **Extensible Architecture:** Supports easy customization and extension through Python and C++ APIs, with minimal boilerplate.

## Overview

PyTorch is a powerful deep learning framework built to provide flexibility and speed. It's designed to be a great tool for both research and production environments.

### Core Components

PyTorch provides a comprehensive set of tools:

*   **torch:**  The core Tensor library, similar to NumPy, with GPU support. ([torch documentation](https://pytorch.org/docs/stable/torch.html))
*   **torch.autograd:** Automatic differentiation for building and training neural networks. ([torch.autograd documentation](https://pytorch.org/docs/stable/autograd.html))
*   **torch.jit:** Enables model serialization and optimization through TorchScript. ([torch.jit documentation](https://pytorch.org/docs/stable/jit.html))
*   **torch.nn:** The neural networks library, offering modular and flexible network building blocks. ([torch.nn documentation](https://pytorch.org/docs/stable/nn.html))
*   **torch.multiprocessing:** Offers Python multiprocessing with shared memory for tensors, facilitating efficient data loading and training. ([torch.multiprocessing documentation](https://pytorch.org/docs/stable/multiprocessing.html))
*   **torch.utils:** Provides data loading and utility functions. ([torch.utils documentation](https://pytorch.org/docs/stable/data.html))

## Installation

Find the optimal installation method for your needs at [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

### Binaries

Install pre-built packages using Conda or pip.

#### NVIDIA Jetson Platforms

Pre-built wheels are available for NVIDIA Jetson devices. ([Jetson PyTorch Installation](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048) / [L4T Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch))

### From Source

Build PyTorch from source for customization or the latest features.

#### Prerequisites

*   Python 3.9 or later
*   A C++17-compliant compiler (e.g., gcc 9.4.0 or newer on Linux, Visual Studio on Windows)

##### NVIDIA CUDA Support

*   Install the supported CUDA Toolkit and cuDNN versions. ([CUDA Downloads](https://developer.nvidia.com/cuda-downloads) / [cuDNN](https://developer.nvidia.com/cudnn) / [cuDNN Support Matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/support-matrix.html))
*   Set environment variables as needed.

##### AMD ROCm Support

*   Install AMD ROCm (version 4.0 or later).  ([AMD ROCm Installation](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html))
*   Configure environment variables for ROCm installation path.

##### Intel GPU Support
*   Follow the [PyTorch Prerequisites for Intel GPUs](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpus.html) instructions for Intel GPU support.

#### Installation Steps

1.  Get the PyTorch Source:

    ```bash
    git clone https://github.com/pytorch/pytorch
    cd pytorch
    git submodule sync
    git submodule update --init --recursive
    ```

2.  Install Dependencies:

    ```bash
    conda install cmake ninja  # or equivalent package manager
    pip install -r requirements.txt
    pip install mkl-static mkl-include #Linux/MacOS
    ```

    Additional OS-specific dependencies:
    * On Linux:  `pip install mkl-static mkl-include` and setup `magma` and  `triton` (optional)
    * On MacOS: `conda install pkg-config libuv`
    * On Windows: `conda install -c conda-forge libuv=1.39`

3.  Install PyTorch:

    *   **Linux/MacOS**:

        ```bash
        #For ROCm, run before install
        python tools/amd_build/build_amd.py (if using ROCm)
        export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}" #Linux/MacOS
        python -m pip install -r requirements-build.txt #Linux/MacOS
        python -m pip install --no-build-isolation -v -e .
        ```

    *   **Windows:**

        ```cmd
        python -m pip install --no-build-isolation -v -e .
        ```

    Adjust build options optionally, by using `ccmake` or `cmake-gui` to modify build parameters.

### Docker Image

*   Use pre-built images from Docker Hub.  ([Docker Hub PyTorch](https://hub.docker.com/r/pytorch/pytorch))
    ```bash
    docker run --gpus all --rm -ti --ipc=host pytorch/pytorch:latest
    ```
*   Build your own Docker image.  See the `docker.Makefile` for details.

### Building the Documentation

Build the PyTorch documentation using Sphinx. ([Sphinx](http://www.sphinx-doc.org))

```bash
cd docs/
pip install -r requirements.txt
make html
make serve
```

#### Building a PDF

1.  Make PDF:

    ```bash
    make latexpdf
    ```

2.  Enter the latex directory and build pdf:

    ```bash
    cd build/latex
    make LATEXOPTS="-interaction=nonstopmode"
    ```

## Getting Started

Explore the resources below:

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

*   Forums: [https://discuss.pytorch.org](https://discuss.pytorch.org)
*   GitHub Issues: Report bugs, request features, or ask questions.
*   Slack:  [PyTorch Slack](https://pytorch.slack.com/) (request an invite:  https://goo.gl/forms/PP1AGvNHpSaJP8to1)
*   Newsletter: [PyTorch Newsletter](https://eepurl.com/cbG0rv)
*   Facebook Page:  [PyTorch Facebook](https://www.facebook.com/pytorch)

## Releases and Contributing

Regular releases and contributions are welcome. Please adhere to the contribution guidelines.

*   [Contributing Guidelines](CONTRIBUTING.md)
*   [Release Information](RELEASE.md)

## The Team

PyTorch is a community-driven project, maintained by a dedicated team and supported by numerous contributors. ([Team Details](https://github.com/pytorch/pytorch#the-team))

## License

PyTorch is licensed under a BSD-style license.  ([LICENSE](LICENSE))