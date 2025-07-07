[![PyTorch Logo](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/pytorch-logo-dark.png)](https://github.com/pytorch/pytorch)

# PyTorch: The Open Source Deep Learning Platform

**PyTorch is a powerful and flexible open-source machine learning framework, empowering researchers and developers to build and deploy cutting-edge AI models.**  Get started today at the [original PyTorch repository](https://github.com/pytorch/pytorch).

**Key Features:**

*   **Tensor Computation with GPU Acceleration:** Leverage the power of GPUs for blazing-fast tensor operations, similar to NumPy but optimized for performance.
*   **Dynamic Neural Networks with Autograd:** Build and modify neural networks with unparalleled flexibility using a tape-based autograd system for automatic differentiation.
*   **Python-First Approach:** Seamlessly integrate with your existing Python workflows, libraries, and tools, including NumPy, SciPy, and Cython.
*   **Imperative Programming Style:**  Enjoy an intuitive and easy-to-debug programming experience.
*   **Fast and Lean Design:** Benefit from minimal framework overhead and integration with optimized acceleration libraries like Intel MKL, cuDNN, and NCCL.
*   **Extensible and Customizable:** Easily write custom neural network modules and extend PyTorch's functionality without complex abstractions.

## Core Components

PyTorch provides a modular architecture comprising the following key components:

*   **torch:** A Tensor library with NumPy-like functionality, featuring robust GPU support. ([torch documentation](https://pytorch.org/docs/stable/torch.html))
*   **torch.autograd:**  A tape-based automatic differentiation engine, supporting all differentiable Tensor operations. ([torch.autograd documentation](https://pytorch.org/docs/stable/autograd.html))
*   **torch.jit:**  A compilation stack (TorchScript) enabling serialization and optimization of PyTorch code. ([torch.jit documentation](https://pytorch.org/docs/stable/jit.html))
*   **torch.nn:** A neural network library deeply integrated with autograd, providing maximum flexibility. ([torch.nn documentation](https://pytorch.org/docs/stable/nn.html))
*   **torch.multiprocessing:** Python multiprocessing with magical memory sharing of torch Tensors across processes. Useful for data loading and Hogwild training. ([torch.multiprocessing documentation](https://pytorch.org/docs/stable/multiprocessing.html))
*   **torch.utils:** Data loading and utility functions for convenient data handling. ([torch.utils documentation](https://pytorch.org/docs/stable/data.html))

## Installation

Find the most up-to-date and recommended installation methods on the [PyTorch website](https://pytorch.org/get-started/locally/).

### Binaries

Install pre-built binaries using Conda or pip wheels.  Consult the website for specific installation commands.

*   **NVIDIA Jetson Platforms:** Pre-built wheels are available for Jetson Nano, TX1/TX2, Xavier NX/AGX, and AGX Orin.  Find them [here](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048) and the L4T container [here](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch).

### From Source

#### Prerequisites

*   Python 3.9 or later
*   A C++17-compatible compiler (e.g., gcc 9.4.0+ or clang)
*   Visual Studio or Visual Studio Build Tool (Windows)

    **NVIDIA CUDA Support:** Install CUDA and cuDNN, and a compatible compiler.
    **AMD ROCm Support:** Install AMD ROCm.
    **Intel GPU Support:** Follow the [Intel GPU prerequisites](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpus.html).

#### Steps

1.  **Get the Source:**
    ```bash
    git clone https://github.com/pytorch/pytorch
    cd pytorch
    git submodule sync
    git submodule update --init --recursive
    ```

2.  **Install Dependencies:**
    ```bash
    conda install cmake ninja  # or pip install cmake ninja
    pip install -r requirements.txt
    ```
3.  **Install PyTorch:** (After setting up the specific dependencies for your system.)  See the original README for specific instructions based on OS and GPU support.

### Docker Image

*   **Pre-built Images:** Use pre-built images from Docker Hub: `docker run --gpus all --rm -ti --ipc=host pytorch/pytorch:latest`
*   **Build from Source:** Build a custom Docker image.  See `docker.Makefile` in the repository.

### Building the Documentation

*   Requires Sphinx and pytorch_sphinx_theme2.
*   `cd docs/`
*   `pip install -r requirements.txt`
*   `make html` (for HTML)
*   `make latexpdf` and `make LATEXOPTS="-interaction=nonstopmode"` (for PDF)

## Getting Started

Explore the wealth of resources to quickly start learning and using PyTorch:

*   **Tutorials:**  [https://pytorch.org/tutorials/](https://pytorch.org/tutorials/)
*   **Examples:** [https://github.com/pytorch/examples](https://github.com/pytorch/examples)
*   **API Reference:** [https://pytorch.org/docs/](https://pytorch.org/docs/)
*   **Glossary:** [https://github.com/pytorch/pytorch/blob/main/GLOSSARY.md](https://github.com/pytorch/pytorch/blob/main/GLOSSARY.md)

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

*   **Forums:**  Discuss implementations and research: [https://discuss.pytorch.org](https://discuss.pytorch.org)
*   **GitHub Issues:** Report bugs, request features, and discuss installations.
*   **Slack:**  [PyTorch Slack](https://pytorch.slack.com/)  (request invite via form)
*   **Newsletter:**  Subscribe for announcements:  [https://eepurl.com/cbG0rv](https://eepurl.com/cbG0rv)
*   **Facebook:** [https://www.facebook.com/pytorch](https://www.facebook.com/pytorch)

## Releases and Contributing

PyTorch typically releases minor versions three times a year.  Report bugs via [GitHub issues](https://github.com/pytorch/pytorch/issues).  Contributions are welcome; please see the [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. For release information see [RELEASE.md](RELEASE.md).

## The Team

PyTorch is a community-driven project maintained by a team of engineers and researchers.  For a list of contributors, see the original README.

## License

PyTorch is licensed under a BSD-style license; see the [LICENSE](LICENSE) file.