[![PyTorch Logo](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/pytorch-logo-dark.png)](https://github.com/pytorch/pytorch)

# PyTorch: Deep Learning with Python

**PyTorch is an open-source machine learning framework that accelerates the path from research prototyping to production deployment.**

## Key Features

*   **GPU-Accelerated Tensor Computation:** Offers powerful tensor operations similar to NumPy, with seamless GPU acceleration for faster computations.
*   **Dynamic Neural Networks:** Enables the creation of flexible neural networks using a tape-based autograd system, allowing for dynamic and adaptable models.
*   **Python-First Design:** Deeply integrated with Python, providing an intuitive and familiar development experience with easy access to Python libraries like NumPy, SciPy, and Cython.
*   **Imperative Programming:** Designed for an imperative experience, which means code is executed line by line, making debugging and understanding straightforward.
*   **Fast and Lean:** Optimized for speed and efficiency, integrating with acceleration libraries such as Intel MKL, cuDNN, and NCCL to maximize performance, and includes custom memory allocators for memory efficiency.
*   **Extensible Architecture:** Allows for easy extension through custom modules written in Python or C/C++, minimizing abstraction and boilerplate.

## Installation

Detailed installation instructions, including binary installations via Conda or pip wheels, are available at [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/).

### Binaries

*   **Conda:** Installation via Conda is recommended. Follow the instructions at [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/).
*   **Pip:**  PyTorch wheels are available via pip. Instructions can be found at [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/).
*   **NVIDIA Jetson Platforms:** Pre-built wheels are available for NVIDIA Jetson platforms, see [here](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048) and [here](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch)

### From Source

*   **Prerequisites:** Requires Python 3.9 or later, a C++17-compliant compiler (e.g., GCC 9.4.0+), and (optionally) CUDA, ROCm, or Intel GPU support.
*   **Steps:** Clone the repository, install dependencies, and build the package.  See the original README for detailed instructions.

### Docker Image

*   **Pre-built Images:**  Pull pre-built images from Docker Hub using commands provided in the original README.
*   **Build Your Own:**  Build a Docker image with CUDA 11.1 support and cuDNN v8 using the provided Dockerfile.

### Documentation

*   **Building the Documentation:** Follow the instructions to install necessary packages and build documentation in HTML and other formats.
*   **Building a PDF:** Generate a PDF version of the documentation using `make latexpdf` and related commands.

## Getting Started

Explore the following resources to quickly start with PyTorch:

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

*   **Forums:** Discuss implementations, research, etc. [https://discuss.pytorch.org](https://discuss.pytorch.org)
*   **GitHub Issues:** Bug reports, feature requests, install issues, RFCs, thoughts, etc.
*   **Slack:** [PyTorch Slack](https://pytorch.slack.com/) for general chat, online discussions, collaboration, etc. (see original README for invite link)
*   **Newsletter:** Sign-up for the no-noise, one-way email newsletter: [https://eepurl.com/cbG0rv](https://eepurl.com/cbG0rv)
*   **Facebook Page:** Important announcements about PyTorch. [https://www.facebook.com/pytorch](https://www.facebook.com/pytorch)
*   **Brand Guidelines:** [pytorch.org](https://pytorch.org/)

## Releases and Contributing

PyTorch typically releases three minor versions per year.  Report bugs through GitHub Issues.  Contributions are welcome; review the [Contribution page](CONTRIBUTING.md) and the [Release page](RELEASE.md) for more details.

## The Team

PyTorch is a community-driven project maintained by a core team and supported by a vast network of contributors.  See the original README for the current team.

## License

PyTorch is licensed under a BSD-style license, as found in the [LICENSE](LICENSE) file.

**[Original Repository](https://github.com/pytorch/pytorch)**