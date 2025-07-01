![PyTorch Logo](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/pytorch-logo-dark.png)

# PyTorch: Your Path to Flexible and Efficient Deep Learning

**PyTorch is a leading open-source machine learning framework, providing a seamless bridge between research and production with its Python-first approach.**  ([Original Repo](https://github.com/pytorch/pytorch))

## Key Features

*   **Tensor Computation with GPU Acceleration:** Leverages GPUs for fast tensor operations, similar to NumPy.
*   **Dynamic Neural Networks:** Built on a tape-based autograd system for maximum flexibility in network design.
*   **Python-First Design:** Deeply integrated with Python, allowing for natural use of existing Python packages (NumPy, SciPy, etc.) and custom layers.
*   **Imperative Style:** Intuitive and easy-to-debug code execution with clear stack traces.
*   **Fast and Lean:** Optimized for speed with minimal framework overhead, integrating acceleration libraries like Intel MKL, cuDNN, and NCCL.
*   **Easy Extension:** Simple API for writing custom neural network modules in Python or C/C++.

## Installation

Find the installation instructions on the official PyTorch website:  [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

### Binaries

*   **Conda:** Use Conda for straightforward installation.
*   **Pip Wheels:** Easily install with pre-built wheels.
*   **NVIDIA Jetson Platforms:**  Pre-built wheels are available for Jetson Nano, TX1/TX2, Xavier NX/AGX, and AGX Orin. See: [Jetson PyTorch](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048).
*   **Previous Versions:** Installation instructions for older releases are available on the [website](https://pytorch.org/get-started/previous-versions).

### From Source

Building from source provides greater control and customization.

#### Prerequisites

*   Python 3.9 or later
*   A C++17-compliant compiler (e.g., GCC 9.4.0+)
*   Visual Studio or Visual Studio Build Tool (Windows)

#### CUDA Support

*   Install a supported CUDA version (see [support matrix](https://pytorch.org/get-started/locally/)) and cuDNN v8.5+.
*   Set environment variables as needed.

#### ROCm Support

*   Install AMD ROCm 4.0 or later.
*   Set `ROCM_PATH` if ROCm is not in the default location.
*   Set `PYTORCH_ROCM_ARCH` for specific GPU architecture.

#### Intel GPU Support

*   Follow the [Intel GPU prerequisites](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpus.html) instructions.

#### Steps

1.  Get the PyTorch source:
    ```bash
    git clone https://github.com/pytorch/pytorch
    cd pytorch
    git submodule sync
    git submodule update --init --recursive
    ```
2.  Install dependencies:
    ```bash
    conda install cmake ninja  # or pip install
    pip install -r requirements.txt
    ```
3.  Install PyTorch:
    ```bash
    export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"  # Linux & macOS (optional)
    python setup.py develop
    ```

### Docker Image

*   **Pre-built Images:** Run pre-built images from Docker Hub:
    ```bash
    docker run --gpus all --rm -ti --ipc=host pytorch/pytorch:latest
    ```
*   **Build your own Image:**  A Dockerfile is provided for custom builds with CUDA 11.1 support and cuDNN v8.

## Getting Started

*   [Tutorials](https://pytorch.org/tutorials/): Comprehensive tutorials for learning PyTorch.
*   [Examples](https://github.com/pytorch/examples): Practical PyTorch code examples.
*   [API Reference](https://pytorch.org/docs/): Detailed API documentation.
*   [Glossary](https://github.com/pytorch/pytorch/blob/main/GLOSSARY.md): Key terms and concepts.

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

*   **Forums:** Discuss implementations and research:  [https://discuss.pytorch.org](https://discuss.pytorch.org)
*   **GitHub Issues:** Report bugs, request features, and discuss installation issues.
*   **Slack:** Join the PyTorch Slack for general chat, online discussions, and collaboration (form for invitation: https://goo.gl/forms/PP1AGvNHpSaJP8to1)
*   **Newsletter:**  Get important PyTorch announcements (sign-up: https://eepurl.com/cbG0rv)
*   **Facebook Page:** Stay updated on the PyTorch Facebook page: https://www.facebook.com/pytorch

## Releases and Contributing

*   Typically, PyTorch has three minor releases a year.
*   Report bugs by [filing an issue](https://github.com/pytorch/pytorch/issues).
*   Contribute by submitting bug fixes without prior discussion.
*   Discuss new features beforehand by opening an issue.
*   Learn more about contributions on the [Contribution page](CONTRIBUTING.md) and about PyTorch releases, see [Release page](RELEASE.md).

## The Team

PyTorch is a community-driven project led by a dedicated team.  Maintained by [Soumith Chintala](http://soumith.ch), [Gregory Chanan](https://github.com/gchanan), [Dmytro Dzhulgakov](https://github.com/dzhulgakov), [Edward Yang](https://github.com/ezyang), and [Nikita Shulga](https://github.com/malfet) with significant contributions from a vast community.

## License

PyTorch is licensed under a BSD-style license; see the [LICENSE](LICENSE) file.