[![PyTorch Logo](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/pytorch-logo-dark.png)](https://github.com/pytorch/pytorch)

# PyTorch: The Deep Learning Framework of Choice for Research and Production

**PyTorch is a powerful and flexible open-source deep learning framework that accelerates the journey from research prototyping to production deployment.** Dive into the cutting-edge with PyTorch, the Python-first framework that empowers developers and researchers alike. Find the original repo [here](https://github.com/pytorch/pytorch).

**Key Features:**

*   **Tensor Computation with GPU Acceleration:** Leverage the power of GPUs for lightning-fast tensor computations, similar to NumPy.
*   **Dynamic Neural Networks with Autograd:** Build and modify neural networks with unparalleled flexibility using PyTorch's tape-based autograd system.
*   **Python-First Design:** Seamlessly integrate with your existing Python workflows and leverage the vast ecosystem of Python libraries.
*   **Imperative Programming:** Experience intuitive and easy-to-debug code execution with PyTorch's imperative style.
*   **Fast and Lean:** Benefit from minimal framework overhead and optimized performance with integrations like Intel MKL, cuDNN, and NCCL.
*   **Easy Extensions:** Easily write custom neural network modules or interface with the PyTorch Tensor API.

## Core Components

| Component                      | Description                                                                                               |
| :----------------------------- | :-------------------------------------------------------------------------------------------------------- |
| [**torch**](https://pytorch.org/docs/stable/torch.html)             | Tensor library with GPU support                                             |
| [**torch.autograd**](https://pytorch.org/docs/stable/autograd.html)      | Automatic differentiation library                                           |
| [**torch.jit**](https://pytorch.org/docs/stable/jit.html)           | Compilation stack (TorchScript) for serializable and optimizable models     |
| [**torch.nn**](https://pytorch.org/docs/stable/nn.html)            | Neural networks library with autograd integration                            |
| [**torch.multiprocessing**](https://pytorch.org/docs/stable/multiprocessing.html) | Python multiprocessing with tensor memory sharing                             |
| [**torch.utils**](https://pytorch.org/docs/stable/data.html)          | Data loading and utility functions                                          |

## Installation

Install PyTorch tailored to your needs:

*   [Installation Instructions](https://pytorch.org/get-started/locally/)

### Installation Methods:

*   **Binaries:**
    *   [Conda](https://docs.conda.io/en/latest/)
    *   [Pip](https://pip.pypa.io/en/stable/)
    *   [Jetson Nano, TX1/TX2, Xavier NX/AGX, and AGX Orin](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048)
*   **From Source:**
    *   [Prerequisites](#prerequisites)
        *   NVIDIA CUDA Support
        *   AMD ROCm Support
        *   Intel GPU Support
    *   Get the PyTorch Source
    *   Install Dependencies
    *   Install PyTorch
        *   Adjust Build Options (Optional)
*   **Docker Image:**
    *   Using pre-built images
    *   Building the image yourself
*   Building the Documentation
    *   Building a PDF
*   Previous Versions

### Prerequisites

If installing from source, you'll need:

*   Python 3.9 or later
*   A compiler with C++17 support (e.g., clang or gcc 9.4.0+)
*   Visual Studio or Visual Studio Build Tool (Windows only)

### [NVIDIA CUDA Support](#nvidia-cuda-support)

*   NVIDIA CUDA
*   NVIDIA cuDNN v8.5 or above

### [AMD ROCm Support](#amd-rocm-support)

*   AMD ROCm 4.0 and above installation

### [Intel GPU Support](#intel-gpu-support)

*   PyTorch Prerequisites for Intel GPUs

#### Get the PyTorch Source

```bash
git clone https://github.com/pytorch/pytorch
cd pytorch
git submodule sync
git submodule update --init --recursive
```

#### Install Dependencies

*   Common dependencies and Linux-specific dependencies, macOS-specific dependencies, and Windows-specific dependencies.
  *   See the [README](https://github.com/pytorch/pytorch#install-dependencies) for installation commands.

#### Install PyTorch

*   Installation steps for Linux, macOS, Windows, CPU-only builds, CUDA-based builds, and Intel GPU builds.
  *   See the [README](https://github.com/pytorch/pytorch#install-pytorch) for installation commands.

#### [Docker Image](#docker-image)

*   Run the pre-built image with the command: `docker run --gpus all --rm -ti --ipc=host pytorch/pytorch:latest`
*   Build the Docker image using the `docker.Makefile` file.

## Getting Started

Unlock your potential with these valuable resources:

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

*   Forums: [Discuss implementations, research, etc.](https://discuss.pytorch.org/)
*   GitHub Issues: Bug reports, feature requests, install issues, RFCs, thoughts, etc.
*   Slack: [PyTorch Slack](https://pytorch.slack.com/) hosts a primary audience of moderate to experienced PyTorch users and developers
*   Newsletter: [Sign-up](https://eepurl.com/cbG0rv) for announcements from PyTorch.
*   Facebook Page: [PyTorch Facebook](https://www.facebook.com/pytorch)
*   [Brand Guidelines](https://pytorch.org/)

## Releases and Contributing

*   Minor releases are made 3 times a year.
*   File an [issue](https://github.com/pytorch/pytorch/issues) to report bugs.
*   Contribute bug-fixes without further discussion.
*   Discuss new features before sending a PR.
*   Learn about contributions on the [Contribution page](CONTRIBUTING.md).
*   Learn about PyTorch releases on the [Release page](RELEASE.md).

## The Team

PyTorch is a community-driven project. The current maintainers include: [Soumith Chintala](http://soumith.ch), [Gregory Chanan](https://github.com/gchanan), [Dmytro Dzhulgakov](https://github.com/dzhulgakov), [Edward Yang](https://github.com/ezyang), and [Nikita Shulga](https://github.com/malfet), with significant contributions from many others.

## License

PyTorch is licensed under a BSD-style license, as found in the [LICENSE](LICENSE) file.