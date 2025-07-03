[![PyTorch Logo](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/pytorch-logo-dark.png)](https://github.com/pytorch/pytorch)

# PyTorch: Open Source Machine Learning Framework

**PyTorch is a leading open-source machine learning framework, empowering researchers and developers to build and deploy cutting-edge AI models.** Explore the official [PyTorch repository](https://github.com/pytorch/pytorch) for the latest updates and contributions.

**Key Features:**

*   **Tensor Computation with GPU Acceleration:** Provides a powerful tensor library akin to NumPy, optimized for high-performance computing on GPUs.
*   **Dynamic Neural Networks:** Enables flexible, tape-based autograd for building and modifying neural networks with ease.
*   **Python-First Design:** Deeply integrated with Python, allowing seamless use with your favorite Python libraries like NumPy, SciPy, and Cython.
*   **Imperative Programming:** Offers an intuitive and easy-to-debug imperative programming experience.
*   **Fast and Lean:** Leverages optimized libraries like Intel MKL, cuDNN, and NCCL for superior speed and efficiency.
*   **Extensible:** Provides a simple API for creating custom neural network modules and integrating with the PyTorch tensor API.

## Core Components

PyTorch is built on the following key components:

| Component                     | Description                                                                                                               |
| :---------------------------- | :------------------------------------------------------------------------------------------------------------------------ |
| [**torch**](https://pytorch.org/docs/stable/torch.html)        | A Tensor library like NumPy, with strong GPU support                                                                    |
| [**torch.autograd**](https://pytorch.org/docs/stable/autograd.html) | A tape-based automatic differentiation library that supports all differentiable Tensor operations in torch                                                              |
| [**torch.jit**](https://pytorch.org/docs/stable/jit.html)         | A compilation stack (TorchScript) to create serializable and optimizable models from PyTorch code                                                                      |
| [**torch.nn**](https://pytorch.org/docs/stable/nn.html)          | A neural networks library deeply integrated with autograd designed for maximum flexibility                                                                   |
| [**torch.multiprocessing**](https://pytorch.org/docs/stable/multiprocessing.html) | Python multiprocessing, but with magical memory sharing of torch Tensors across processes. Useful for data loading and Hogwild training                                                                       |
| [**torch.utils**](https://pytorch.org/docs/stable/data.html)       | DataLoader and other utility functions for convenience                                                                                  |

## Why Choose PyTorch?

PyTorch is an ideal choice for:

*   **GPU-Accelerated Computing:** Replace NumPy for faster tensor computations on GPUs.
*   **Flexible Deep Learning Research:** Create and experiment with neural networks with unmatched speed and flexibility.

### A GPU-Ready Tensor Library

PyTorch provides Tensors that can live either on the CPU or the GPU and accelerates the computation by a huge amount, offering a wide variety of tensor routines to accelerate and fit your scientific computation needs such as slicing, indexing, mathematical operations, linear algebra, reductions, and more!

### Dynamic Neural Networks: Tape-Based Autograd

PyTorch's unique reverse-mode auto-differentiation allows you to change the way your network behaves arbitrarily with zero lag or overhead, providing unmatched flexibility.

### Python-First Development

PyTorch is built to be deeply integrated into Python. You can write your new neural network layers in Python itself, using your favorite libraries and use packages such as Cython and Numba.

### Imperative Programming Experience

PyTorch is designed to be intuitive, linear in thought, and easy to use.

### Fast and Lean

PyTorch has minimal framework overhead and maximizes speed by integrating acceleration libraries such as Intel MKL, and NVIDIA (cuDNN, NCCL).

### Extensions Without Pain

Writing new neural network modules, or interfacing with PyTorch's Tensor API was designed to be straightforward
and with minimal abstractions.

## Installation

For detailed installation instructions, including binary installations via Conda or pip wheels, please visit: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

### Binaries

*   Commands to install binaries via Conda or pip wheels are on our website: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

#### NVIDIA Jetson Platforms

Python wheels for NVIDIA's Jetson Nano, Jetson TX1/TX2, Jetson Xavier NX/AGX, and Jetson AGX Orin are provided [here](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048) and the L4T container is published [here](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch)

They require JetPack 4.2 and above, and [@dusty-nv](https://github.com/dusty-nv) and [@ptrblck](https://github.com/ptrblck) are maintaining them.

### From Source

*   Follow the detailed instructions in the original README.
    *   Prerequisites
    *   Get the PyTorch Source
    *   Install Dependencies
    *   Install PyTorch

### Docker Image

*   Using pre-built images
*   Building the image yourself

### Building the Documentation

*   Building the Documentation
    *   Building a PDF

### Previous Versions

*   Installation instructions and binaries for previous PyTorch versions may be found
    on [our website](https://pytorch.org/get-started/previous-versions).

## Getting Started

*   [Tutorials](https://pytorch.org/tutorials/)
*   [Examples](https://github.com/pytorch/examples)
*   [The API Reference](https://pytorch.org/docs/)
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

*   Forums: Discuss implementations, research, etc. https://discuss.pytorch.org
*   GitHub Issues: Bug reports, feature requests, install issues, RFCs, thoughts, etc.
*   Slack: The [PyTorch Slack](https://pytorch.slack.com/) hosts a primary audience of moderate to experienced PyTorch users and developers for general chat, online discussions, collaboration, etc. If you are a beginner looking for help, the primary medium is [PyTorch Forums](https://discuss.pytorch.org). If you need a slack invite, please fill this form: https://goo.gl/forms/PP1AGvNHpSaJP8to1
*   Newsletter: No-noise, a one-way email newsletter with important announcements about PyTorch. You can sign-up here: https://eepurl.com/cbG0rv
*   Facebook Page: Important announcements about PyTorch. https://www.facebook.com/pytorch
*   For brand guidelines, please visit our website at [pytorch.org](https://pytorch.org/)

## Releases and Contributing

PyTorch typically releases three minor versions per year. Report bugs and contribute new features by following the guidelines outlined in [CONTRIBUTING.md](CONTRIBUTING.md) and [RELEASE.md](RELEASE.md).

## The Team

PyTorch is a community-driven project with many skillful engineers and researchers. The project is currently maintained by [Soumith Chintala](http://soumith.ch), [Gregory Chanan](https://github.com/gchanan), [Dmytro Dzhulgakov](https://github.com/dzhulgakov), [Edward Yang](https://github.com/ezyang), and [Nikita Shulga](https://github.com/malfet) with major contributions coming from hundreds of talented individuals in various forms and means.

## License

PyTorch is licensed under a BSD-style license, detailed in the [LICENSE](LICENSE) file.