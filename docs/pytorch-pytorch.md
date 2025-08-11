![PyTorch Logo](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/pytorch-logo-dark.png)

# PyTorch: Deep Learning and Tensor Computation with GPU Acceleration

**PyTorch empowers researchers and developers with a flexible, efficient, and Python-first platform for building and deploying cutting-edge machine learning models.**  ([Original Repo](https://github.com/pytorch/pytorch))

## Key Features

*   **Tensor Computation with GPU Acceleration:** Perform numerical computations similar to NumPy, with seamless GPU acceleration for significantly faster performance.
*   **Dynamic Neural Networks with Autograd:** Build and modify neural networks dynamically, offering unparalleled flexibility for research and experimentation.
*   **Python-First Design:** Leverage the power of Python with deep integration, allowing you to use familiar libraries and tools.
*   **Imperative Programming:** Experience intuitive and easy-to-debug code execution, with clear stack traces and straightforward error handling.
*   **Fast and Lean:** Benefit from minimal framework overhead and optimized integrations with acceleration libraries like Intel MKL, cuDNN, and NCCL.
*   **Extensible:** Easily create custom neural network modules and extend PyTorch's functionality using Python, C/C++, and other familiar tools.

## Comprehensive Overview

PyTorch offers a robust and versatile platform, composed of essential components for deep learning:

| Component                 | Description                                                                                                          |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **torch**                 | A Tensor library (similar to NumPy) with robust GPU support.                                                         |
| **torch.autograd**        | A tape-based automatic differentiation engine, enabling dynamic neural networks.                                      |
| **torch.jit**             | Compilation stack (TorchScript) for serializing and optimizing PyTorch code.                                       |
| **torch.nn**              | A dedicated neural networks library tightly integrated with autograd for maximum flexibility.                        |
| **torch.multiprocessing** | Enables Python multiprocessing with memory sharing of torch Tensors for optimized data loading and training.          |
| **torch.utils**           | Contains `DataLoader` and other utility functions to facilitate the development process.                         |

### GPU-Ready Tensor Library

Utilize tensors (similar to NumPy's ndarray) that can reside on either CPU or GPU. PyTorch's optimized routines accelerate calculations such as slicing, indexing, mathematical operations, linear algebra, and reductions.

### Dynamic Neural Networks: Tape-Based Autograd

PyTorch uses a technique called reverse-mode auto-differentiation, granting the flexibility to change network behavior without delay. This is one of the fastest implementations, offering great speed and adaptability for research.

![Dynamic graph](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/dynamic_graph.gif)

### Python First

PyTorch is designed to seamlessly integrate with Python, enabling you to use it just like NumPy, SciPy, or scikit-learn. You can create custom neural network layers and leverage your favorite Python packages like Cython and Numba, without reinventing the wheel.

### Imperative Experiences

PyTorch is designed for a linear and intuitive programming experience. Execution is straightforward, error messages and debugging are easy to understand, and stack traces accurately pinpoint the source of your code.

### Fast and Lean

Leverage acceleration libraries such as Intel MKL, NVIDIA cuDNN, and NCCL to maximize speed. The CPU and GPU Tensor and neural network backends are mature and extensively tested, resulting in high performance for both small and large networks.

### Extensions Without Pain

PyTorch makes writing new neural network modules and interfacing with its Tensor API straightforward. Use the torch API within Python, integrate NumPy-based libraries, or create custom layers in C/C++ with the convenient extension API.

## Installation

Find detailed installation instructions, including binary and source options, on our website: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

*   [Binaries](https://pytorch.org/get-started/locally/)
*   [From Source](https://pytorch.org/get-started/locally/)
*   [Docker Image](https://pytorch.org/get-started/locally/)

## Getting Started

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

*   Forums: Discuss implementations, research, etc. https://discuss.pytorch.org
*   GitHub Issues: Bug reports, feature requests, install issues, RFCs, thoughts, etc.
*   Slack: The [PyTorch Slack](https://pytorch.slack.com/) hosts a primary audience of moderate to experienced PyTorch users and developers for general chat, online discussions, collaboration, etc. If you are a beginner looking for help, the primary medium is [PyTorch Forums](https://discuss.pytorch.org). If you need a slack invite, please fill this form: https://goo.gl/forms/PP1AGvNHpSaJP8to1
*   Newsletter: No-noise, a one-way email newsletter with important announcements about PyTorch. You can sign-up here: https://eepurl.com/cbG0rv
*   Facebook Page: Important announcements about PyTorch. https://www.facebook.com/pytorch
*   For brand guidelines, please visit our website at [pytorch.org](https://pytorch.org/)

## Releases and Contributing

PyTorch releases typically occur three times a year. Report bugs via [filing an issue](https://github.com/pytorch/pytorch/issues).  Contribute new features and fixes by first opening an issue to discuss them with the team. See our [Contribution page](CONTRIBUTING.md) and [Release page](RELEASE.md) for more information.

## The Team

PyTorch is a community-driven project, maintained by skilled engineers and researchers.  The current maintainers and major contributors are listed.

## License

PyTorch is licensed under a BSD-style license (see [LICENSE](LICENSE)).