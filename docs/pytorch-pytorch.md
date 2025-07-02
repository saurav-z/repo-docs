[![PyTorch Logo](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/pytorch-logo-dark.png)](https://github.com/pytorch/pytorch)

# PyTorch: The Deep Learning Framework for Research and Production

PyTorch is a leading open-source machine learning framework that accelerates the path from research prototyping to production deployment, providing flexible and efficient tools for building and training deep learning models. **[Explore the full PyTorch repository on GitHub](https://github.com/pytorch/pytorch)!**

**Key Features:**

*   **Tensor Computation with GPU Acceleration:** Offers tensor operations similar to NumPy, with seamless support for GPUs, enabling significant performance gains for computationally intensive tasks.
*   **Dynamic Neural Networks with Autograd:** Supports dynamic computation graphs and reverse-mode automatic differentiation (autograd) for unparalleled flexibility in model design and modification.
*   **Python-First Development:** Fully integrated with Python, allowing for intuitive and familiar coding, leveraging existing Python libraries like NumPy, SciPy, and Cython.
*   **Imperative Design:** Provides an imperative and intuitive coding experience, allowing for easier debugging and understanding of code execution.
*   **Fast and Lean:** Optimized for speed and memory efficiency, with integration of acceleration libraries (e.g., Intel MKL, cuDNN, NCCL) and custom GPU memory allocators.
*   **Extensible Architecture:** Offers straightforward APIs for writing custom neural network modules and extending the framework using Python or C++.

**Why Choose PyTorch?**

*   **Flexibility:** Easily adapt and modify models for research and experimentation.
*   **Speed:** Benefit from optimized performance on both CPUs and GPUs.
*   **Ease of Use:** Intuitive Python integration simplifies development and debugging.
*   **Community:** Leverage a large and active community of developers, researchers, and users.

## Core Components

PyTorch is built upon several core components that are interconnected and work together to provide powerful machine learning capabilities:

| Component | Description                                                                                               |
| :-------- | :-------------------------------------------------------------------------------------------------------- |
| [**torch**](https://pytorch.org/docs/stable/torch.html)          | Tensor library with GPU support (similar to NumPy).                                         |
| [**torch.autograd**](https://pytorch.org/docs/stable/autograd.html) | Automatic differentiation engine for all differentiable Tensor operations.                     |
| [**torch.jit**](https://pytorch.org/docs/stable/jit.html)         | Compilation stack (TorchScript) for serializing and optimizing PyTorch models.           |
| [**torch.nn**](https://pytorch.org/docs/stable/nn.html)          | Neural networks library with deep integration with autograd for model flexibility.        |
| [**torch.multiprocessing**](https://pytorch.org/docs/stable/multiprocessing.html) | Python multiprocessing with shared memory for efficient data loading and training.               |
| [**torch.utils**](https://pytorch.org/docs/stable/data.html)     | Data loading utilities, including `DataLoader`, for efficient data handling.              |

## Installation

Find the latest installation instructions on the official PyTorch website.

*   [Installation Instructions](https://pytorch.org/get-started/locally/)

### Installation Methods

Choose the method that best suits your needs:

*   **Binaries:** Install pre-built packages using Conda or pip.
*   **From Source:** Build from source for customization and optimization.
*   **Docker Image:** Use pre-built or custom Docker images for a containerized environment.

### Installation Guides

*   **NVIDIA Jetson Platforms:** Installation guides and pre-built wheels for NVIDIA Jetson platforms are available [here](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048).
*   **Build from Source:** Detailed instructions for building PyTorch from source are provided within the main README.

## Getting Started

*   **Tutorials:** ([https://pytorch.org/tutorials/](https://pytorch.org/tutorials/))
*   **Examples:** ([https://github.com/pytorch/examples](https://github.com/pytorch/examples))
*   **API Reference:** ([https://pytorch.org/docs/](https://pytorch.org/docs/))
*   **Glossary:** ([https://github.com/pytorch/pytorch/blob/main/GLOSSARY.md](https://github.com/pytorch/pytorch/blob/main/GLOSSARY.md))

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

*   **Forums:** Discuss implementations, research, etc. - [https://discuss.pytorch.org](https://discuss.pytorch.org)
*   **GitHub Issues:** Bug reports, feature requests, and more.
*   **Slack:** The [PyTorch Slack](https://pytorch.slack.com/) hosts a primary audience of moderate to experienced PyTorch users and developers for general chat, online discussions, collaboration, etc. If you are a beginner looking for help, the primary medium is [PyTorch Forums](https://discuss.pytorch.org). If you need a slack invite, please fill this form: https://goo.gl/forms/PP1AGvNHpSaJP8to1
*   **Newsletter:** No-noise, a one-way email newsletter with important announcements about PyTorch. You can sign-up here: https://eepurl.com/cbG0rv
*   **Facebook Page:** Important announcements about PyTorch. https://www.facebook.com/pytorch
*   For brand guidelines, please visit our website at [pytorch.org](https://pytorch.org/)

## Releases and Contributing

PyTorch typically releases new versions with bug fixes and new features three times a year.  Contributions are welcome!

*   Please report any bugs by [filing an issue](https://github.com/pytorch/pytorch/issues).
*   To learn more about contributing to Pytorch, see our [Contribution page](CONTRIBUTING.md). For more information about PyTorch releases, see [Release page](RELEASE.md).

## The Team

PyTorch is a community-driven project. The project is maintained by [Soumith Chintala](http://soumith.ch), [Gregory Chanan](https://github.com/gchanan), [Dmytro Dzhulgakov](https://github.com/dzhulgakov), [Edward Yang](https://github.com/ezyang), and [Nikita Shulga](https://github.com/malfet). A non-exhaustive but growing list of contributors includes [Trevor Killeen](https://github.com/killeent), [Sasank Chilamkurthy](https://github.com/chsasank), [Sergey Zagoruyko](https://github.com/szagoruyko), [Adam Lerer](https://github.com/adamlerer), [Francisco Massa](https://github.com/fmassa), [Alykhan Tejani](https://github.com/alykhantejani), [Luca Antiga](https://github.com/lantiga), [Alban Desmaison](https://github.com/albanD), [Andreas Koepf](https://github.com/andreaskoepf), [James Bradbury](https://github.com/jekbradbury), [Zeming Lin](https://github.com/ebetica), [Yuandong Tian](https://github.com/yuandong-tian), [Guillaume Lample](https://github.com/glample), [Marat Dukhan](https://github.com/Maratyszcza), [Natalia Gimelshein](https://github.com/ngimel), [Christian Sarofeen](https://github.com/csarofeen), [Martin Raison](https://github.com/martinraison), [Edward Yang](https://github.com/ezyang), [Zachary Devito](https://github.com/zdevito). <!-- codespell:ignore -->

Note: This project is unrelated to [hughperkins/pytorch](https://github.com/hughperkins/pytorch) with the same name. Hugh is a valuable contributor to the Torch community and has helped with many things Torch and PyTorch.

## License

PyTorch is available under a BSD-style license, as found in the [LICENSE](LICENSE) file.