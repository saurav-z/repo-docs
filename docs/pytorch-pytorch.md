![PyTorch Logo](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/pytorch-logo-dark.png)

# PyTorch: Deep Learning with Flexibility and Speed

**PyTorch is a powerful and versatile open-source machine learning framework that accelerates the journey from research prototyping to production deployment.** Explore the full potential of PyTorch at the [original repository](https://github.com/pytorch/pytorch).

## Key Features of PyTorch

*   **Tensor Computation with GPU Acceleration:** Leverage the power of GPUs for fast tensor operations, similar to NumPy but with significant performance gains.
*   **Dynamic Neural Networks with Autograd:** Build and modify neural networks with unmatched flexibility using a tape-based automatic differentiation system.
*   **Python-First Development:** Enjoy deep integration with Python, allowing you to use familiar libraries like NumPy, SciPy, and Cython.
*   **Imperative and Intuitive:** Experience an intuitive and easy-to-use framework where code execution is straightforward and debugging is simplified.
*   **Fast and Lean:** Benefit from minimal framework overhead and integration with acceleration libraries like Intel MKL, cuDNN, and NCCL for optimal performance.
*   **Effortless Extensions:** Easily create custom neural network modules and interface with PyTorch's Tensor API using Python or C/C++.

## Core Components

| Component                  | Description                                                                          |
| :------------------------- | :----------------------------------------------------------------------------------- |
| [**torch**](https://pytorch.org/docs/stable/torch.html)           | Tensor library with GPU support.                                    |
| [**torch.autograd**](https://pytorch.org/docs/stable/autograd.html) | Automatic differentiation library.                                    |
| [**torch.jit**](https://pytorch.org/docs/stable/jit.html)          | Compilation stack (TorchScript) for serializable and optimizable models. |
| [**torch.nn**](https://pytorch.org/docs/stable/nn.html)           | Neural networks library.                                             |
| [**torch.multiprocessing**](https://pytorch.org/docs/stable/multiprocessing.html) | Python multiprocessing with memory sharing for tensors.                       |
| [**torch.utils**](https://pytorch.org/docs/stable/data.html)        | DataLoaders and utilities.                                            |

## Installation

Find installation instructions and binaries on the official PyTorch website: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

### Installation Methods

*   **Binaries:** Install via Conda or pip wheels.
*   **From Source:** Build from source to customize or contribute.
    *   **Prerequisites:** Ensure Python 3.9 or later, a C++17-compatible compiler, and (optionally) CUDA, ROCm, or Intel GPU support.
    *   **Steps:**
        1.  Get the source code using `git clone`.
        2.  Install dependencies using `pip install -r requirements.txt`.
        3.  Install PyTorch using `python -m pip install --no-build-isolation -v -e .`
*   **Docker Image:** Use pre-built images or build your own for consistent environments.

### Platform-Specific Instructions

*   **NVIDIA Jetson Platforms:** Find pre-built wheels and L4T containers [here](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048).
*   **Intel GPU Support:** [Follow these instructions](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpus.html) for installation and setup.
*   **AMD ROCm Support:** Install [AMD ROCm](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html) for Linux systems and configure environment variables as needed.

## Getting Started

*   **Tutorials:** [https://pytorch.org/tutorials/](https://pytorch.org/tutorials/)
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

*   **Forums:** [https://discuss.pytorch.org](https://discuss.pytorch.org)
*   **GitHub Issues:** Bug reports, feature requests, installation issues, etc.
*   **Slack:** [PyTorch Slack](https://pytorch.slack.com/)
*   **Newsletter:** [Sign-up here](https://eepurl.com/cbG0rv)
*   **Facebook Page:** [https://www.facebook.com/pytorch](https://www.facebook.com/pytorch)
*   **Brand Guidelines:** Visit [pytorch.org](https://pytorch.org/)

## Releases and Contributing

PyTorch typically releases three minor versions per year.  Report bugs via [GitHub Issues](https://github.com/pytorch/pytorch/issues).  Contribute new features after discussing them via an issue.  Review the [Contribution page](CONTRIBUTING.md) and [Release page](RELEASE.md) for more details.

## The Team

The PyTorch community is driven by a team of talented engineers and researchers. Current maintainers include Soumith Chintala, Gregory Chanan, Dmytro Dzhulgakov, Edward Yang, and Nikita Shulga.

## License

PyTorch is licensed under a BSD-style license (see the [LICENSE](LICENSE) file).