![PyTorch Logo](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/pytorch-logo-dark.png)

# PyTorch: The Deep Learning Framework for Research and Production

**PyTorch is a powerful and versatile open-source deep learning framework, enabling you to build, train, and deploy cutting-edge AI models.** Explore the original repository on [GitHub](https://github.com/pytorch/pytorch) to contribute and learn more.

## Key Features

*   **Tensor Computation with GPU Acceleration:** Leverage the power of GPUs for fast tensor operations, similar to NumPy, for efficient numerical computation.
*   **Dynamic Neural Networks:** Build and train flexible deep neural networks using a tape-based autograd system, offering unparalleled control and adaptability.
*   **Python-First Design:** Enjoy seamless integration with Python, using familiar libraries like NumPy and SciPy, and the flexibility to extend with packages like Cython and Numba.
*   **Imperative Programming:** Experience intuitive code execution with straightforward debugging and clear stack traces, simplifying development.
*   **Fast and Lean:** Benefit from minimal framework overhead and optimized acceleration libraries (Intel MKL, cuDNN, NCCL) for speed and memory efficiency.
*   **Easy Extensions:** Easily write custom neural network modules and interface with PyTorch's Tensor API using Python or C/C++.

## Core Components

| Component                    | Description                                                                                                      |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **torch**                    | Tensor library with GPU support, similar to NumPy.                                                             |
| **torch.autograd**           | Automatic differentiation library for all differentiable Tensor operations.                                       |
| **torch.jit**                | Compilation stack (TorchScript) for serializing and optimizing PyTorch code.                                    |
| **torch.nn**                 | Neural networks library deeply integrated with autograd for maximum flexibility.                                 |
| **torch.multiprocessing**    | Python multiprocessing with shared memory for Tensors, ideal for data loading and Hogwild training.            |
| **torch.utils**              | DataLoader and other utility functions for convenient data handling.                                             |

## Installation

Get started by selecting the appropriate installation method for your needs:

*   **Binaries:** Install pre-built packages using Conda or pip wheels. Instructions are available on the [PyTorch website](https://pytorch.org/get-started/locally/).
*   **From Source:** Build PyTorch from source code, allowing for customization and advanced configurations.
    *   **Prerequisites:** Python 3.9+, a C++17 compiler, and (optionally) CUDA, ROCm, or Intel GPU support.
    *   **Steps:**
        1.  Clone the repository.
        2.  Install dependencies.
        3.  Build and install PyTorch.
*   **Docker Image:** Use pre-built or build your own Docker images for a containerized environment.
    *   **Pre-built images:** `docker run --gpus all --rm -ti --ipc=host pytorch/pytorch:latest`
*   **NVIDIA Jetson Platforms:** wheels provided [here](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048) and the L4T container is published [here](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch)

## Getting Started

*   **Tutorials:** Dive into the fundamentals with comprehensive tutorials at [https://pytorch.org/tutorials/](https://pytorch.org/tutorials/).
*   **Examples:** Explore practical PyTorch code examples across various domains at [https://github.com/pytorch/examples](https://github.com/pytorch/examples).
*   **API Reference:** Access the complete PyTorch API documentation at [https://pytorch.org/docs/](https://pytorch.org/docs/).
*   **Glossary:** Understand key terms and concepts in the [GLOSSARY.md](https://github.com/pytorch/pytorch/blob/main/GLOSSARY.md) file.

## Resources

*   **PyTorch.org:** [https://pytorch.org/](https://pytorch.org/)
*   **PyTorch Tutorials:** [https://pytorch.org/tutorials/](https://pytorch.org/tutorials/)
*   **PyTorch Examples:** [https://github.com/pytorch/examples](https://github.com/pytorch/examples)
*   **PyTorch Models:** [https://pytorch.org/hub/](https://pytorch.org/hub/)
*   **Intro to Deep Learning with PyTorch from Udacity:** [https://www.udacity.com/course/deep-learning-pytorch--ud188](https://www.udacity.com/course/deep-learning-pytorch--ud188)
*   **Intro to Machine Learning with PyTorch from Udacity:** [https://www.udacity.com/course/intro-to-machine-learning-nanodegree--nd229](https://www.udacity.com/course/intro-to-machine-learning-nanodegree--nd229)
*   **Deep Neural Networks with PyTorch from Coursera:** [https://www.coursera.org/learn/deep-neural-networks-with-pytorch](https://www.coursera.org/learn/deep-neural-networks-with-pytorch)
*   **PyTorch Twitter:** [https://twitter.com/PyTorch](https://twitter.com/PyTorch)
*   **PyTorch Blog:** [https://pytorch.org/blog/](https://pytorch.org/blog/)
*   **PyTorch YouTube:** [https://www.youtube.com/channel/UCWXI5YeOsh03QvJ59PMaXFw](https://www.youtube.com/channel/UCWXI5YeOsh03QvJ59PMaXFw)

## Communication

*   **Forums:** Discuss implementations and research at [https://discuss.pytorch.org](https://discuss.pytorch.org).
*   **GitHub Issues:** Report bugs, request features, and discuss installation issues.
*   **Slack:** The [PyTorch Slack](https://pytorch.slack.com/) hosts a primary audience of moderate to experienced PyTorch users and developers for general chat, online discussions, collaboration, etc. If you are a beginner looking for help, the primary medium is [PyTorch Forums](https://discuss.pytorch.org). If you need a slack invite, please fill this form: https://goo.gl/forms/PP1AGvNHpSaJP8to1
*   **Newsletter:** Stay informed with the no-noise, one-way email newsletter, subscribe here: [https://eepurl.com/cbG0rv](https://eepurl.com/cbG0rv)
*   **Facebook Page:** Important announcements about PyTorch. [https://www.facebook.com/pytorch](https://www.facebook.com/pytorch)

## Releases and Contributing

PyTorch typically releases three minor versions annually. To contribute, please review the [Contribution page](CONTRIBUTING.md) and the [Release page](RELEASE.md).

## The Team

PyTorch is a community-driven project. Maintainers: Soumith Chintala, Gregory Chanan, Dmytro Dzhulgakov, Edward Yang, and Nikita Shulga.

## License

PyTorch is licensed under a BSD-style license, as detailed in the [LICENSE](LICENSE) file.