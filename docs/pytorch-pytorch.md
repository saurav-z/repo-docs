![PyTorch Logo](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/pytorch-logo-dark.png)

# PyTorch: Deep Learning Framework for Research and Production

**PyTorch is a powerful and flexible deep learning framework that accelerates the machine learning research and deployment.** Developed by Meta (formerly Facebook), PyTorch provides a seamless path from research prototyping to production deployment. [Explore the PyTorch Repository](https://github.com/pytorch/pytorch).

**Key Features:**

*   **Tensor Computation with GPU Acceleration:** Leverage the power of GPUs for fast tensor operations, similar to NumPy.
*   **Dynamic Neural Networks:** Build and modify neural networks with unprecedented flexibility using a tape-based autograd system.
*   **Python-First Approach:** Seamlessly integrate PyTorch with your existing Python workflows, leveraging your favorite libraries like NumPy, SciPy, and Cython.
*   **Imperative Programming:** Enjoy an intuitive and easy-to-debug experience with imperative-style code execution.
*   **Fast and Lean:** Benefit from minimal framework overhead and optimized performance, including integrations with libraries like Intel MKL, cuDNN, and NCCL.
*   **Extensible Architecture:** Effortlessly create custom neural network modules and extend the framework with your own operations.

## Core Components

PyTorch consists of the following core components:

| Component | Description | Link |
| ----------- | ----------- | ----------- |
| **torch** | Tensor library with GPU support |  [torch](https://pytorch.org/docs/stable/torch.html) |
| **torch.autograd** | Tape-based automatic differentiation | [torch.autograd](https://pytorch.org/docs/stable/autograd.html) |
| **torch.jit** | Compilation stack for serializable and optimizable models | [torch.jit](https://pytorch.org/docs/stable/jit.html) |
| **torch.nn** | Neural networks library | [torch.nn](https://pytorch.org/docs/stable/nn.html) |
| **torch.multiprocessing** | Multiprocessing with shared tensors | [torch.multiprocessing](https://pytorch.org/docs/stable/multiprocessing.html) |
| **torch.utils** | Data loading utilities | [torch.utils](https://pytorch.org/docs/stable/data.html) |

## Why Use PyTorch?

PyTorch excels as:

*   A powerful replacement for NumPy, enabling GPU acceleration for scientific computing.
*   A dynamic deep learning research platform, providing unmatched flexibility and speed.

### A GPU-Ready Tensor Library

PyTorch provides Tensors, similar to NumPy's ndarrays, allowing data to reside on the CPU or GPU, significantly accelerating computations. This library offers a broad range of tensor operations for scientific computation, including slicing, indexing, and linear algebra, optimized for speed.

### Dynamic Neural Networks: Tape-Based Autograd

Unlike frameworks with static computation graphs, PyTorch employs reverse-mode automatic differentiation, or "autograd," enabling real-time changes to network behavior without overhead. This unique approach, inspired by research, ensures the best balance of speed and flexibility for dynamic research environments.

### Python First

PyTorch is built to be deeply integrated into Python, allowing for a natural, intuitive experience. It seamlessly integrates with NumPy, SciPy, scikit-learn, and other Python tools. You can define new layers in Python, and extend them with libraries like Cython and Numba, without reinventing the wheel.

### Imperative Experiences

PyTorch is designed for an intuitive, linear development process. You execute code, and it runs immediately, allowing for easy debugging. Stack traces accurately pinpoint where your code is defined, making troubleshooting a breeze.

### Fast and Lean

PyTorch minimizes framework overhead and leverages acceleration libraries like Intel MKL, cuDNN, and NCCL. The core CPU and GPU tensor and neural network backends are highly mature. Additionally, PyTorch offers efficient memory usage, enabling the training of larger deep learning models.

### Extensions Without Pain

Extending PyTorch is simple. Creating neural network modules or interacting with the Tensor API is straightforward, with minimal abstraction. Write new layers in Python or C/C++ using our extension API, which is efficient and requires minimal boilerplate.

## Installation

Install binaries via Conda or pip wheels at: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/).

**Installation from Source:**

1.  **Prerequisites:**

    *   Python 3.9 or later
    *   C++17 compatible compiler (e.g., GCC 9.4.0 or newer)
    *   Visual Studio or Visual Studio Build Tool (Windows)
2.  **Clone the Repository:**

    ```bash
    git clone https://github.com/pytorch/pytorch
    cd pytorch
    git submodule sync
    git submodule update --init --recursive
    ```

3.  **Install Dependencies:**

    ```bash
    conda install cmake ninja  # or pip install
    pip install -r requirements.txt
    ```

4.  **Install PyTorch:**

    ```bash
    export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"
    python -m pip install --no-build-isolation -v -e .
    ```

    Detailed instructions for CUDA, ROCm, and Intel GPU support are available in the original [README](https://github.com/pytorch/pytorch).

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

*   Forums: [https://discuss.pytorch.org](https://discuss.pytorch.org)
*   GitHub Issues
*   Slack: [PyTorch Slack](https://pytorch.slack.com/) (invite form: https://goo.gl/forms/PP1AGvNHpSaJP8to1)
*   Newsletter: Sign-up [here](https://eepurl.com/cbG0rv)
*   Facebook: [https://www.facebook.com/pytorch](https://www.facebook.com/pytorch)

## Releases and Contributing

PyTorch has three minor releases a year. Please report bugs by [filing an issue](https://github.com/pytorch/pytorch/issues). For contributions, follow the guidelines in [CONTRIBUTING.md](CONTRIBUTING.md) and [RELEASE.md](RELEASE.md).

## The Team

PyTorch is a community-driven project maintained by engineers and researchers including [Soumith Chintala](http://soumith.ch), [Gregory Chanan](https://github.com/gchanan), [Dmytro Dzhulgakov](https://github.com/dzhulgakov), [Edward Yang](https://github.com/ezyang), [Alban Desmaison](https://github.com/albanD), [Piotr Bialecki](https://github.com/ptrblck) and [Nikita Shulga](https://github.com/malfet) with contributions from many talented individuals.

## License

PyTorch is licensed under a BSD-style license, see the [LICENSE](LICENSE) file.