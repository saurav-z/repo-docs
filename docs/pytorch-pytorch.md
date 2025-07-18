[![PyTorch Logo](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/pytorch-logo-dark.png)](https://github.com/pytorch/pytorch)

# PyTorch: The Flexible and Fast Deep Learning Framework

**PyTorch is a powerful, open-source deep learning framework built for research and production, offering flexibility, speed, and a Python-first approach.**

---

**Key Features:**

*   **Tensor Computation with GPU Acceleration:**  Offers powerful tensor operations, similar to NumPy, with seamless GPU integration for significantly faster computation.
*   **Dynamic Neural Networks with Autograd:**  Builds and trains deep neural networks using a tape-based autograd system, enabling unparalleled flexibility and dynamic behavior.
*   **Python-First Design:** Deeply integrated with Python, allowing you to leverage your existing Python knowledge, libraries like NumPy, SciPy, and Cython, and the full power of the Python ecosystem.
*   **Imperative Programming Style:**  Designed to be intuitive and easy to use with an imperative programming style, simplifying debugging and code understanding.
*   **Fast and Lean:** Highly optimized with minimal framework overhead, integrating with acceleration libraries like Intel MKL, cuDNN, and NCCL for maximum speed and efficiency.
*   **Effortless Extensions:** Provides a straightforward API for writing new neural network modules and extending PyTorch with custom operations in both Python and C/C++.

---

## Core Components

PyTorch's functionality is built upon several core components:

| Component              | Description                                                                                                |
| ---------------------- | ---------------------------------------------------------------------------------------------------------- |
| **torch**              | Tensor library with GPU support, similar to NumPy.                                                        |
| **torch.autograd**     | Automatic differentiation library for computing gradients of all differentiable Tensor operations in `torch`. |
| **torch.jit**          | Compilation stack (TorchScript) for creating serializable and optimizable models from PyTorch code.       |
| **torch.nn**           | Neural networks library deeply integrated with autograd for maximum flexibility.                          |
| **torch.multiprocessing** | Multiprocessing support with efficient memory sharing for tensors, useful for data loading and training.      |
| **torch.utils**        | Data loading and other utility functions for ease of use.                                                  |

## [More About PyTorch](https://pytorch.org/tutorials/beginner/basics/intro.html)

PyTorch is a versatile tool that can be used in many different areas:

### A GPU-Ready Tensor Library

PyTorch provides Tensors that can live either on the CPU or the GPU and accelerates the
computation by a huge amount.

We provide a wide variety of tensor routines to accelerate and fit your scientific computation needs
such as slicing, indexing, mathematical operations, linear algebra, reductions.
And they are fast!

### Dynamic Neural Networks: Tape-Based Autograd

PyTorch has a unique way of building neural networks: using and replaying a tape recorder.

Most frameworks such as TensorFlow, Theano, Caffe, and CNTK have a static view of the world.
One has to build a neural network and reuse the same structure again and again.
Changing the way the network behaves means that one has to start from scratch.

With PyTorch, we use a technique called reverse-mode auto-differentiation, which allows you to
change the way your network behaves arbitrarily with zero lag or overhead. Our inspiration comes
from several research papers on this topic, as well as current and past work such as
[torch-autograd](https://github.com/twitter/torch-autograd),
[autograd](https://github.com/HIPS/autograd),
[Chainer](https://chainer.org), etc.

While this technique is not unique to PyTorch, it's one of the fastest implementations of it to date.
You get the best of speed and flexibility for your crazy research.

![Dynamic graph](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/dynamic_graph.gif)

### Python First

PyTorch is not a Python binding into a monolithic C++ framework.
It is built to be deeply integrated into Python.
You can use it naturally like you would use [NumPy](https://www.numpy.org/) / [SciPy](https://www.scipy.org/) / [scikit-learn](https://scikit-learn.org) etc.
You can write your new neural network layers in Python itself, using your favorite libraries
and use packages such as [Cython](https://cython.org/) and [Numba](http://numba.pydata.org/).
Our goal is to not reinvent the wheel where appropriate.

### Imperative Experiences

PyTorch is designed to be intuitive, linear in thought, and easy to use.
When you execute a line of code, it gets executed. There isn't an asynchronous view of the world.
When you drop into a debugger or receive error messages and stack traces, understanding them is straightforward.
The stack trace points to exactly where your code was defined.
We hope you never spend hours debugging your code because of bad stack traces or asynchronous and opaque execution engines.

### Fast and Lean

PyTorch has minimal framework overhead. We integrate acceleration libraries
such as [Intel MKL](https://software.intel.com/mkl) and NVIDIA ([cuDNN](https://developer.nvidia.com/cudnn), [NCCL](https://developer.nvidia.com/nccl)) to maximize speed.
At the core, its CPU and GPU Tensor and neural network backends
are mature and have been tested for years.

Hence, PyTorch is quite fast — whether you run small or large neural networks.

The memory usage in PyTorch is extremely efficient compared to Torch or some of the alternatives.
We've written custom memory allocators for the GPU to make sure that
your deep learning models are maximally memory efficient.
This enables you to train bigger deep learning models than before.

### Extensions Without Pain

Writing new neural network modules, or interfacing with PyTorch's Tensor API was designed to be straightforward
and with minimal abstractions.

You can write new neural network layers in Python using the torch API
[or your favorite NumPy-based libraries such as SciPy](https://pytorch.org/tutorials/advanced/numpy_extensions_tutorial.html).

If you want to write your layers in C/C++, we provide a convenient extension API that is efficient and with minimal boilerplate.
No wrapper code needs to be written. You can see [a tutorial here](https://pytorch.org/tutorials/advanced/cpp_extension.html) and [an example here](https://github.com/pytorch/extension-cpp).

## Installation

Detailed installation instructions are available on the official PyTorch website:  [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

### Binaries

Install pre-built binaries via Conda or pip wheels.

#### NVIDIA Jetson Platforms

Pre-built wheels are available for NVIDIA Jetson platforms.

### From Source

#### Prerequisites

*   Python 3.9 or later
*   C++17 compiler (e.g., clang, gcc 9.4.0+)
*   Visual Studio or Visual Studio Build Tool (Windows)

*   **CUDA Support:**  Install NVIDIA CUDA, cuDNN (v8.5+), and a compatible compiler.  See the [CUDA support matrix](https://pytorch.org/get-started/locally/) for supported versions.
*   **ROCm Support:** Install AMD ROCm (4.0+) for AMD GPU support on Linux.
*   **Intel GPU Support:**  Follow the [Intel GPU prerequisites](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpus.html).

#### Steps

1.  **Get the PyTorch Source:**

    ```bash
    git clone https://github.com/pytorch/pytorch
    cd pytorch
    git submodule sync
    git submodule update --init --recursive
    ```

2.  **Install Dependencies:**

    ```bash
    conda install cmake ninja  # Or pip install for other environments
    pip install -r requirements.txt
    ```

    *   **Linux:**  `pip install mkl-static mkl-include`;  CUDA-specific:  install magma.  Intel GPU-specific:  `make triton`
    *   **macOS:** `pip install mkl-static mkl-include`;  for distributed: `conda install pkg-config libuv`
    *   **Windows:** `pip install mkl-static mkl-include`; for distributed: `conda install -c conda-forge libuv=1.39`

3.  **Install PyTorch:**
    *   **Linux with ROCm:**  Run  `python tools/amd_build/build_amd.py` before installing.
    ```bash
    export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"
    python -m pip install -r requirements-build.txt
    python -m pip install --no-build-isolation -v -e .
    ```
    *   **macOS:**
    ```bash
    export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"
    python -m pip install -r requirements-build.txt
    python -m pip install --no-build-isolation -v -e .
    ```
    *   **Windows (CPU-only):**
    ```cmd
    python -m pip install --no-build-isolation -v -e .
    ```
    *   **Windows (CUDA-based):**
        *   Configure environment variables for MKL, CUDA, and Ninja (if used).
        *   Run  `python -m pip install --no-build-isolation -v -e .`
    *   **Windows (Intel GPU):**
        *   Set up the necessary environment variables (CMAKE_PREFIX_PATH, etc.).
        *   Run  `python -m pip install --no-build-isolation -v -e .`

##### Adjust Build Options (Optional)

```bash
export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"
CMAKE_ONLY=1 python setup.py build
ccmake build  # or cmake-gui build
```

### Docker Image

#### Using pre-built images

```bash
docker run --gpus all --rm -ti --ipc=host pytorch/pytorch:latest
```

#### Building the image yourself

```bash
make -f docker.Makefile
```

### Building the Documentation

*   Install Sphinx and the pytorch_sphinx_theme2.
*   Install torch in your environment.
*   Navigate to the `docs/` directory and run:
    ```bash
    pip install -r requirements.txt
    make html
    make serve
    ```

#### Building a PDF

1.  Run `make latexpdf`.
2.  Navigate to `build/latex` and run `make LATEXOPTS="-interaction=nonstopmode"`.
3.  Run the command one more time so that it generates the correct table
    of contents and index.

### Previous Versions

Installation instructions and binaries for previous PyTorch versions may be found
on [our website](https://pytorch.org/get-started/previous-versions).

## Getting Started

Explore the following resources:

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

*   **Forums:**  [https://discuss.pytorch.org](https://discuss.pytorch.org)
*   **GitHub Issues:**  [https://github.com/pytorch/pytorch/issues](https://github.com/pytorch/pytorch/issues)
*   **Slack:** [PyTorch Slack](https://pytorch.slack.com/).  Get an invite via [this form](https://goo.gl/forms/PP1AGvNHpSaJP8to1)
*   **Newsletter:**  [https://eepurl.com/cbG0rv](https://eepurl.com/cbG0rv)
*   **Facebook:** [https://www.facebook.com/pytorch](https://www.facebook.com/pytorch)

## Releases and Contributing

PyTorch typically releases three minor versions per year.  Report bugs via [GitHub Issues](https://github.com/pytorch/pytorch/issues).

We welcome contributions!  Discuss new features/extensions via issue before sending a PR.  See the [Contribution page](CONTRIBUTING.md) and [Release page](RELEASE.md).

## The Team

PyTorch is a community-driven project maintained by a dedicated team and supported by many talented contributors.  Major contributors include [Soumith Chintala](http://soumith.ch), [Gregory Chanan](https://github.com/gchanan), [Dmytro Dzhulgakov](https://github.com/dzhulgakov), [Edward Yang](https://github.com/ezyang), and [Nikita Shulga](https://github.com/malfet).

## License

PyTorch is licensed under a BSD-style license, found in the [LICENSE](LICENSE) file.