<!-- PyTorch Logo -->
![PyTorch Logo](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/pytorch-logo-dark.png)

# PyTorch: Open Source Deep Learning Framework

**PyTorch is a powerful and flexible open-source deep learning framework that accelerates the path from research prototyping to production deployment.** ([Original Repo](https://github.com/pytorch/pytorch))

## Key Features

*   **Tensor Computation with GPU Acceleration:** Leverage the power of GPUs for fast tensor operations, similar to NumPy.
*   **Dynamic Neural Networks:** Build and modify neural networks on-the-fly with a tape-based autograd system.
*   **Python-First Approach:** Seamless integration with Python and support for popular libraries like NumPy and SciPy.
*   **Imperative Design:** Intuitive and easy-to-debug code execution, making development straightforward.
*   **Fast and Lean:** Optimized for speed, utilizing acceleration libraries and efficient memory management.
*   **Extensible:** Easily create custom neural network modules and interfaces with the Tensor API.

## Getting Started with PyTorch

PyTorch provides a comprehensive ecosystem for deep learning, including:

*   [Tutorials](https://pytorch.org/tutorials/) to guide you through the fundamentals.
*   [Examples](https://github.com/pytorch/examples) demonstrating practical applications.
*   [API Reference](https://pytorch.org/docs/) for detailed documentation.
*   [Glossary](https://github.com/pytorch/pytorch/blob/main/GLOSSARY.md) to understand the terminology.

## Dive Deeper into PyTorch:

### A GPU-Ready Tensor Library

PyTorch utilizes tensors, which are similar to NumPy's ndarrays, for scientific computing. PyTorch tensors can reside on CPUs or GPUs, dramatically accelerating computation. It offers a wide range of tensor operations like slicing, indexing, mathematical operations, linear algebra, and reductions, optimized for speed.

### Dynamic Neural Networks: Tape-Based Autograd

PyTorch distinguishes itself with its dynamic approach to building neural networks through reverse-mode auto-differentiation. This technique allows for flexible network architectures that can be modified with zero overhead, providing an excellent balance between speed and research adaptability.

<img src="https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/dynamic_graph.gif" alt="Dynamic graph" width="500"/>

### Python First

PyTorch is designed to be deeply integrated into Python, offering natural integration for NumPy, SciPy, and scikit-learn users. It enables you to extend PyTorch using familiar libraries like Cython and Numba, keeping you from reinventing the wheel unnecessarily.

### Imperative Experiences

PyTorch is built for intuitive, linear execution, providing a straightforward debugging experience. Errors and stack traces point directly to your code's definition, ensuring easy troubleshooting.

### Fast and Lean

PyTorch focuses on minimal framework overhead and utilizes acceleration libraries like Intel MKL and NVIDIA (cuDNN, NCCL) to maximize speed. Its CPU and GPU tensor and neural network backends are mature and provide robust performance.

### Extensions Without Pain

Writing new neural network modules or interfacing with PyTorch's Tensor API is designed to be effortless. You can use Python's `torch` API or NumPy-based libraries to create your layers, and a convenient extension API is available for C/C++ layers, which provides efficiency with minimal boilerplate.

## Installation

Follow the instructions to install PyTorch:

*   **Binaries:** Install pre-built binaries via Conda or pip wheels available on the [PyTorch website](https://pytorch.org/get-started/locally/).
*   **NVIDIA Jetson Platforms:** Find Python wheels for Jetson Nano, TX1/TX2, Xavier NX/AGX, and AGX Orin [here](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048).
*   **From Source:** Detailed instructions for building from source are provided below.
*   **Docker Image:** Use pre-built images or build your own.

### From Source - Prerequisites

*   Python 3.9 or later.
*   A compiler that supports C++17 (gcc 9.4.0 or newer is required, on Linux).
*   Visual Studio or Visual Studio Build Tool (Windows only).

**Environment Setup**

*   **Linux:**

```bash
$ source <CONDA_INSTALL_DIR>/bin/activate
$ conda create -y -n <CONDA_NAME>
$ conda activate <CONDA_NAME>
```

*   **Windows:**

```bash
$ source <CONDA_INSTALL_DIR>\Scripts\activate.bat
$ conda create -y -n <CONDA_NAME>
$ conda activate <CONDA_NAME>
$ call "C:\Program Files\Microsoft Visual Studio\<VERSION>\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
```

**NVIDIA CUDA Support**
*   [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads)
*   [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) v8.5 or above
*   [Compiler](https://gist.github.com/ax3l/9489132) compatible with CUDA

**AMD ROCm Support**
*   [AMD ROCm](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html) 4.0 and above installation

**Intel GPU Support**
*   Follow the [PyTorch Prerequisites for Intel GPUs](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpus.html) instructions.

### Get the PyTorch Source

```bash
git clone https://github.com/pytorch/pytorch
cd pytorch
git submodule sync
git submodule update --init --recursive
```

### Install Dependencies

**Common**

```bash
conda install cmake ninja
pip install -r requirements.txt
```

**On Linux**

```bash
pip install mkl-static mkl-include
.ci/docker/common/install_magma_conda.sh 12.4
make triton
```

**On MacOS**

```bash
pip install mkl-static mkl-include
conda install pkg-config libuv
```

**On Windows**

```bash
pip install mkl-static mkl-include
conda install -c conda-forge libuv=1.39
```

### Install PyTorch

**On Linux**

```bash
python tools/amd_build/build_amd.py (for ROCm)
export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"
python -m pip install --no-build-isolation -v -e .
```

**On macOS**

```bash
python -m pip install --no-build-isolation -v -e .
```

**On Windows**

```cmd
python -m pip install --no-build-isolation -v -e .
```

### Docker Image

You can also build your own docker image:
```bash
make -f docker.Makefile
```

### Building the Documentation

Install [Sphinx](http://www.sphinx-doc.org) and the pytorch_sphinx_theme2 and run `make html`.

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

*   Forums: [discuss.pytorch.org](https://discuss.pytorch.org)
*   GitHub Issues
*   Slack: [PyTorch Slack](https://pytorch.slack.com/) (request an invite via [this form](https://goo.gl/forms/PP1AGvNHpSaJP8to1))
*   Newsletter: [Sign-up](https://eepurl.com/cbG0rv)
*   Facebook Page: [PyTorch Facebook](https://www.facebook.com/pytorch)

## Releases and Contributing

PyTorch typically releases three minor versions annually. Report bugs via [GitHub Issues](https://github.com/pytorch/pytorch/issues).

Contributions are welcome!  Discuss new features before submitting a PR.  See the [Contribution page](CONTRIBUTING.md) for guidance.  For more about releases, see [Release page](RELEASE.md).

## The Team

PyTorch is maintained by a community of engineers and researchers, including:

*   [Soumith Chintala](http://soumith.ch)
*   [Gregory Chanan](https://github.com/gchanan)
*   [Dmytro Dzhulgakov](https://github.com/dzhulgakov)
*   [Edward Yang](https://github.com/ezyang)
*   [Nikita Shulga](https://github.com/malfet)

And many others.

## License

PyTorch is licensed under a BSD-style license; see the [LICENSE](LICENSE) file.