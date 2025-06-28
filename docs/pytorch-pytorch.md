[![PyTorch Logo](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/pytorch-logo-dark.png)](https://github.com/pytorch/pytorch)

# PyTorch: Deep Learning with Flexibility and Speed

**PyTorch is an open-source machine learning framework that accelerates the path from research prototyping to production deployment.**  You can find the original repository [here](https://github.com/pytorch/pytorch).

**Key Features:**

*   **Tensor Computation with GPU Acceleration:**  Provides tensor operations similar to NumPy, but with support for powerful GPU acceleration, enabling faster computation for demanding tasks.
*   **Dynamic Neural Networks:**  Built on a tape-based autograd system, PyTorch allows for flexible and dynamic neural network architectures, ideal for cutting-edge research and experimentation.
*   **Python-First Design:**  Deeply integrated with Python, making it easy to use with your favorite Python packages (NumPy, SciPy, etc.) and extending functionality with custom code.
*   **Imperative Programming:**  Offers an intuitive and easy-to-debug imperative programming experience, allowing developers to write and understand code linearly.
*   **Fast and Lean:**  Optimized for performance with minimal framework overhead, integrating acceleration libraries like Intel MKL and NVIDIA cuDNN, ensuring efficient use of resources.
*   **Easy Extensibility:**  Simplifies the creation of custom neural network modules and interfaces with its Tensor API.

## Core Components

PyTorch offers a modular design with key components:

*   **torch:** The Tensor library.
*   **torch.autograd:** The automatic differentiation library.
*   **torch.jit:**  The TorchScript compilation stack.
*   **torch.nn:** The neural networks library.
*   **torch.multiprocessing:** Python multiprocessing with memory sharing.
*   **torch.utils:** Data loading and utility functions.

## Getting Started

*   **[Tutorials](https://pytorch.org/tutorials/):** Start learning the basics of PyTorch.
*   **[Examples](https://github.com/pytorch/examples):** Explore easy-to-understand PyTorch code.
*   **[API Reference](https://pytorch.org/docs/):** Reference for all PyTorch APIs.

## Installation

You can install PyTorch through binaries or from source.

### Binaries

Install via Conda or pip wheels.  Detailed instructions are on the [PyTorch website](https://pytorch.org/get-started/locally/).

#### NVIDIA Jetson Platforms

Python wheels are provided for NVIDIA Jetson Nano, TX1/TX2, Xavier NX/AGX, and AGX Orin.  See [here](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048) for instructions.

### From Source

#### Prerequisites
* Python 3.9 or later
* C++17 compliant compiler.
* Windows requires Visual Studio or Visual Studio Build Tool.

**CUDA, ROCm, and Intel GPU Support**

*   **CUDA:**  Follow the instructions to compile with NVIDIA CUDA by selecting a supported version of CUDA from our support matrix and installing the prerequisites.
*   **ROCm:** Compile with AMD ROCm support by installing AMD ROCm 4.0 and above.
*   **Intel GPU:** Install the [PyTorch Prerequisites for Intel GPUs](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpus.html) and follow the instructions.

#### Build Instructions

```bash
git clone https://github.com/pytorch/pytorch
cd pytorch
git submodule sync
git submodule update --init --recursive
```

#### Install Dependencies

**Common**

```bash
conda install cmake ninja
pip install -r requirements.txt
```

**Linux**

```bash
pip install mkl-static mkl-include
.ci/docker/common/install_magma_conda.sh 12.4  # CUDA only
make triton #Optional for torch.compile
```

**macOS**

```bash
pip install mkl-static mkl-include
conda install pkg-config libuv # Optional for torch.distributed
```

**Windows**

```bash
pip install mkl-static mkl-include
conda install -c conda-forge libuv=1.39 # Optional for torch.distributed
```

#### Install PyTorch

**Linux**

```bash
# If compiling for ROCm
python tools/amd_build/build_amd.py

export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"
python setup.py develop
```

**macOS**

```bash
python3 setup.py develop
```

**Windows**
```cmd
python setup.py develop
```

##### Adjust Build Options (Optional)

```bash
#Linux and MacOS
export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"
CMAKE_ONLY=1 python setup.py build
ccmake build  # or cmake-gui build

```

## Building the Documentation

*   Requires [Sphinx](http://www.sphinx-doc.org) and pytorch_sphinx_theme2.

```bash
cd docs/
pip install -r requirements.txt
make html
make serve
```

### Building a PDF

```bash
make latexpdf
cd build/latex
make LATEXOPTS="-interaction=nonstopmode"
```

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

*   [Forums](https://discuss.pytorch.org): Discuss implementations and research.
*   [GitHub Issues](https://github.com/pytorch/pytorch/issues): Bug reports, feature requests, etc.
*   [Slack](https://pytorch.slack.com/): Chat and collaboration.
*   [Newsletter](https://eepurl.com/cbG0rv): Announcements.
*   [Facebook](https://www.facebook.com/pytorch): Important announcements.

## Releases and Contributing

PyTorch has three minor releases a year.  [File an issue](https://github.com/pytorch/pytorch/issues) for bug reports.  To contribute, review the [Contribution page](CONTRIBUTING.md) and [Release page](RELEASE.md).

## The Team

PyTorch is a community-driven project.  Maintained by [Soumith Chintala](http://soumith.ch), [Gregory Chanan](https://github.com/gchanan), [Dmytro Dzhulgakov](https://github.com/dzhulgakov), [Edward Yang](https://github.com/ezyang), and [Nikita Shulga](https://github.com/malfet) with contributions from many others.

## License

PyTorch is available under a BSD-style license ([LICENSE](LICENSE)).