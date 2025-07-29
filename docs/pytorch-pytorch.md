[![PyTorch Logo](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/pytorch-logo-dark.png)](https://github.com/pytorch/pytorch)

# PyTorch: A Deep Learning Framework for Research and Production

**PyTorch is a leading open-source deep learning framework, providing a flexible and efficient platform for building and deploying machine learning models.**

## Key Features:

*   **Tensor Computation with GPU Acceleration:**  Leverage the power of GPUs for fast tensor operations, mirroring the functionality of NumPy with added speed.
*   **Dynamic Neural Networks with Tape-Based Autograd:** Build and modify neural networks with unparalleled flexibility, allowing for on-the-fly adjustments without the need to rebuild.
*   **Python-First Design:** Seamlessly integrates with the Python ecosystem, offering ease of use and compatibility with existing libraries like NumPy, SciPy, and Cython.
*   **Imperative Programming:** Benefit from an intuitive, imperative programming style, making debugging and understanding your code straightforward.
*   **Fast and Lean:** Optimized for performance, PyTorch incorporates acceleration libraries (Intel MKL, cuDNN, NCCL) and efficient memory management, enabling the training of large and complex models.
*   **Extensible and Customizable:** Easily create new neural network modules and integrate with PyTorch's Tensor API, expanding your model building capabilities.

## Core Components:

| Component                 | Description                                                                                                                                                     |
| :------------------------ | :-------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **torch**                 | Tensor library like NumPy, with strong GPU support.                                                                                                            |
| **torch.autograd**        | Tape-based automatic differentiation library for differentiable Tensor operations.                                                                                |
| **torch.jit**             | Compilation stack (TorchScript) to create serializable and optimizable models from PyTorch code.                                                                |
| **torch.nn**              | Neural networks library integrated with autograd, offering maximum flexibility.                                                                                  |
| **torch.multiprocessing** | Python multiprocessing with magical memory sharing of torch Tensors across processes, useful for data loading and Hogwild training.                             |
| **torch.utils**           | DataLoader and other utility functions.                                                                                                                      |

## [Get Started with PyTorch](https://pytorch.org/tutorials/beginner/basics/intro.html)

## Installation

Choose the installation method that best suits your needs:

### [Binaries](https://pytorch.org/get-started/locally/)
*   **Recommended:**  Use Conda or pip wheels for a straightforward setup.  Instructions available at [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
*   **NVIDIA Jetson Platforms:** Pre-built wheels are available for Jetson Nano, TX1/TX2, Xavier NX/AGX, and AGX Orin ([details](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048)).

### From Source

**Prerequisites:**  Ensure you have Python 3.9+, a C++17 compatible compiler, and (optionally) CUDA, ROCm, or Intel GPU support.

#### **NVIDIA CUDA Support**

*   Install [CUDA](https://developer.nvidia.com/cuda-downloads), [cuDNN](https://developer.nvidia.com/cudnn), and a compatible compiler.
*   If building with CUDA, install [NVTX](https://docs.nvidia.com/gameworks/content/gameworkslibrary/nvtx/nvidia_tools_extension_library_nvtx.htm).
*   Follow the instructions to disable CUDA: `USE_CUDA=0`.

#### **AMD ROCm Support**

*   Install [AMD ROCm](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html) 4.0+ on Linux.
*   Set the `ROCM_PATH` environment variable if ROCm is not in the default location.
*   To disable ROCm support, export the environment variable `USE_ROCM=0`.

#### **Intel GPU Support**

*   Follow the [PyTorch Prerequisites for Intel GPUs](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpus.html) for Linux and Windows.
*   To disable Intel GPU support, export the environment variable `USE_XPU=0`.

**Steps:**

1.  Get the PyTorch Source:

    ```bash
    git clone https://github.com/pytorch/pytorch
    cd pytorch
    git submodule sync
    git submodule update --init --recursive
    ```
2.  Install Dependencies (Common):

    ```bash
    conda install cmake ninja  # or install with your package manager
    pip install -r requirements.txt
    ```
3.  Install Dependencies (Platform-Specific):

    *   **Linux:**  `pip install mkl-static mkl-include` and optionally install `magma` and `triton`.
    *   **macOS:** `pip install mkl-static mkl-include`, `conda install pkg-config libuv` (if torch.distributed is needed).
    *   **Windows:** `pip install mkl-static mkl-include`, `conda install -c conda-forge libuv=1.39` (if torch.distributed is needed).

4.  Install PyTorch:

    *   **Linux (with ROCm):** Run `python tools/amd_build/build_amd.py` before installation.
    *   **Installation:** Follow the instructions for your platform using `python -m pip install --no-build-isolation -v -e .`.

5.  **Optional: Adjust Build Options**  Use `CMAKE_ONLY=1 python setup.py build; ccmake build` to configure CMake variables before building.

### Docker Image

*   **Pre-built Images:** Pull pre-built images from Docker Hub: `docker run --gpus all --rm -ti --ipc=host pytorch/pytorch:latest`
*   **Build Your Own:**  Build images with CUDA 11.1 support (requires Docker > 18.06).  Use the `docker.Makefile` and specify `PYTHON_VERSION` or `CMAKE_VARS` as needed.

### Building the Documentation

1.  Install Sphinx and the `pytorch_sphinx_theme2`.
2.  `cd docs/`, then `pip install -r requirements.txt; make html; make serve`.

#### Building a PDF

1.  Install `texlive` and LaTeX (e.g., `brew install --cask mactex` on macOS).
2.  `make latexpdf`, then `make LATEXOPTS="-interaction=nonstopmode"` inside `build/latex`.

### [Previous Versions](https://pytorch.org/get-started/previous-versions)

Find installation instructions and binaries for previous PyTorch versions.

## Resources

*   [PyTorch.org](https://pytorch.org/)
*   [PyTorch Tutorials](https://pytorch.org/tutorials/)
*   [PyTorch Examples](https://github.com/pytorch/examples)
*   [PyTorch Models](https://pytorch.org/hub/)
*   And More... (Udacity, Coursera, Twitter, Blog, YouTube - see original README)

## Communication

*   [Forums](https://discuss.pytorch.org/)
*   [GitHub Issues](https://github.com/pytorch/pytorch/issues)
*   [Slack](https://pytorch.slack.com/)
*   [Newsletter](https://eepurl.com/cbG0rv)
*   [Facebook Page](https://www.facebook.com/pytorch)

## Releases and Contributing

PyTorch has three minor releases a year.  Report bugs via [GitHub Issues](https://github.com/pytorch/pytorch/issues).  For contributions, follow the guidelines in [CONTRIBUTING.md](CONTRIBUTING.md) and [RELEASE.md](RELEASE.md).

## The Team

Maintained by [Soumith Chintala](http://soumith.ch), [Gregory Chanan](https://github.com/gchanan), [Dmytro Dzhulgakov](https://github.com/dzhulgakov), [Edward Yang](https://github.com/ezyang), and [Nikita Shulga](https://github.com/malfet) with contributions from many.

## License

PyTorch is licensed under a BSD-style license, found in the [LICENSE](LICENSE) file.

---
[Back to Top](https://github.com/pytorch/pytorch)