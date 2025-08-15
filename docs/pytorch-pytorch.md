[![PyTorch Logo](https://github.com/pytorch/pytorch/blob/9708fcf92db88b80b9010c68662d634434da3106/docs/source/_static/img/pytorch-logo-dark.png)](https://github.com/pytorch/pytorch)

# PyTorch: Deep Learning with Flexibility and Speed

**PyTorch is a powerful and versatile open-source machine learning framework that empowers researchers and developers to build cutting-edge AI models.** ([Original Repo](https://github.com/pytorch/pytorch))

## Key Features

*   **Tensor Computation with GPU Acceleration:**  Leverage the power of GPUs for fast tensor operations, similar to NumPy, for optimized performance.
*   **Dynamic Neural Networks:** Build and customize deep neural networks with a flexible, tape-based autograd system for dynamic and efficient computation graphs.
*   **Python Integration:** Seamlessly integrate with your existing Python workflows and utilize familiar libraries like NumPy, SciPy, and Cython.
*   **Imperative Programming:**  Benefit from an intuitive and easy-to-debug imperative programming style that simplifies development.
*   **Fast and Lean:** Optimized for speed and memory efficiency, PyTorch supports large models and integrates with acceleration libraries like Intel MKL and NVIDIA cuDNN.
*   **Extensible Architecture:**  Easily create custom neural network modules and extend PyTorch's functionality with straightforward APIs.

## Core Components

PyTorch provides a modular design with key components for building and deploying machine learning models:

| Component               | Description                                                                                                   |
| :---------------------- | :------------------------------------------------------------------------------------------------------------ |
| `torch`                 | Tensor library with GPU support, similar to NumPy.                                                           |
| `torch.autograd`        | Automatic differentiation engine for calculating gradients.                                                   |
| `torch.jit`             | Compiler for creating serializable and optimized models from PyTorch code (TorchScript).                     |
| `torch.nn`              | Neural networks library with flexible building blocks.                                                       |
| `torch.multiprocessing` | Enables multiprocessing with shared memory for efficient data loading and training.                            |
| `torch.utils`           | Data loading utilities like `DataLoader` for efficient data handling.                                     |

## Installation

Get started with PyTorch quickly by following these installation steps:

### 1.  Binaries (Recommended for ease of use)

Install pre-built binaries using either:

*   **Conda:**  Follow the instructions on the [official PyTorch website](https://pytorch.org/get-started/locally/).
*   **Pip:**  Find the appropriate `pip` installation command on the [official PyTorch website](https://pytorch.org/get-started/locally/).

#### NVIDIA Jetson Platforms

Specialized wheels are available for NVIDIA Jetson platforms (Nano, TX1/TX2, Xavier NX/AGX, AGX Orin). Installation instructions and container details are linked within the original README.

### 2.  From Source (For advanced users and customization)

#### Prerequisites

*   Python 3.9 or later
*   A compiler with C++17 support (e.g., clang or gcc)
*   Visual Studio or Visual Studio Build Tool (Windows only) -  ensure you have the correct tools configured

#### CUDA, ROCm, and Intel GPU Support

*   **NVIDIA CUDA:**  Install CUDA and cuDNN. Refer to the [CUDA Support Matrix](https://pytorch.org/get-started/locally/) for version compatibility and then follow instructions in the original README.
*   **AMD ROCm:**  Install AMD ROCm and set the `ROCM_PATH` if necessary (Linux only). Follow instructions in the original README.
*   **Intel GPU:**  Follow the Intel GPU installation instructions.  Refer to the original README for more details.

#### Installation Steps

1.  **Get the PyTorch Source:**
    ```bash
    git clone https://github.com/pytorch/pytorch
    cd pytorch
    git submodule sync
    git submodule update --init --recursive
    ```

2.  **Install Dependencies:**
    ```bash
    conda install cmake ninja # or pip install cmake ninja - depends on your setup
    pip install -r requirements.txt
    ```
    - Additional steps may be required for specific OS or features (see original README).

3.  **Install PyTorch:**

    *   Linux:
        ```bash
        export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}" # (This is typically fine, even without conda - but might need some adjustments.  Make sure to use the right setup)
        python -m pip install --no-build-isolation -v -e .
        ```
    *   macOS:
        ```bash
        python -m pip install --no-build-isolation -v -e .
        ```
    *   Windows:  Follow the detailed instructions in the original README, as these are more complex.

    -  (Optional) Adjust build options by using `ccmake build` after setting CMAKE_PREFIX_PATH and CMAKE_ONLY

### 3. Docker Image

*   **Pre-built Images:** Pull pre-built images from Docker Hub:
    ```bash
    docker run --gpus all --rm -ti --ipc=host pytorch/pytorch:latest
    ```

*   **Build Your Own Image:**  Use the provided `Dockerfile`.

### 4. Building the Documentation

The README provides detailed instructions on how to build documentation.

## Getting Started & Resources

Explore the world of PyTorch with these helpful resources:

*   [Tutorials](https://pytorch.org/tutorials/)
*   [Examples](https://github.com/pytorch/examples)
*   [API Reference](https://pytorch.org/docs/)
*   [Glossary](https://github.com/pytorch/pytorch/blob/main/GLOSSARY.md)
*   Additional links to Udacity, Coursera and other resources are included in the original README.

## Communication and Community

*   **Forums:**  Discuss implementations and research: [discuss.pytorch.org](https://discuss.pytorch.org/)
*   **GitHub Issues:** Report bugs, request features: [Issue Tracker](https://github.com/pytorch/pytorch/issues)
*   **Slack:**  [PyTorch Slack](https://pytorch.slack.com/) for discussions and collaboration (requires form submission for invite)
*   **Newsletter:** Sign up for the newsletter at [eepurl.com/cbG0rv](https://eepurl.com/cbG0rv)
*   **Facebook Page:** [facebook.com/pytorch](https://www.facebook.com/pytorch)

## Releases and Contributing

PyTorch releases are typically done three times a year. For contribution guidelines, please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file.  For release information, please consult the [RELEASE.md](RELEASE.md) file.

## The Team & License

PyTorch is a community-driven project, and the original README contains links to key contributors.  PyTorch is licensed under a BSD-style license, which is available in the [LICENSE](LICENSE) file.