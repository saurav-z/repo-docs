<a href="https://github.com/pytorch/pytorch">
    <img src="https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/pytorch-logo-dark.png" alt="PyTorch Logo" width="300">
</a>

# PyTorch: Deep Learning with GPU Acceleration

**PyTorch is an open-source machine learning framework that accelerates the path from research prototyping to production deployment.**  ([Original Repository](https://github.com/pytorch/pytorch))

## Key Features

*   **Tensor Computation with GPU Acceleration:** Leverage the power of GPUs for fast tensor operations, similar to NumPy but significantly faster.
*   **Dynamic Neural Networks:** Build and modify neural networks with unparalleled flexibility using PyTorch's tape-based autograd system.
*   **Python-First Design:** Seamlessly integrate PyTorch with your existing Python workflows and favorite libraries (NumPy, SciPy, etc.).
*   **Intuitive and Imperative:** Experience a clear, straightforward programming model that simplifies debugging and understanding.
*   **Fast and Lean:** Benefit from a minimal framework overhead and efficient memory management, optimized for speed.
*   **Easy Extensibility:** Effortlessly create custom neural network modules and integrate with PyTorch's Tensor API.

## Core Components

PyTorch is built from the following core components:

*   [`torch`](https://pytorch.org/docs/stable/torch.html): Tensor library with GPU support.
*   [`torch.autograd`](https://pytorch.org/docs/stable/autograd.html): Tape-based automatic differentiation.
*   [`torch.jit`](https://pytorch.org/docs/stable/jit.html): Compilation stack (TorchScript) for serializable and optimizable models.
*   [`torch.nn`](https://pytorch.org/docs/stable/nn.html): Neural networks library with autograd integration.
*   [`torch.multiprocessing`](https://pytorch.org/docs/stable/multiprocessing.html): Memory sharing for Tensors across processes.
*   [`torch.utils`](https://pytorch.org/docs/stable/data.html): DataLoader and other utility functions.

## Installation

Choose your preferred installation method:

### Binaries

Install binaries using Conda or pip wheels.  See [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) for specific commands.

#### NVIDIA Jetson Platforms

Pre-built wheels are available for Jetson Nano, TX1/TX2, Xavier NX/AGX, and AGX Orin.  See [here](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048).

### From Source

#### Prerequisites

*   Python 3.9 or later
*   A C++17-compliant compiler (e.g., GCC 9.4.0+)
*   Visual Studio or Visual Studio Build Tool (Windows)

Ensure your environment is properly set up (Linux example below):

```bash
$ source <CONDA_INSTALL_DIR>/bin/activate
$ conda create -y -n <CONDA_NAME>
$ conda activate <CONDA_NAME>
```

##### NVIDIA CUDA Support
[Select a supported version of CUDA from our support matrix](https://pytorch.org/get-started/locally/), then install CUDA and cuDNN.  See the original README for detailed instructions.

##### AMD ROCm Support
Install AMD ROCm 4.0 or later.  See the original README for detailed instructions.

##### Intel GPU Support
Follow the instructions at [PyTorch Prerequisites for Intel GPUs](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpus.html).

#### Steps

1.  Get the PyTorch Source:

    ```bash
    git clone https://github.com/pytorch/pytorch
    cd pytorch
    git submodule sync
    git submodule update --init --recursive
    ```

2.  Install Dependencies:

    ```bash
    conda install cmake ninja # or pip install cmake ninja
    pip install -r requirements.txt
    ```

    **(Linux) Also:**

    ```bash
    pip install mkl-static mkl-include
    # CUDA only
    .ci/docker/common/install_magma_conda.sh 12.4
    # (optional) Triton installation (see original README)
    make triton
    ```

    **(macOS) Also:**

    ```bash
    pip install mkl-static mkl-include # x86 only
    conda install pkg-config libuv # for torch.distributed
    ```

    **(Windows) Also:**

    ```bash
    pip install mkl-static mkl-include
    conda install -c conda-forge libuv=1.39 # for torch.distributed
    ```

3.  Install PyTorch:

    **(Linux) ROCm:**

    ```bash
    # Only run this if you're compiling for ROCm
    python tools/amd_build/build_amd.py
    ```

    **(All):**

    ```bash
    export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"
    python setup.py develop
    ```
    **(macOS):**

     ```bash
     python3 setup.py develop
     ```

     **(Windows):** follow the guide in the original README

##### Adjust Build Options (Optional)
Use `CMAKE_ONLY=1 python setup.py build` followed by `ccmake build` or `cmake-gui build`. See original README for more detail.

### Docker Image

#### Using Pre-built Images

```bash
docker run --gpus all --rm -ti --ipc=host pytorch/pytorch:latest
```

#### Building Your Own Image

```bash
make -f docker.Makefile
```

## Getting Started

*   [Tutorials](https://pytorch.org/tutorials/): Learn the basics and how to use PyTorch.
*   [Examples](https://github.com/pytorch/examples): Easy-to-understand PyTorch code across all domains.
*   [API Reference](https://pytorch.org/docs/): Detailed documentation of the PyTorch API.
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

*   [Forums](https://discuss.pytorch.org): Discuss implementations, research, and more.
*   [GitHub Issues](https://github.com/pytorch/pytorch/issues): Bug reports, feature requests, installation issues, and more.
*   [Slack](https://pytorch.slack.com/): For experienced PyTorch users and developers (request invite via form in original README).
*   [Newsletter](https://eepurl.com/cbG0rv): Announcements about PyTorch.
*   [Facebook Page](https://www.facebook.com/pytorch): Important announcements.
*   [Brand Guidelines](https://pytorch.org/): Official branding information.

## Releases and Contributing

PyTorch typically releases three minor versions per year.  Report bugs via [GitHub Issues](https://github.com/pytorch/pytorch/issues). Contribute new features by first opening an issue to discuss the feature. See [CONTRIBUTING.md](CONTRIBUTING.md) and [RELEASE.md](RELEASE.md) for details.

## The Team

PyTorch is a community-driven project. The project is currently maintained by [Soumith Chintala](http://soumith.ch), [Gregory Chanan](https://github.com/gchanan), [Dmytro Dzhulgakov](https://github.com/dzhulgakov), [Edward Yang](https://github.com/ezyang), and [Nikita Shulga](https://github.com/malfet) with major contributions from many others.

## License

PyTorch is licensed under a BSD-style license. See the [LICENSE](LICENSE) file.