[![PyTorch Logo](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/pytorch-logo-dark.png)](https://github.com/pytorch/pytorch)

# PyTorch: Deep Learning with Python

**PyTorch empowers researchers and developers to build cutting-edge machine learning models with flexibility and speed.**  Dive into the world of PyTorch, a powerful and versatile Python-based machine learning framework, and [explore its source code on GitHub](https://github.com/pytorch/pytorch).

## Key Features

*   **Tensor Computation with GPU Acceleration:** Leverage the power of GPUs for blazing-fast tensor operations, similar to NumPy but optimized for performance.
*   **Dynamic Neural Networks:** Build and modify neural networks with unparalleled flexibility using a tape-based autograd system for dynamic computation graphs.
*   **Python-First Design:** Seamlessly integrate with your existing Python workflow and leverage the rich ecosystem of Python libraries.
*   **Imperative Programming:** Experience an intuitive and easy-to-debug development process with imperative style.
*   **Fast and Lean:** Benefit from minimal framework overhead and optimized acceleration libraries for both CPU and GPU.
*   **Easy Extensibility:** Effortlessly extend PyTorch with custom layers and integrate with other Python libraries.

## Core Components

PyTorch is built from the following core components:

*   [**torch**](https://pytorch.org/docs/stable/torch.html): The tensor library, similar to NumPy, but with GPU support.
*   [**torch.autograd**](https://pytorch.org/docs/stable/autograd.html): The automatic differentiation library supporting all differentiable tensor operations.
*   [**torch.jit**](https://pytorch.org/docs/stable/jit.html): A compilation stack (TorchScript) to create serializable and optimizable models from PyTorch code.
*   [**torch.nn**](https://pytorch.org/docs/stable/nn.html): The neural networks library, designed for flexibility, deeply integrated with autograd.
*   [**torch.multiprocessing**](https://pytorch.org/docs/stable/multiprocessing.html): Python multiprocessing with magical memory sharing of torch tensors across processes, crucial for data loading and training.
*   [**torch.utils**](https://pytorch.org/docs/stable/data.html): DataLoader and other convenient utility functions.

## Installation

Get started quickly with PyTorch through the following methods:

### Binaries

Install PyTorch using pre-built binaries. Commands for Conda and pip wheels are available on the [PyTorch website](https://pytorch.org/get-started/locally/).

#### NVIDIA Jetson Platforms

Install wheels for NVIDIA's Jetson Nano, TX1/TX2, Xavier NX/AGX, and AGX Orin. Instructions [available here](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048) and [here](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch)

### From Source

Build and install PyTorch from source for greater customization.

#### Prerequisites

Ensure you have:
*   Python 3.9 or later
*   A C++17-compliant compiler (e.g., GCC 9.4.0 or newer on Linux)
*   Microsoft Visual Studio or Build Tools (Windows only)

Refer to the provided example environment setup for both Linux and Windows.

##### NVIDIA CUDA Support

1.  **CUDA:** Install a [supported version of CUDA](https://pytorch.org/get-started/locally/).
2.  **cuDNN:** Install [cuDNN](https://developer.nvidia.com/cudnn) v8.5 or above.
3.  **Compiler:** Ensure you have a [CUDA-compatible compiler](https://gist.github.com/ax3l/9489132).

  You can disable CUDA support by setting `USE_CUDA=0`.

##### AMD ROCm Support

1.  **ROCm:** Install [AMD ROCm](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html) 4.0 and above.

  You can disable ROCm support by setting `USE_ROCM=0`.

##### Intel GPU Support

1.  Follow the [PyTorch prerequisites for Intel GPUs](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpus.html) instructions.

  You can disable Intel GPU support by setting `USE_XPU=0`.

#### Steps
1.  **Get the Source:**
```bash
git clone https://github.com/pytorch/pytorch
cd pytorch
git submodule sync
git submodule update --init --recursive
```

2.  **Install Dependencies:**
```bash
conda install cmake ninja # Recommended, if using conda
pip install -r requirements.txt # Run this from the PyTorch directory after cloning the source code
```

3.  **Install PyTorch:**

    *   **Linux (with ROCm):**
        ```bash
        # Only run this if you're compiling for ROCm
        python tools/amd_build/build_amd.py
        export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}" # Conda users
        python setup.py develop
        ```

    *   **macOS:**
        ```bash
        python3 setup.py develop
        ```

    *   **Windows:**
        ```cmd
        python setup.py develop
        ```

    *   **CPU-Only Builds:** If you're using a CPU, run:
        ```cmd
        python setup.py develop
        ```

4. **Adjust Build Options (Optional):**
    ```bash
    export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}" # Conda users
    CMAKE_ONLY=1 python setup.py build
    ccmake build  # or cmake-gui build
    ```
    (For macOS and others:  `MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ CMAKE_ONLY=1 python setup.py build`)

### Docker Image

Simplify development with pre-built or custom Docker images.

#### Using pre-built images
```bash
docker run --gpus all --rm -ti --ipc=host pytorch/pytorch:latest
```
#### Building the image yourself
```bash
make -f docker.Makefile
```

### Building the Documentation

Build documentation to explore the details of PyTorch

```bash
cd docs/
pip install -r requirements.txt
make html
make serve
```
### Previous Versions

Find installation instructions for previous PyTorch versions on [our website](https://pytorch.org/get-started/previous-versions).

## Getting Started

Explore [tutorials](https://pytorch.org/tutorials/), [examples](https://github.com/pytorch/examples), and [API reference](https://pytorch.org/docs/) to kickstart your PyTorch journey.  Refer to the [Glossary](https://github.com/pytorch/pytorch/blob/main/GLOSSARY.md)

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

*   **Forums:** [discuss.pytorch.org](https://discuss.pytorch.org)
*   **GitHub Issues:** Bug reports, feature requests, etc.
*   **Slack:** The [PyTorch Slack](https://pytorch.slack.com/). If you need a slack invite, please fill this form: https://goo.gl/forms/PP1AGvNHpSaJP8to1
*   **Newsletter:** Subscribe to receive PyTorch announcements via email:  https://eepurl.com/cbG0rv
*   **Facebook Page:** Stay updated on Facebook:  https://www.facebook.com/pytorch

## Releases and Contributing

PyTorch typically releases three minor versions annually.  Report bugs through [GitHub issues](https://github.com/pytorch/pytorch/issues).

We welcome contributions.  For new features or major changes, open an issue to discuss your ideas before submitting a pull request. See our [Contribution page](CONTRIBUTING.md) and [Release page](RELEASE.md).

## The Team

PyTorch is a community project maintained by a dedicated team: [Soumith Chintala](http://soumith.ch), [Gregory Chanan](https://github.com/gchanan), [Dmytro Dzhulgakov](https://github.com/dzhulgakov), [Edward Yang](https://github.com/ezyang), and [Nikita Shulga](https://github.com/malfet).

## License

PyTorch is licensed under a BSD-style license, found in the [LICENSE](LICENSE) file.