![PyTorch Logo](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/pytorch-logo-dark.png)

# PyTorch: The Deep Learning Framework for Research and Production

**PyTorch empowers researchers and developers to build cutting-edge machine learning models with flexibility and speed.**  Discover the power of PyTorch, an open-source deep learning framework built for Python, designed to accelerate research and simplify production deployment. ([Original Repository](https://github.com/pytorch/pytorch))

## Key Features of PyTorch

*   **GPU-Accelerated Tensor Computation:** Leverage the power of GPUs for blazing-fast tensor operations, similar to NumPy, accelerating your scientific computing.
*   **Dynamic Neural Networks with Autograd:** Build and modify neural networks with unparalleled flexibility using a tape-based automatic differentiation system.
*   **Python-First Approach:** Integrate seamlessly with your existing Python workflow, including NumPy, SciPy, and other libraries.
*   **Intuitive and Imperative Design:** Enjoy an easy-to-use, imperative programming style that simplifies debugging and understanding.
*   **Fast and Lean:** Benefit from minimal framework overhead and optimized performance with Intel MKL, cuDNN, and NCCL integrations.
*   **Easy Extension and Customization:** Effortlessly create new neural network modules and extend PyTorch with your own Python or C/C++ code.

## Installation

Find detailed installation instructions tailored to your operating system and hardware configuration on [our website](https://pytorch.org/get-started/locally/).  Choose your preferred method:

### Binaries
Install pre-built packages via Conda or pip wheels.

#### NVIDIA Jetson Platforms

Pre-built wheels are available for NVIDIA Jetson platforms.  See installation details [here](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048).

### From Source

Follow these steps for a source installation:

#### Prerequisites
*   Python 3.9 or later
*   A C++17-compliant compiler (gcc 9.4.0 or newer recommended for Linux)
*   Visual Studio or Visual Studio Build Tool (Windows only)

#### NVIDIA CUDA Support (Optional)
*   Install [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads), [cuDNN](https://developer.nvidia.com/cudnn) (v8.5+), and a compatible [compiler](https://gist.github.com/ax3l/9489132).
*   Disable CUDA support with `USE_CUDA=0`.

#### AMD ROCm Support (Optional)
*   Install [AMD ROCm](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html) (4.0+).
*   Set `ROCM_PATH` if ROCm is not in the default location.  Optionally, set `PYTORCH_ROCM_ARCH`.
*   Disable ROCm support with `USE_ROCM=0`.

#### Intel GPU Support (Optional)
*   Follow the [PyTorch Prerequisites for Intel GPUs](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpus.html).
*   Disable Intel GPU support with `USE_XPU=0`.

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
    conda install cmake ninja  # or pip install cmake ninja
    pip install -r requirements.txt
    ```

    *   **Linux:**
        ```bash
        pip install mkl-static mkl-include
        # CUDA only (Optional):
        .ci/docker/common/install_magma_conda.sh 12.4 #specify CUDA version if using conda.
        # (Optional)
        make triton #install Triton, which is used by torch.compile
        ```
    *   **macOS:**
        ```bash
        pip install mkl-static mkl-include # for Intel x86
        conda install pkg-config libuv  # if torch.distributed is needed
        ```
    *   **Windows:**
        ```bash
        pip install mkl-static mkl-include
        conda install -c conda-forge libuv=1.39 # if torch.distributed is needed
        ```

3.  **Install PyTorch:**

    *   **Linux/macOS:**
        ```bash
        export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"
        python -m pip install --no-build-isolation -v -e .
        ```
    *   **Windows:**
        ```cmd
        python -m pip install --no-build-isolation -v -e .
        ```
    *   **CPU-only builds:**
        ```cmd
        python -m pip install --no-build-isolation -v -e .
        ```
    *   **CUDA-based Build** (Windows): Requires additional configuration and installation. Refer to the README for specific instructions.
    *   **Intel GPU builds** (Windows):  Requires specific prerequisites and build commands; see the original README.

#### Adjust Build Options (Optional)
Adjust build configurations with `ccmake build` or `cmake-gui build` after setting environment variables.

### Docker Image

Leverage pre-built Docker images or build your own:

#### Using pre-built images
Run prebuilt images from Docker Hub. For example:
```bash
docker run --gpus all --rm -ti --ipc=host pytorch/pytorch:latest
```

#### Building the image yourself
Build Docker images with CUDA 11.1 support using `make -f docker.Makefile`.

### Building the Documentation

Build documentation locally using Sphinx:

```bash
cd docs/
pip install -r requirements.txt
make html
make serve
```

#### Building a PDF
Build the PDF documentation using `make latexpdf` and `make LATEXOPTS="-interaction=nonstopmode"`.

### Previous Versions
See [our website](https://pytorch.org/get-started/previous-versions) for older versions.

## Getting Started

*   [Tutorials](https://pytorch.org/tutorials/): Master the fundamentals of PyTorch.
*   [Examples](https://github.com/pytorch/examples): Explore practical PyTorch code across diverse domains.
*   [API Reference](https://pytorch.org/docs/): Consult the comprehensive API documentation.

## Resources

*   [PyTorch.org](https://pytorch.org/)
*   [PyTorch Tutorials](https://pytorch.org/tutorials/)
*   [PyTorch Examples](https://github.com/pytorch/examples)
*   [PyTorch Models](https://pytorch.org/hub/)
*   [And more...](see original README)

## Communication

*   **Forums:** Engage in discussions and seek support: [https://discuss.pytorch.org](https://discuss.pytorch.org)
*   **GitHub Issues:** Report bugs and request features.
*   **Slack:** Join the PyTorch Slack community (apply via form for invite).
*   **Newsletter:** Stay updated with the no-noise PyTorch newsletter.
*   **Social Media:** Follow PyTorch on Facebook and Twitter.

## Releases and Contributing

PyTorch typically releases three minor versions annually.  Report bugs via [GitHub Issues](https://github.com/pytorch/pytorch/issues).  Contributions are welcome; please discuss new features beforehand.  Consult the [Contribution page](CONTRIBUTING.md) and [Release page](RELEASE.md).

## The Team

PyTorch is a community project driven by talented engineers and researchers. The core maintainers are listed, along with significant contributors.

## License

PyTorch is licensed under a BSD-style license; see the [LICENSE](LICENSE) file.