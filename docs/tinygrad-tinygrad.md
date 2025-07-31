<div align="center">

<picture>
  <source media="(prefers-color-scheme: light)" srcset="/docs/logo_tiny_light.svg">
  <img alt="tiny corp logo" src="/docs/logo_tiny_dark.svg" width="50%" height="50%">
</picture>

</div>

# tinygrad: A Tiny, Fully-Featured Deep Learning Framework

**tinygrad is a minimalist deep learning framework, offering a powerful alternative to PyTorch and micrograd, designed for simplicity and ease of use.** This project, maintained by [tiny corp](https://tinygrad.org/), provides a streamlined approach to deep learning, making it ideal for both research and practical applications.

[View the original repository on GitHub](https://github.com/tinygrad/tinygrad)

### Key Features

*   **Fully-Featured:** Despite its small size, tinygrad supports complex deep learning tasks.
*   **Easy Accelerator Support:** Designed for easy implementation of new accelerators, enabling both inference and training.
*   **LLaMA and Stable Diffusion Support:** Run advanced models with ease.
*   **Laziness:** Experience efficient computation through kernel fusion and optimization.
*   **Neural Network Capabilities:** Build and train neural networks with a simple autograd/tensor library, optimizer, and data loading support.
*   **Extensive Accelerator Support:**
    *   GPU (OpenCL)
    *   CPU (C Code)
    *   LLVM
    *   METAL
    *   CUDA
    *   AMD
    *   NV
    *   QCOM
    *   WEBGPU
*   **Simplified Installation:** Install easily from source or directly via pip.

## Installation

### From Source

```bash
git clone https://github.com/tinygrad/tinygrad.git
cd tinygrad
python3 -m pip install -e .
```

### Direct (master)

```bash
python3 -m pip install git+https://github.com/tinygrad/tinygrad.git
```

## Documentation

Comprehensive documentation and a quick start guide are available on the [docs website](https://docs.tinygrad.org/).

## Contributing

Contributions are welcome! Please review the [contributing guidelines](https://github.com/tinygrad/tinygrad#contributing) before submitting pull requests.