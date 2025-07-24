html
<div align="center">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="/docs/logo_tiny_light.svg">
    <img alt="tiny corp logo" src="/docs/logo_tiny_dark.svg" width="50%" height="50%">
  </picture>
</div>

# tinygrad: A Deep Learning Framework That's Surprisingly Powerful

[tinygrad](https://github.com/tinygrad/tinygrad) is a minimalist deep learning framework designed for simplicity and flexibility, offering a compelling alternative to PyTorch and micrograd. Developed by [tiny corp](https://tinygrad.org/).

### Key Features

*   **Runs LLaMA and Stable Diffusion:** Build and deploy state-of-the-art models with ease.
*   **Lazy Evaluation:** Experience efficient computation with intelligent kernel fusion, optimizing performance.
*   **Neural Network Capabilities:** Build and train neural networks with autograd, tensor operations, and built-in optimizers.
*   **Multi-Accelerator Support:** Supports a wide range of accelerators including GPU (OpenCL), CPU (C Code), LLVM, METAL, CUDA, AMD, NV, QCOM, and WEBGPU, with easy extensibility.

### Installation

```bash
git clone https://github.com/tinygrad/tinygrad.git
cd tinygrad
python3 -m pip install -e .
```

### Documentation

Dive deeper into tinygrad's capabilities with comprehensive documentation available on the [docs website](https://docs.tinygrad.org/).

### Contributing

Help us improve tinygrad!  Refer to the original repo for [Contributing Guidelines](https://github.com/tinygrad/tinygrad#contributing) and [Bounties](https://docs.google.com/spreadsheets/d/1WKHbT-7KOgjEawq5h5Ic1qUWzpfAzuD_J06N1JwOCGs/edit?usp=sharing).

[![GitHub Repo stars](https://img.shields.io/github/stars/tinygrad/tinygrad)](https://github.com/tinygrad/tinygrad/stargazers)
[![Unit Tests](https://github.com/tinygrad/tinygrad/actions/workflows/test.yml/badge.svg)](https://github.com/tinygrad/tinygrad/actions/workflows/test.yml)
[![Discord](https://img.shields.io/discord/1068976834382925865)](https://discord.gg/ZjZadyC7PK)