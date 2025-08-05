<div align="center">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="/docs/logo_tiny_light.svg">
    <img alt="tiny corp logo" src="/docs/logo_tiny_dark.svg" width="50%" height="50%">
  </picture>
</div>

# tinygrad: A Tiny Deep Learning Framework for Maximum Flexibility

[tinygrad](https://github.com/tinygrad/tinygrad) is a lightweight, fully-featured deep learning framework, offering a unique balance of simplicity and power, making it the ideal choice for researchers and developers seeking unparalleled control over their deep learning models.  

**Key Features:**

*   **Lightweight & Flexible:** Designed for ease of use and customization, allowing you to quickly add support for new accelerators.
*   **Accelerated Performance:** Supports a wide range of accelerators including CPU, GPU (OpenCL, Metal, CUDA, AMD, NV, QCOM, WEBGPU), and LLVM, with more easily added.
*   **Runs LLaMA and Stable Diffusion:**  Experience the power of large language models and image generation with tinygrad.
*   **Lazy Evaluation:**  Optimize computations by automatically fusing operations into efficient kernels.
*   **Neural Network Capabilities:**  Provides essential building blocks for neural networks, including autograd, a tensor library, optimizers, and data loaders.
*   **Easy to Install:** Simple installation from source or directly via pip.
*   **Similar API to PyTorch:**  Easily transition from popular frameworks like PyTorch for a simplified experience.

**Why Choose tinygrad?**

Unlike frameworks like PyTorch, tinygrad's RISC-like architecture makes it easier to understand and modify, ideal for research or working on new hardware.

**Quickstart Example:**

```python
from tinygrad import Tensor

x = Tensor.eye(3, requires_grad=True)
y = Tensor([[2.0,0,-2.0]], requires_grad=True)
z = y.matmul(x).sum()
z.backward()

print(x.grad.tolist())  # dz/dx
print(y.grad.tolist())  # dz/dy
```

**Installation:**

```bash
git clone https://github.com/tinygrad/tinygrad.git
cd tinygrad
python3 -m pip install -e .
```
or
```bash
python3 -m pip install git+https://github.com/tinygrad/tinygrad.git
```

**Documentation:**

Comprehensive documentation and a quick start guide are available on the [docs website](https://docs.tinygrad.org/).

**Contributing:**

We welcome contributions! Please review the [contributing guidelines](https://github.com/tinygrad/tinygrad#contributing) to ensure your pull request is accepted.

[![GitHub Repo stars](https://img.shields.io/github/stars/tinygrad/tinygrad)](https://github.com/tinygrad/tinygrad/stargazers)
[![Unit Tests](https://github.com/tinygrad/tinygrad/actions/workflows/test.yml/badge.svg)](https://github.com/tinygrad/tinygrad/actions/workflows/test.yml)
[![Discord](https://img.shields.io/discord/1068976834382925865)](https://discord.gg/ZjZadyC7PK)