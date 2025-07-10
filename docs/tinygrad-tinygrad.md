<div align="center">
<picture>
  <source media="(prefers-color-scheme: light)" srcset="/docs/logo_tiny_light.svg">
  <img alt="tiny corp logo" src="/docs/logo_tiny_dark.svg" width="50%" height="50%">
</picture>
</div>

# tinygrad: A Minimal Deep Learning Framework for Maximum Flexibility

**Tinygrad is a surprisingly capable deep learning framework, offering a lightweight and flexible alternative to larger frameworks.**  [Explore the code on GitHub](https://github.com/tinygrad/tinygrad)!

### [Homepage](https://github.com/tinygrad/tinygrad) | [Documentation](https://docs.tinygrad.org/) | [Discord](https://discord.gg/ZjZadyC7PK)

[![GitHub Repo stars](https://img.shields.io/github/stars/tinygrad/tinygrad)](https://github.com/tinygrad/tinygrad/stargazers)
[![Unit Tests](https://github.com/tinygrad/tinygrad/actions/workflows/test.yml/badge.svg)](https://github.com/tinygrad/tinygrad/actions/workflows/test.yml)
[![Discord](https://img.shields.io/discord/1068976834382925865)](https://discord.gg/ZjZadyC7PK)

---

## Key Features of tinygrad

*   **Lightweight and Efficient:** Tinygrad's simplicity makes it easy to understand and extend, while still offering impressive performance.
*   **Easy Accelerator Support:** Quickly add support for new hardware by implementing approximately 25 low-level operations.
*   **LLaMA and Stable Diffusion Support:** Run complex models with minimal code.
*   **Lazy Evaluation:** Experience performance gains through the power of lazy evaluation.
*   **Neural Network Capabilities:** Build and train neural networks with autograd, tensor libraries, optimizers, and data loaders.
*   **Extensive Hardware Support:** Ready-to-go support for a wide range of accelerators.
    *   [x] GPU (OpenCL)
    *   [x] CPU (C Code)
    *   [x] LLVM
    *   [x] METAL
    *   [x] CUDA
    *   [x] AMD
    *   [x] NV
    *   [x] QCOM
    *   [x] WEBGPU

## Installation

The recommended method to install tinygrad is from source:

### From Source

```bash
git clone https://github.com/tinygrad/tinygrad.git
cd tinygrad
python3 -m pip install -e .
```

### Direct (Master)

```bash
python3 -m pip install git+https://github.com/tinygrad/tinygrad.git
```

## Documentation

Comprehensive documentation, including a quick start guide, is available on the [docs website](https://docs.tinygrad.org/).

### Quick Example: Tinygrad vs. PyTorch

```python
# Example comparing the syntax with PyTorch
from tinygrad import Tensor

x = Tensor.eye(3, requires_grad=True)
y = Tensor([[2.0,0,-2.0]], requires_grad=True)
z = y.matmul(x).sum()
z.backward()

print(x.grad.tolist())  # dz/dx
print(y.grad.tolist())  # dz/dy
```

```python
# The equivalent code with PyTorch:
import torch

x = torch.eye(3, requires_grad=True)
y = torch.tensor([[2.0,0,-2.0]], requires_grad=True)
z = y.matmul(x).sum()
z.backward()

print(x.grad.tolist())  # dz/dx
print(y.grad.tolist())  # dz/dy
```

## Contributing

We welcome contributions! Please review the [Contributing Guidelines](https://github.com/tinygrad/tinygrad#contributing) for details on how to contribute effectively.