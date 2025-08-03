<div align="center">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="/docs/logo_tiny_light.svg">
    <img alt="tiny corp logo" src="/docs/logo_tiny_dark.svg" width="50%" height="50%">
  </picture>
</div>

# tinygrad: A Deep Learning Framework for RISC Architectures

[tinygrad](https://github.com/tinygrad/tinygrad) is a deep learning framework designed for simplicity, efficiency, and extensibility, making it a great alternative to PyTorch or micrograd. Developed by [tiny corp](https://tinygrad.org).

**Key Features:**

*   **Lightweight and Simple:** Tinygrad is designed to be easy to understand and extend, making it ideal for learning and experimentation.
*   **Multiple Accelerator Support:** Supports a wide range of hardware, including GPU (OpenCL, METAL, CUDA, AMD, NV, QCOM, WEBGPU), CPU, and LLVM. Easily add support for your own accelerator!
*   **Runs LLaMA and Stable Diffusion:** Demonstrate its capabilities by running complex models.
*   **Lazy Evaluation:** Efficiently fuses operations into optimized kernels.
*   **Neural Network Capabilities:** Built-in autograd, tensor library, optimizer, and dataloader functionality for building and training neural networks.

**Quick Start:**

```python
from tinygrad import Tensor

x = Tensor.eye(3, requires_grad=True)
y = Tensor([[2.0,0,-2.0]], requires_grad=True)
z = y.matmul(x).sum()
z.backward()

print(x.grad.tolist())  # dz/dx
print(y.grad.tolist())  # dz/dy
```

**Install:**

```bash
git clone https://github.com/tinygrad/tinygrad.git
cd tinygrad
python3 -m pip install -e .
```

**Explore the Docs:**  For in-depth information and examples, visit the [documentation website](https://docs.tinygrad.org/).

**Join the Community:**

*   [Discord](https://discord.gg/ZjZadyC7PK)

[![GitHub Repo stars](https://img.shields.io/github/stars/tinygrad/tinygrad)](https://github.com/tinygrad/tinygrad/stargazers)
[![Unit Tests](https://github.com/tinygrad/tinygrad/actions/workflows/test.yml/badge.svg)](https://github.com/tinygrad/tinygrad/actions/workflows/test.yml)
[![Discord](https://img.shields.io/discord/1068976834382925865)](https://discord.gg/ZjZadyC7PK)

**Contribute:**

We welcome contributions! Please review the [Contribution Guidelines](https://github.com/tinygrad/tinygrad#contributing) for details on submitting pull requests, bug fixes, and feature additions.