<div align="center">

<picture>
  <source media="(prefers-color-scheme: light)" srcset="/docs/logo_tiny_light.svg">
  <img alt="tiny corp logo" src="/docs/logo_tiny_dark.svg" width="50%" height="50%">
</picture>

</div>

# tinygrad: A Deep Learning Framework for Everyone

**tinygrad is a fully-featured deep learning framework designed for simplicity, offering a streamlined approach to building and deploying neural networks.**  

[View the original repository on GitHub](https://github.com/tinygrad/tinygrad)

### Key Features:

*   **Lightweight and Efficient:** Tinygrad's design prioritizes simplicity, making it easy to understand, modify, and extend.
*   **Accelerated Performance:**  Supports a growing list of accelerators for both inference and training, including:
    *   GPU (OpenCL)
    *   CPU (C Code)
    *   LLVM
    *   METAL
    *   CUDA
    *   AMD
    *   NV
    *   QCOM
    *   WEBGPU
*   **Easy to Extend:**  Adding support for new accelerators is straightforward, requiring only ~25 low-level operations.
*   **Runs Cutting-Edge Models:** Execute complex models like [LLaMA](/docs/showcase.md#llama) and [Stable Diffusion](/docs/showcase.md#stable-diffusion) with ease.
*   **Lazy Evaluation:**  Benefit from kernel fusion through lazy evaluation for optimized performance.
*   **Neural Network Capabilities:**  Offers the essential components for building and training neural networks, including autograd, tensor operations, optimizers, and data loaders.
*   **Simple API:**  Provides a clean and intuitive API for defining and executing tensor computations.

### Key Links:

*   [Homepage](https://github.com/tinygrad/tinygrad)
*   [Documentation](https://docs.tinygrad.org/)
*   [Discord](https://discord.gg/ZjZadyC7PK)

[![GitHub Repo stars](https://img.shields.io/github/stars/tinygrad/tinygrad)](https://github.com/tinygrad/tinygrad/stargazers)
[![Unit Tests](https://github.com/tinygrad/tinygrad/actions/workflows/test.yml/badge.svg)](https://github.com/tinygrad/tinygrad/actions/workflows/test.yml)
[![Discord](https://img.shields.io/discord/1068976834382925865)](https://discord.gg/ZjZadyC7PK)

---

### Installation

**Recommended:** Install from source:

```bash
git clone https://github.com/tinygrad/tinygrad.git
cd tinygrad
python3 -m pip install -e .
```

**Alternative:** Install directly from the master branch:

```bash
python3 -m pip install git+https://github.com/tinygrad/tinygrad.git
```

### Example: Tensor Operations

```python
from tinygrad import Tensor

x = Tensor.eye(3, requires_grad=True)
y = Tensor([[2.0,0,-2.0]], requires_grad=True)
z = y.matmul(x).sum()
z.backward()

print(x.grad.tolist())  # dz/dx
print(y.grad.tolist())  # dz/dy
```

### Contributing

Contributions are welcome!  Please review the [Contribution Guidelines](https://github.com/tinygrad/tinygrad#contributing) before submitting a pull request.