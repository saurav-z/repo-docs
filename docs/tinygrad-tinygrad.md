<div align="center">

<picture>
  <source media="(prefers-color-scheme: light)" srcset="/docs/logo_tiny_light.svg">
  <img alt="tiny corp logo" src="/docs/logo_tiny_dark.svg" width="50%" height="50%">
</picture>

</div>

# tinygrad: A Tiny Deep Learning Framework for Maximum Flexibility

**tinygrad** is a minimalist deep learning framework designed for simplicity and ease of adding new hardware support. It's the perfect choice for researchers and developers looking for a lean, customizable, and highly efficient deep learning solution.

[**Explore tinygrad on GitHub**](https://github.com/tinygrad/tinygrad) | [Documentation](https://docs.tinygrad.org/) | [Discord](https://discord.gg/ZjZadyC7PK)

[![GitHub Repo stars](https://img.shields.io/github/stars/tinygrad/tinygrad)](https://github.com/tinygrad/tinygrad/stargazers)
[![Unit Tests](https://github.com/tinygrad/tinygrad/actions/workflows/test.yml/badge.svg)](https://github.com/tinygrad/tinygrad/actions/workflows/test.yml)
[![Discord](https://img.shields.io/discord/1068976834382925865)](https://discord.gg/ZjZadyC7PK)

---

## Key Features of tinygrad:

*   **Ultra-lightweight Design:** Built for simplicity, tinygrad is easy to understand and extend.
*   **Accelerated Performance:** Supports multiple accelerators, including GPU (OpenCL, Metal, CUDA, AMD, NV, QCOM, WEBGPU), CPU (C Code, LLVM).
*   **Lazy Evaluation:** Experience efficient computation through lazy evaluation, fusing operations into optimized kernels.
*   **Neural Network Support:**  Easily build and train neural networks with autograd, a tensor library, and an optimizer.
*   **LLaMA and Stable Diffusion Support:** Run cutting-edge models with a streamlined approach.
*   **Simple to Extend:** Easily add support for new hardware by implementing ~25 low-level operations.

## Get Started with tinygrad

### Installation

Install tinygrad from source with the following commands:

```bash
git clone https://github.com/tinygrad/tinygrad.git
cd tinygrad
python3 -m pip install -e .
```

Or install directly from master:

```bash
python3 -m pip install git+https://github.com/tinygrad/tinygrad.git
```

## Example:  A Quick Comparison with PyTorch

See the difference in a simple matrix operation:

```python
from tinygrad import Tensor

x = Tensor.eye(3, requires_grad=True)
y = Tensor([[2.0,0,-2.0]], requires_grad=True)
z = y.matmul(x).sum()
z.backward()

print(x.grad.tolist())  # dz/dx
print(y.grad.tolist())  # dz/dy
```

Compared to PyTorch:

```python
import torch

x = torch.eye(3, requires_grad=True)
y = torch.tensor([[2.0,0,-2.0]], requires_grad=True)
z = y.matmul(x).sum()
z.backward()

print(x.grad.tolist())  # dz/dx
print(y.grad.tolist())  # dz/dy
```

## Contributing

We welcome contributions!  Please review the [contributing guidelines](https://github.com/tinygrad/tinygrad/blob/master/README.md#contributing) before submitting a pull request.

##  More Resources

*   [Documentation](https://docs.tinygrad.org/)
*   [Discord](https://discord.gg/ZjZadyC7PK)
*   [tinygrad on GitHub](https://github.com/tinygrad/tinygrad)