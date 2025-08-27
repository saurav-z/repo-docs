html
<div align="center">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="/docs/logo_tiny_light.svg">
    <img alt="tiny corp logo" src="/docs/logo_tiny_dark.svg" width="50%" height="50%">
  </picture>
</div>

# tinygrad: A Deep Learning Framework for Everyone

**tinygrad is a lightweight, fully-featured deep learning framework designed for simplicity and ease of use.**  Built to be a more accessible alternative to PyTorch, tinygrad offers a streamlined approach to building and deploying deep learning models. 
[Explore the GitHub Repository](https://github.com/tinygrad/tinygrad)

### Key Features:

*   **Simple & Efficient:** Designed for minimal complexity, making it easy to understand and extend.
*   **Accelerated Performance:** Supports a wide range of accelerators, including GPU (OpenCL, Metal, CUDA), CPU (C, LLVM), WebGPU, and more.
*   **LLaMA and Stable Diffusion Ready:** Run cutting-edge models with ease.
*   **Lazy Evaluation:** Optimizes computations through lazy evaluation, leading to efficient kernel fusion.
*   **Neural Network Support:** Provides essential components for building and training neural networks, including autograd, tensor operations, optimizers, and data loaders.
*   **Extensible Architecture:** Easily add support for new accelerators with only ~25 low-level operations.

### Get Started

**Installation:**

```bash
git clone https://github.com/tinygrad/tinygrad.git
cd tinygrad
python3 -m pip install -e .
```

**Direct (Master):**

```bash
python3 -m pip install git+https://github.com/tinygrad/tinygrad.git
```

### Core Functionality

The framework supports the following features:
*   **Laziness:** A demonstration of how operations are fused into a single kernel for performance.
*   **Neural Networks:** A practical example of training a linear network for MNIST.
*   **Accelerators:** A long list of different accelerators supported, and how to check the default accelerator.

### Documentation

Comprehensive documentation, including a quick start guide, is available on the [docs website](https://docs.tinygrad.org/).

### Example: Tinygrad vs. PyTorch

See how tinygrad mirrors PyTorch for a simple calculation with autograd:

```python
from tinygrad import Tensor

x = Tensor.eye(3, requires_grad=True)
y = Tensor([[2.0,0,-2.0]], requires_grad=True)
z = y.matmul(x).sum()
z.backward()

print(x.grad.tolist())  # dz/dx
print(y.grad.tolist())  # dz/dy
```

```python
import torch

x = torch.eye(3, requires_grad=True)
y = torch.tensor([[2.0,0,-2.0]], requires_grad=True)
z = y.matmul(x).sum()
z.backward()

print(x.grad.tolist())  # dz/dx
print(y.grad.tolist())  # dz/dy
```

### Contributing

We welcome contributions! Please review the [contribution guidelines](https://github.com/tinygrad/tinygrad/blob/master/README.md#contributing) to help with your pull request.

### Resources

*   [Homepage](https://github.com/tinygrad/tinygrad)
*   [Documentation](https://docs.tinygrad.org/)
*   [Discord](https://discord.gg/ZjZadyC7PK)
*   [tinygrad/tinygrad on GitHub](https://github.com/tinygrad/tinygrad)