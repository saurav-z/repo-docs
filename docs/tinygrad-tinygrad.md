<div align="center">

<picture>
  <source media="(prefers-color-scheme: light)" srcset="/docs/logo_tiny_light.svg">
  <img alt="tiny corp logo" src="/docs/logo_tiny_dark.svg" width="50%" height="50%">
</picture>

</div>

# tinygrad: A Deep Learning Framework for RISC Architectures

**tinygrad** is a lightweight deep learning framework, offering a streamlined approach to building and deploying neural networks, perfect for those seeking simplicity and efficiency. [Explore the tinygrad project on GitHub](https://github.com/tinygrad/tinygrad).

---

**Key Features of tinygrad:**

*   **Lightweight and Efficient:** Designed for simplicity, making it easy to understand and extend.
*   **Accelerated Performance:** Supports multiple accelerators including GPU (OpenCL), CPU (C Code), LLVM, METAL, CUDA, AMD, NV, QCOM, and WEBGPU.
*   **Flexible and Extensible:** Easily add support for new accelerators with just a few low-level operations.
*   **Runs LLaMA and Stable Diffusion:** Demonstrates the framework's capabilities with complex models.
*   **Lazy Evaluation:** Optimizes computations by fusing operations into efficient kernels.
*   **Comprehensive Neural Network Support:** Includes autograd, tensor library, optimizers, and data loaders.

### Key Links:

*   [Homepage](https://github.com/tinygrad/tinygrad)
*   [Documentation](https://docs.tinygrad.org/)
*   [Discord](https://discord.gg/ZjZadyC7PK)

[![GitHub Repo stars](https://img.shields.io/github/stars/tinygrad/tinygrad)](https://github.com/tinygrad/tinygrad/stargazers)
[![Unit Tests](https://github.com/tinygrad/tinygrad/actions/workflows/test.yml/badge.svg)](https://github.com/tinygrad/tinygrad/actions/workflows/test.yml)
[![Discord](https://img.shields.io/discord/1068976834382925865)](https://discord.gg/ZjZadyC7PK)

---

## Core Functionality

Despite its small size, tinygrad is a fully featured deep learning framework. It's designed to be the easiest framework to add new accelerators to, with support for both inference and training.

### Get Started with Neural Networks

Build and train neural networks with tinygrad.

```python
from tinygrad import Tensor, nn

class LinearNet:
  def __init__(self):
    self.l1 = Tensor.kaiming_uniform(784, 128)
    self.l2 = Tensor.kaiming_uniform(128, 10)
  def __call__(self, x:Tensor) -> Tensor:
    return x.flatten(1).dot(self.l1).relu().dot(self.l2)

model = LinearNet()
optim = nn.optim.Adam([model.l1, model.l2], lr=0.001)

x, y = Tensor.rand(4, 1, 28, 28), Tensor([2,4,3,7])  # replace with real mnist dataloader

with Tensor.train():
  for i in range(10):
    optim.zero_grad()
    loss = model(x).sparse_categorical_crossentropy(y).backward()
    optim.step()
    print(i, loss.item())
```
See [examples/beautiful_mnist.py](examples/beautiful_mnist.py) for a complete MNIST example.

### Installation

Install tinygrad from source or directly using pip:

#### From Source
```bash
git clone https://github.com/tinygrad/tinygrad.git
cd tinygrad
python3 -m pip install -e .
```

#### Direct (master)
```bash
python3 -m pip install git+https://github.com/tinygrad/tinygrad.git
```

### Quick Comparison to PyTorch

```python
from tinygrad import Tensor

x = Tensor.eye(3, requires_grad=True)
y = Tensor([[2.0,0,-2.0]], requires_grad=True)
z = y.matmul(x).sum()
z.backward()

print(x.grad.tolist())  # dz/dx
print(y.grad.tolist())  # dz/dy
```

The same in PyTorch:
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

We welcome contributions! Please review the [Contribution Guidelines](https://github.com/tinygrad/tinygrad/blob/master/README.md#contributing) before submitting a pull request.