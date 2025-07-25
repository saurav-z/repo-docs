html
<div align="center">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="/docs/logo_tiny_light.svg">
    <img alt="tiny corp logo" src="/docs/logo_tiny_dark.svg" width="50%" height="50%">
  </picture>
</div>

<h1 align="center">tinygrad: A Deep Learning Framework Designed for Simplicity and Efficiency</h1>

<div align="center">
  <a href="https://github.com/tinygrad/tinygrad">
    <img src="https://img.shields.io/github/stars/tinygrad/tinygrad?style=social" alt="GitHub stars">
  </a>
  <a href="https://discord.gg/ZjZadyC7PK">
    <img src="https://img.shields.io/discord/1068976834382925865?label=Discord&logo=discord" alt="Discord">
  </a>
</div>

**Tinygrad is a deep learning framework that aims for simplicity, making it easy to add new hardware backends and experiment with novel architectures.  <a href="https://github.com/tinygrad/tinygrad">Explore the tinygrad repository!</a>**

---

## Key Features

*   **Simple and Extensible:**  Built for ease of use, making it straightforward to add support for new accelerators.
*   **Accelerated Performance:** Supports a wide range of hardware, including GPU (OpenCL), CPU (C Code), LLVM, METAL, CUDA, AMD, NV, QCOM, and WEBGPU, and is easy to add more.
*   **Runs Cutting-Edge Models:** Can run complex models like LLaMA and Stable Diffusion.
*   **Lazy Evaluation:** Optimizes operations through lazy evaluation, resulting in efficient kernel fusion.
*   **Neural Network Support:** Provides the essential components for building and training neural networks, including autograd, tensor library, optimizers, and data loaders.
*   **Minimal Dependencies:** Tinygrad is designed to be lightweight, reducing the overhead of installation and dependencies.

## Examples

### LLaMA and Stable Diffusion

tinygrad can run LLaMA and Stable Diffusion, showing its capabilities in handling complex models.

### Lazy Evaluation Example

Demonstrates how matmul operations are fused into a single kernel for performance gains:

```bash
DEBUG=3 python3 -c "from tinygrad import Tensor;
N = 1024; a, b = Tensor.empty(N, N), Tensor.empty(N, N);
(a.reshape(N, 1, N) * b.T.reshape(1, N, N)).sum(axis=2).realize()"
```

### Neural Network Example

A basic linear neural network implemented with tinygrad:

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
## Accelerators

tinygrad supports a variety of hardware accelerators:

-   [x] GPU (OpenCL)
-   [x] CPU (C Code)
-   [x] LLVM
-   [x] METAL
-   [x] CUDA
-   [x] AMD
-   [x] NV
-   [x] QCOM
-   [x] WEBGPU

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

### Quick Comparison to PyTorch

Demonstrates the similarity in syntax between tinygrad and PyTorch:

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

## Contributing

We welcome contributions! Please review the [Contributing Guidelines](https://github.com/tinygrad/tinygrad/blob/master/CONTRIBUTING.md) before submitting a pull request.