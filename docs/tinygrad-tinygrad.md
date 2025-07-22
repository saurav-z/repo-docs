<div align="center">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="/docs/logo_tiny_light.svg">
    <img alt="tiny corp logo" src="/docs/logo_tiny_dark.svg" width="50%" height="50%">
  </picture>
</div>

# tinygrad: A Deep Learning Framework for Everyone

**tinygrad is a lightweight deep learning framework designed for simplicity and ease of use, bridging the gap between PyTorch and micrograd.**  Explore the power of tinygrad and its capabilities!

[**View the project on GitHub**](https://github.com/tinygrad/tinygrad) | [Documentation](https://docs.tinygrad.org/) | [Discord](https://discord.gg/ZjZadyC7PK)

[![GitHub Repo stars](https://img.shields.io/github/stars/tinygrad/tinygrad)](https://github.com/tinygrad/tinygrad/stargazers)
[![Unit Tests](https://github.com/tinygrad/tinygrad/actions/workflows/test.yml/badge.svg)](https://github.com/tinygrad/tinygrad/actions/workflows/test.yml)
[![Discord](https://img.shields.io/discord/1068976834382925865)](https://img.shields.io/discord/1068976834382925865)

---

## Key Features

*   **Lightweight & Simple:** Designed for extreme simplicity, making it easy to understand and extend.
*   **Accelerator Agnostic:**  Supports various accelerators, including GPU (OpenCL), CPU (C Code), LLVM, METAL, CUDA, AMD, NV, QCOM, and WEBGPU, with easy addition of new ones.
*   **Runs LLaMA and Stable Diffusion:**  Demonstrates the framework's versatility with support for complex models.
*   **Lazy Evaluation:**  Optimizes performance by fusing operations into efficient kernels, as demonstrated in the matmul example below.
*   **Neural Network Capabilities:** Provides the essential components for building and training neural networks, including autograd, tensor library, and optimizers.

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

## Accelerators Supported

*   [x] GPU (OpenCL)
*   [x] CPU (C Code)
*   [x] LLVM
*   [x] METAL
*   [x] CUDA
*   [x] AMD
*   [x] NV
*   [x] QCOM
*   [x] WEBGPU

To check your default accelerator, run: `python3 -c "from tinygrad import Device; print(Device.DEFAULT)"`

## Installation

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

## Documentation & Quick Start

Comprehensive documentation and a quick start guide are available on the [docs website](https://docs.tinygrad.org/).

### Example: tinygrad vs. PyTorch

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

Contributions are welcome! Please review the [Contributing Guidelines](https://github.com/tinygrad/tinygrad#contributing) before submitting a pull request.