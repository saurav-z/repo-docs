<div align="center">
<picture>
  <source media="(prefers-color-scheme: light)" srcset="/docs/logo_tiny_light.svg">
  <img alt="tiny corp logo" src="/docs/logo_tiny_dark.svg" width="50%" height="50%">
</picture>
</div>

# tinygrad: A Deep Learning Framework for Everyone

**tinygrad** is a deep learning framework designed for simplicity and ease of use, offering a lightweight alternative between PyTorch and micrograd. Check out the [original repo](https://github.com/tinygrad/tinygrad) for more.

[![GitHub Repo stars](https://img.shields.io/github/stars/tinygrad/tinygrad)](https://github.com/tinygrad/tinygrad/stargazers)
[![Unit Tests](https://github.com/tinygrad/tinygrad/actions/workflows/test.yml/badge.svg)](https://github.com/tinygrad/tinygrad/actions/workflows/test.yml)
[![Discord](https://img.shields.io/discord/1068976834382925865)](https://discord.gg/ZjZadyC7PK)

## Key Features

*   **Simplicity:** Tinygrad is designed for ease of understanding and modification, making it ideal for learning and experimenting with deep learning.
*   **Accelerated Performance:** Supports multiple accelerators, including GPU (OpenCL, Metal, CUDA, AMD, NV), CPU (C Code, LLVM), and WebGPU.
*   **LLaMA and Stable Diffusion Support:** Run cutting-edge models like LLaMA and Stable Diffusion with tinygrad.
*   **Lazy Evaluation:** Leverage the power of lazy evaluation to optimize kernel fusion and computational efficiency.
*   **Neural Network Capabilities:** Build and train neural networks with autograd, tensor manipulation, optimizers, and data loaders.

## Examples

### Lazy Evaluation in Action

Observe how operations are fused into optimized kernels:

```bash
DEBUG=3 python3 -c "from tinygrad import Tensor;
N = 1024; a, b = Tensor.rand(N, N), Tensor.rand(N, N);
c = (a.reshape(N, 1, N) * b.T.reshape(1, N, N)).sum(axis=2);
print((c.numpy() - (a.numpy() @ b.numpy())).mean())"
```

Increase `DEBUG` to `4` to view the generated code.

### Neural Network Example

A concise example demonstrating neural network creation and training:

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

For a full version that achieves 98% accuracy on MNIST in about 5 seconds, see [examples/beautiful_mnist.py](examples/beautiful_mnist.py).

## Supported Accelerators

tinygrad supports the following accelerators:

*   [x] GPU (OpenCL)
*   [x] CPU (C Code)
*   [x] LLVM
*   [x] METAL
*   [x] CUDA
*   [x] AMD
*   [x] NV
*   [x] QCOM
*   [x] WEBGPU

Easily add support for your preferred accelerator by implementing ~25 low-level ops.

To check the default accelerator run: `python3 -c "from tinygrad import Device; print(Device.DEFAULT)"`

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

## Documentation

Comprehensive documentation, including a quick start guide, is available on the [docs website](https://docs.tinygrad.org/).

### Quick Example: Comparing to PyTorch

```python
from tinygrad import Tensor

x = Tensor.eye(3, requires_grad=True)
y = Tensor([[2.0,0,-2.0]], requires_grad=True)
z = y.matmul(x).sum()
z.backward()

print(x.grad.tolist())  # dz/dx
print(y.grad.tolist())  # dz/dy
```

The same code in PyTorch:

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

Contribute to tinygrad by following these guidelines:

*   **Avoid Code Golf:** Prioritize readability and simplicity over line count reduction.
*   **Documentation and Whitespace:** Only well-known contributors should modify documentation or whitespace.
*   **Benchmarking:**  All speedup claims must be benchmarked.
*   **External Code:**  Avoid changes outside the core `tinygrad/` directory unless necessary.
*   **PR Size:** Break large changes into smaller, focused pull requests.

### Running Tests

Install pre-commit hooks with `pre-commit install` for automated linting and testing.

Test examples:
```bash
python3 -m pip install -e '.[testing]'  # install extra deps for testing
python3 test/test_ops.py                # just the ops tests
python3 -m pytest test/                 # whole test suite
```

#### Process Replay Tests

For refactors or speedups, include `[pr]` in your pull request title and utilize [process replay](https://github.com/tinygrad/tinygrad/blob/master/test/external/process_replay/README.md) to compare generated kernels.