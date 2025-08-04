<div align="center">

<picture>
  <source media="(prefers-color-scheme: light)" srcset="/docs/logo_tiny_light.svg">
  <img alt="tiny corp logo" src="/docs/logo_tiny_dark.svg" width="50%" height="50%">
</picture>

</div>

# tinygrad: A Deep Learning Framework for Everyone

**tinygrad is a minimalist deep learning framework, offering a streamlined approach to building and deploying neural networks with exceptional efficiency.**

[View the original repository on GitHub](https://github.com/tinygrad/tinygrad) | [Documentation](https://docs.tinygrad.org/) | [Discord](https://discord.gg/ZjZadyC7PK)

[![GitHub Repo stars](https://img.shields.io/github/stars/tinygrad/tinygrad)](https://github.com/tinygrad/tinygrad/stargazers)
[![Unit Tests](https://github.com/tinygrad/tinygrad/actions/workflows/test.yml/badge.svg)](https://github.com/tinygrad/tinygrad/actions/workflows/test.yml)
[![Discord](https://img.shields.io/discord/1068976834382925865)](https://discord.gg/ZjZadyC7PK)

---

## Key Features

*   **Lightweight and Efficient:** tinygrad's simplicity makes it ideal for both research and production, with a focus on minimal dependencies and resource usage.
*   **Accelerated Performance:** Supports a wide range of accelerators, including GPU (OpenCL, Metal, CUDA, AMD), CPU (C Code, LLVM), and even WebGPU, enabling fast training and inference.
*   **Easy to Extend:** The framework's design makes adding support for new accelerators remarkably straightforward.
*   **LLaMA and Stable Diffusion Support:** Run state-of-the-art models like LLaMA and Stable Diffusion with tinygrad.
*   **Lazy Evaluation:** Automatic kernel fusion through lazy evaluation optimizes computations for superior performance.
*   **Neural Network Capabilities:** Build and train neural networks with autograd, optimizers, and data loaders.

## Getting Started

### Installation

Install tinygrad from source:

```bash
git clone https://github.com/tinygrad/tinygrad.git
cd tinygrad
python3 -m pip install -e .
```

Or install the latest master branch directly:

```bash
python3 -m pip install git+https://github.com/tinygrad/tinygrad.git
```

## Examples

### Running Matmuls with Laziness

See how lazy evaluation fuses operations into a single kernel:

```bash
DEBUG=3 python3 -c "from tinygrad import Tensor;
N = 1024; a, b = Tensor.empty(N, N), Tensor.empty(N, N);
(a.reshape(N, 1, N) * b.T.reshape(1, N, N)).sum(axis=2).realize()"
```

Change `DEBUG` to `4` to view the generated code.

### Simple Neural Network Example

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

For a full MNIST example, see `examples/beautiful_mnist.py`.

## Supported Accelerators

tinygrad supports:

-   [x] [GPU (OpenCL)](tinygrad/runtime/ops_gpu.py)
-   [x] [CPU (C Code)](tinygrad/runtime/ops_cpu.py)
-   [x] [LLVM](tinygrad/runtime/ops_llvm.py)
-   [x] [METAL](tinygrad/runtime/ops_metal.py)
-   [x] [CUDA](tinygrad/runtime/ops_cuda.py)
-   [x] [AMD](tinygrad/runtime/ops_amd.py)
-   [x] [NV](tinygrad/runtime/ops_nv.py)
-   [x] [QCOM](tinygrad/runtime/ops_qcom.py)
-   [x] [WEBGPU](tinygrad/runtime/ops_webgpu.py)

To check your default accelerator, run: `python3 -c "from tinygrad import Device; print(Device.DEFAULT)"`

## Documentation

Comprehensive documentation, including a quick start guide, is available on the [docs website](https://docs.tinygrad.org/).

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

We welcome contributions! Please follow these guidelines:

*   **Focus:** Prioritize bug fixes (with tests), solving bounties, adding features (with tests), and clear refactors.
*   **Quality:** Ensure all new code is high-quality and well-tested.
*   **Tests:** Include tests for all new functionality.

### Running Tests

Install pre-commit hooks:

```bash
pre-commit install
```

Run the full test suite:

```bash
python3 -m pytest test/
```

For more examples on how to run the full test suite please refer to the [CI workflow](.github/workflows/test.yml).