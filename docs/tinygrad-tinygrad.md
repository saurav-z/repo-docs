<div align="center">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="/docs/logo_tiny_light.svg">
    <img alt="tiny corp logo" src="/docs/logo_tiny_dark.svg" width="50%" height="50%">
  </picture>
</div>

# tinygrad: A Deep Learning Framework for Everyone

**tinygrad is a lightweight deep learning framework, offering a simplified approach to building and deploying neural networks, and providing an alternative to frameworks like PyTorch and micrograd.** Explore the [tinygrad repository](https://github.com/tinygrad/tinygrad) for the latest updates.

Key Features:

*   **Lightweight & Simple:** Tinygrad is designed for ease of use and understanding.
*   **Accelerated Performance:** Supports various accelerators for both inference and training, including GPU (OpenCL), CPU (C Code), LLVM, METAL, CUDA, AMD, NV, QCOM, and WEBGPU.
*   **Extensible:** Easy to add support for new accelerators with just ~25 low-level operations.
*   **Runs LLaMA and Stable Diffusion:** Demonstrates the framework's capabilities with complex models.
*   **Lazy Evaluation:** Leverages lazy evaluation for efficient kernel fusion.
*   **Neural Network Capabilities:** Build, train, and optimize neural networks with a concise API.

## Why Choose tinygrad?

tinygrad offers a unique perspective on deep learning frameworks:

*   **Simplicity:** Its design makes it easier to understand the underlying principles of deep learning.
*   **Flexibility:** Easily adapt the framework to new hardware or architectures.
*   **Efficiency:** Leverage lazy evaluation for optimized computation graphs.

## Getting Started

### Installation

Install tinygrad from source:

```bash
git clone https://github.com/tinygrad/tinygrad.git
cd tinygrad
python3 -m pip install -e .
```

Alternatively, install the master branch directly:

```bash
python3 -m pip install git+https://github.com/tinygrad/tinygrad.git
```

### Example: Quick Comparison to PyTorch

```python
from tinygrad import Tensor

x = Tensor.eye(3, requires_grad=True)
y = Tensor([[2.0,0,-2.0]], requires_grad=True)
z = y.matmul(x).sum()
z.backward()

print(x.grad.tolist())  # dz/dx
print(y.grad.tolist())  # dz/dy
```

Same code in PyTorch:

```python
import torch

x = torch.eye(3, requires_grad=True)
y = torch.tensor([[2.0,0,-2.0]], requires_grad=True)
z = y.matmul(x).sum()
z.backward()

print(x.grad.tolist())  # dz/dx
print(y.grad.tolist())  # dz/dy
```

## Documentation

For detailed information, refer to the [official documentation](https://docs.tinygrad.org/).

## Contributing

We welcome contributions! Please review the [contributing guidelines](https://github.com/tinygrad/tinygrad#contributing) before submitting a pull request.

### Running Tests

Install pre-commit hooks:

```bash
pre-commit install
```

Run a subset of tests:

```bash
python3 -m pip install -e '.[testing]'  # install extra deps for testing
python3 test/test_ops.py                # just the ops tests
python3 -m pytest test/                 # whole test suite
```

#### Process Replay Tests

[Process replay](https://github.com/tinygrad/tinygrad/blob/master/test/external/process_replay/README.md) compares your PR's generated kernels against master. If your PR is a refactor or speedup without any expected behavior change, It should include [pr] in the pull request title.