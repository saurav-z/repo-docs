<div align="center">

<picture>
  <source media="(prefers-color-scheme: light)" srcset="/docs/logo_tiny_light.svg">
  <img alt="tiny corp logo" src="/docs/logo_tiny_dark.svg" width="50%" height="50%">
</picture>

</div>

# tinygrad: A Deep Learning Framework for Everyone

**tinygrad is a lightweight deep learning framework, offering a PyTorch-like experience with extreme simplicity and broad hardware support.**

[View the source on GitHub](https://github.com/tinygrad/tinygrad) | [Documentation](https://docs.tinygrad.org/) | [Discord](https://discord.gg/ZjZadyC7PK)

[![GitHub Repo stars](https://img.shields.io/github/stars/tinygrad/tinygrad)](https://github.com/tinygrad/tinygrad/stargazers)
[![Unit Tests](https://github.com/tinygrad/tinygrad/actions/workflows/test.yml/badge.svg)](https://github.com/tinygrad/tinygrad/actions/workflows/test.yml)
[![Discord](https://img.shields.io/discord/1068976834382925865)](https://discord.gg/ZjZadyC7PK)

---

## Key Features

*   **Lightweight and Simple:** Tinygrad's design prioritizes simplicity, making it easy to understand, modify, and extend.
*   **Accelerated Performance:** Harness the power of various accelerators, including:
    *   GPU (OpenCL)
    *   CPU (C Code)
    *   LLVM
    *   METAL
    *   CUDA
    *   AMD
    *   NV
    *   QCOM
    *   WEBGPU
*   **Easy Accelerator Support:** Add new accelerators with ease. Tinygrad requires support for only about 25 low-level operations.
*   **Supports Complex Models:** Run LLaMA and Stable Diffusion.
*   **Lazy Evaluation:** Experience efficient computation through lazy evaluation, fusing operations for optimal performance.

### Neural Network Capabilities

Build and train neural networks with ease using tinygrad's autograd/tensor library, optimizer, and dataloading capabilities.
*   **Example:**
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
    *See* [examples/beautiful_mnist.py](examples/beautiful_mnist.py) *for a more complete version*

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

## Documentation and Examples

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

We welcome contributions!  Please review the guidelines below to ensure your pull requests are accepted.

**What We Want:**

*   Bug fixes (with regression tests).
*   Solutions to bounties (see [bounties list](https://docs.google.com/spreadsheets/d/1WKHbT-7KOgjEawq5h5Ic1qUWzpfAzuD_J06N1JwOCGs/edit?usp=sharing)).
*   New features (with tests).
*   Clear, beneficial refactors.
*   Tests and fuzzers.
*   Dead code removal.

**Guidelines:**

*   No code golf.
*   No documentation or whitespace changes unless you're a well-known contributor.
*   "Speedup" claims must be benchmarked.
*   Code outside the core `tinygrad/` folder is generally not changed unless broken.
*   Keep PRs small and focused.

### Running Tests

Install pre-commit hooks with `pre-commit install` to run linters and tests automatically.

For full test suite details, refer to the [CI workflow](.github/workflows/test.yml).

Examples:
```bash
python3 -m pip install -e '.[testing]'  # install extra deps for testing
python3 test/test_ops.py                # just the ops tests
python3 -m pytest test/                 # whole test suite
```

#### Process Replay Tests

If your PR is a refactor or speedup, consider the [process replay](https://github.com/tinygrad/tinygrad/blob/master/test/external/process_replay/README.md) tests.