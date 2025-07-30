<div align="center">

<picture>
  <source media="(prefers-color-scheme: light)" srcset="/docs/logo_tiny_light.svg">
  <img alt="tiny corp logo" src="/docs/logo_tiny_dark.svg" width="50%" height="50%">
</picture>

</div>

# tinygrad: A Deep Learning Framework for Everyone

**tinygrad is a surprisingly capable deep learning framework, offering a lightweight and accessible alternative to larger frameworks.**  Dive into the world of neural networks with an easy-to-understand and extensible framework.

*   [Homepage](https://github.com/tinygrad/tinygrad) | [Documentation](https://docs.tinygrad.org/) | [Discord](https://discord.gg/ZjZadyC7PK)

[![GitHub Repo stars](https://img.shields.io/github/stars/tinygrad/tinygrad)](https://github.com/tinygrad/tinygrad/stargazers)
[![Unit Tests](https://github.com/tinygrad/tinygrad/actions/workflows/test.yml/badge.svg)](https://github.com/tinygrad/tinygrad/actions/workflows/test.yml)
[![Discord](https://img.shields.io/discord/1068976834382925865)](https://discord.gg/ZjZadyC7PK)

---

## Key Features of tinygrad

*   **Runs LLaMA and Stable Diffusion:** Experience the power of cutting-edge AI models with a streamlined framework.
*   **Lazy Evaluation for Efficiency:**  Benefit from automatic kernel fusion and optimized computation through lazy evaluation.  See how it works:

    ```bash
    DEBUG=3 python3 -c "from tinygrad import Tensor;
    N = 1024; a, b = Tensor.empty(N, N), Tensor.empty(N, N);
    (a.reshape(N, 1, N) * b.T.reshape(1, N, N)).sum(axis=2).realize()"
    ```
    Set `DEBUG=4` to inspect the generated code.
*   **Simplified Neural Network Development:** Build and train neural networks with a minimal set of tools, including autograd, tensor libraries, optimizers, and data loaders.

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
*   **Multi-Platform Accelerator Support:**  Accelerate your models on various hardware platforms:

    *   [x] GPU (OpenCL)
    *   [x] CPU (C Code)
    *   [x] LLVM
    *   [x] METAL
    *   [x] CUDA
    *   [x] AMD
    *   [x] NV
    *   [x] QCOM
    *   [x] WEBGPU

    Easily add support for new accelerators by implementing a small number of low-level ops.

## Installation

The recommended way to install tinygrad is from source.

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

Comprehensive documentation, including a quick start guide, is available on the [docs website](https://docs.tinygrad.org/).

### Quick Example - Tinygrad vs. PyTorch

```python
from tinygrad import Tensor

x = Tensor.eye(3, requires_grad=True)
y = Tensor([[2.0,0,-2.0]], requires_grad=True)
z = y.matmul(x).sum()
z.backward()

print(x.grad.tolist())  # dz/dx
print(y.grad.tolist())  # dz/dy
```

vs. PyTorch:
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

Contributions are welcome! Please follow these guidelines to ensure your pull request is accepted:

*   **Avoid Code Golf:** Prioritize readability and clarity over minimal line counts.
*   **Documentation and Whitespace:**  Only well-known contributors should modify documentation and whitespace.
*   **Benchmarking:** Back up speedup claims with benchmarks.
*   **Focus on Core Functionality:**  Limit changes outside the `tinygrad/` folder unless addressing a broken feature.
*   **Smaller, Focused PRs:** Break down large changes into smaller, more manageable pull requests.

**What We Want:**

*   Bug fixes (with regression tests)
*   Bounties (see [bounties](https://docs.google.com/spreadsheets/d/1WKHbT-7KOgjEawq5h5Ic1qUWzpfAzuD_J06N1JwOCGs/edit?usp=sharing))
*   Features (with tests, and consider the line tradeoff)
*   Refactors (clear wins are welcome)
*   Tests/fuzzers
*   Dead code removal

### Running Tests

Install pre-commit hooks with `pre-commit install` to automatically run the linter and a subset of tests on every commit.

To run the full test suite, refer to the [CI workflow](.github/workflows/test.yml).

Examples:

```bash
python3 -m pip install -e '.[testing]'  # install extra deps for testing
python3 test/test_ops.py                # just the ops tests
python3 -m pytest test/                 # whole test suite
```

#### Process replay tests

[Process replay](https://github.com/tinygrad/tinygrad/blob/master/test/external/process_replay/README.md) compares your PR's generated kernels against master. Include [pr] in the pull request title for refactors or speedups without behavior change.

**Get started with tinygrad today and unlock the world of deep learning!** [See the original repo](https://github.com/tinygrad/tinygrad)