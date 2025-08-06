<div align="center">

<picture>
  <source media="(prefers-color-scheme: light)" srcset="/docs/logo_tiny_light.svg">
  <img alt="tiny corp logo" src="/docs/logo_tiny_dark.svg" width="50%" height="50%">
</picture>

</div>

# tinygrad: A Deep Learning Framework for Everyone

**tinygrad** is a lightweight deep learning framework designed for simplicity, making it easy to understand and extend. 

[View the original repository on GitHub](https://github.com/tinygrad/tinygrad).

### Key Features

*   **Lightweight and Simple:** Tinygrad offers a streamlined approach to deep learning, making it easier to learn and experiment with.
*   **Full-Featured:** Despite its small size, tinygrad supports a complete set of deep learning functionalities.
*   **Easy Accelerator Integration:** Adding new accelerators is straightforward, allowing for flexible hardware support.
*   **LLaMA and Stable Diffusion Support:** Run advanced models like LLaMA and Stable Diffusion with tinygrad.
*   **Lazy Evaluation:** Optimized computations are fused into efficient kernels through laziness.
*   **Neural Network Capabilities:** Build and train neural networks with autograd, optimizers, and data loaders.

### Quick Links

*   [Homepage](https://github.com/tinygrad/tinygrad)
*   [Documentation](https://docs.tinygrad.org/)
*   [Discord](https://discord.gg/ZjZadyC7PK)

[![GitHub Repo stars](https://img.shields.io/github/stars/tinygrad/tinygrad)](https://github.com/tinygrad/tinygrad/stargazers)
[![Unit Tests](https://github.com/tinygrad/tinygrad/actions/workflows/test.yml/badge.svg)](https://github.com/tinygrad/tinygrad/actions/workflows/test.yml)
[![Discord](https://img.shields.io/discord/1068976834382925865)](https://discord.gg/ZjZadyC7PK)

---

## What is tinygrad?

tinygrad is a deep learning framework that aims to be a middle ground between PyTorch and micrograd. It is maintained by [tiny corp](https://tinygrad.org).

### Example: Run a matmul and view the fused kernel

```sh
DEBUG=3 python3 -c "from tinygrad import Tensor;
N = 1024; a, b = Tensor.empty(N, N), Tensor.empty(N, N);
(a.reshape(N, 1, N) * b.T.reshape(1, N, N)).sum(axis=2).realize()"
```

### Build Neural Networks
Create your own neural networks quickly

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

See [examples/beautiful_mnist.py](examples/beautiful_mnist.py) for the full version that gets 98% in ~5 seconds

## Supported Accelerators

tinygrad supports a variety of accelerators, making it adaptable to different hardware environments:

-   [x] [GPU (OpenCL)](tinygrad/runtime/ops_gpu.py)
-   [x] [CPU (C Code)](tinygrad/runtime/ops_cpu.py)
-   [x] [LLVM](tinygrad/runtime/ops_llvm.py)
-   [x] [METAL](tinygrad/runtime/ops_metal.py)
-   [x] [CUDA](tinygrad/runtime/ops_cuda.py)
-   [x] [AMD](tinygrad/runtime/ops_amd.py)
-   [x] [NV](tinygrad/runtime/ops_nv.py)
-   [x] [QCOM](tinygrad/runtime/ops_qcom.py)
-   [x] [WEBGPU](tinygrad/runtime/ops_webgpu.py)

You can check your default accelerator by running: `python3 -c "from tinygrad import Device; print(Device.DEFAULT)"`

## Installation

Install tinygrad from source:

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

Find detailed information and a quick start guide on the [docs website](https://docs.tinygrad.org/).

### Quick Example: Tinygrad vs. PyTorch

```python
from tinygrad import Tensor

x = Tensor.eye(3, requires_grad=True)
y = Tensor([[2.0,0,-2.0]], requires_grad=True)
z = y.matmul(x).sum()
z.backward()

print(x.grad.tolist())  # dz/dx
print(y.grad.tolist())  # dz/dy
```

The same thing but in PyTorch:
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

We welcome contributions! Please follow these guidelines to ensure your pull requests are accepted.

*   **No code golf.**
*   **Avoid documentation and whitespace changes** unless you're a well-known contributor.
*   **Benchmark any speedups.**
*   **Focus on the core `tinygrad/` folder.**
*   **Keep PRs concise and focused.**

### How to contribute
*   Bug fixes
*   Solve [bounties](https://docs.google.com/spreadsheets/d/1WKHbT-7KOgjEawq5h5Ic1qUWzpfAzuD_J06N1JwOCGs/edit?usp=sharing)
*   New features
*   Refactors that are clear wins
*   Tests and fuzzers
*   Dead code removal

### Running Tests

Use `pre-commit install` to set up pre-commit hooks, which run linters, mypy, and a subset of tests on each commit.

Examples of running tests locally:
```sh
python3 -m pip install -e '.[testing]'  # install extra deps for testing
python3 test/test_ops.py                # just the ops tests
python3 -m pytest test/                 # whole test suite
```

#### Process Replay Tests

For refactors or speedups, include [pr] in the pull request title. See [process replay](https://github.com/tinygrad/tinygrad/blob/master/test/external/process_replay/README.md) to compare your PR's generated kernels against master.