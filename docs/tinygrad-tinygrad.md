<div align="center">

<picture>
  <source media="(prefers-color-scheme: light)" srcset="/docs/logo_tiny_light.svg">
  <img alt="tiny corp logo" src="/docs/logo_tiny_dark.svg" width="50%" height="50%">
</picture>

</div>

# tinygrad: A Deep Learning Framework Built for Simplicity and Speed

**tinygrad is a lightweight deep learning framework designed to be a streamlined alternative to PyTorch, enabling efficient and flexible neural network development.** ([See the original repo](https://github.com/tinygrad/tinygrad))

**Key Features:**

*   **Lightweight and Efficient:** Designed for minimal overhead and fast execution, perfect for research and resource-constrained environments.
*   **Accelerated:** Supports multiple accelerators, including GPU (OpenCL, Metal, CUDA, AMD, NV, QCOM, WEBGPU), CPU (C Code, LLVM), and more, making it easy to add new hardware support.
*   **LLaMA and Stable Diffusion Compatibility:** Run complex models like LLaMA and Stable Diffusion with ease.
*   **Lazy Evaluation:** Optimizes computations through lazy evaluation, fusing operations into efficient kernels.
*   **Easy to Learn:** The project's simplicity makes it ideal for understanding the fundamentals of deep learning frameworks.
*   **Neural Network Capabilities:** Offers essential building blocks for neural networks, including autograd, a tensor library, optimizers, and data loading capabilities.

**Get Started:**

*   [Homepage](https://github.com/tinygrad/tinygrad)
*   [Documentation](https://docs.tinygrad.org/)
*   [Discord](https://discord.gg/ZjZadyC7PK)

[![GitHub Repo stars](https://img.shields.io/github/stars/tinygrad/tinygrad)](https://github.com/tinygrad/tinygrad/stargazers)
[![Unit Tests](https://github.com/tinygrad/tinygrad/actions/workflows/test.yml/badge.svg)](https://github.com/tinygrad/tinygrad/actions/workflows/test.yml)
[![Discord](https://img.shields.io/discord/1068976834382925865)](https://discord.gg/ZjZadyC7PK)

---

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

---

## Quick Example

```python
from tinygrad import Tensor

x = Tensor.eye(3, requires_grad=True)
y = Tensor([[2.0,0,-2.0]], requires_grad=True)
z = y.matmul(x).sum()
z.backward()

print(x.grad.tolist())  # dz/dx
print(y.grad.tolist())  # dz/dy
```

## Contributing

We welcome contributions!  Please review our guidelines before submitting a pull request.

*   **Bug fixes (with a regression test) are great!**
*   **Solving bounties is greatly appreciated**
*   **Features** (with regression tests)
*   **Refactors that are clear wins**
*   **Tests/fuzzers.**
*   **Dead code removal from core `tinygrad/` folder.**

### Running tests
```bash
python3 -m pip install -e '.[testing]'  # install extra deps for testing
python3 test/test_ops.py                # just the ops tests
python3 -m pytest test/                 # whole test suite
```
```