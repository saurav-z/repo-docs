<div align="center">

<picture>
  <source media="(prefers-color-scheme: light)" srcset="/docs/logo_tiny_light.svg">
  <img alt="tiny corp logo" src="/docs/logo_tiny_dark.svg" width="50%" height="50%">
</picture>

</div>

# tinygrad: A Deep Learning Framework for Everyone

**Tinygrad is a lightweight deep learning framework designed for simplicity and ease of use, offering a unique RISC-like approach to accelerate your machine learning projects.** Check out the original repo [here](https://github.com/tinygrad/tinygrad).

## Key Features

*   **Lightweight and Simple:** Easy to understand and extend, making it ideal for both beginners and experienced developers.
*   **Multi-Accelerator Support:** Runs on various hardware, including GPU (OpenCL, Metal, CUDA, AMD, NV, QCOM, WEBGPU), CPU (C Code, LLVM), and more.
*   **LLaMA and Stable Diffusion:** Supports running complex models like LLaMA and Stable Diffusion.
*   **Lazy Evaluation:** Optimizes computations by fusing operations into efficient kernels.
*   **Neural Network Capabilities:** Provides essential components for building and training neural networks, including autograd, tensor operations, optimizers, and data loaders.
*   **Easy to Add Accelerators:** Supports a total of ~25 low level ops, making it easy to add new accelerators.

## Key Benefits

*   **Faster Development:** The simplicity and design of tinygrad allow for faster development cycles and easy experimentation.
*   **Cross-Platform Compatibility:** Run your models on a variety of hardware.
*   **Community Support:** Join the vibrant [Discord](https://discord.gg/ZjZadyC7PK) community to get help and share your projects.

## Quick Start

### Installation

Install from source:

```bash
git clone https://github.com/tinygrad/tinygrad.git
cd tinygrad
python3 -m pip install -e .
```

Or install directly from master:

```bash
python3 -m pip install git+https://github.com/tinygrad/tinygrad.git
```

### Example

```python
from tinygrad import Tensor

x = Tensor.eye(3, requires_grad=True)
y = Tensor([[2.0,0,-2.0]], requires_grad=True)
z = y.matmul(x).sum()
z.backward()

print(x.grad.tolist())  # dz/dx
print(y.grad.tolist())  # dz/dy
```

This example shows how to perform matrix multiplication with automatic differentiation in tinygrad, similar to PyTorch.

## Documentation

*   Find the official [Documentation](https://docs.tinygrad.org/) for in-depth guides and examples.
*   The [Discord](https://discord.gg/ZjZadyC7PK) community is an excellent resource for support and discussions.

## Contributing

*   Bug fixes (with a regression test) are great!
*   Solving bounties! tinygrad [offers cash bounties](https://docs.google.com/spreadsheets/d/1WKHbT-7KOgjEawq5h5Ic1qUWzpfAzuD_J06N1JwOCGs/edit?usp=sharing) for certain improvements to the library. All new code should be high quality and well tested.
*   Features, with API matching torch or numpy.
*   Refactors that are clear wins, increasing readability.
*   Tests/fuzzers.

### Running tests

```bash
python3 -m pip install -e '.[testing]'  # install extra deps for testing
python3 test/test_ops.py                # just the ops tests
python3 -m pytest test/                 # whole test suite
```

#### Process replay tests

[Process replay](https://github.com/tinygrad/tinygrad/blob/master/test/external/process_replay/README.md) compares your PR's generated kernels against master. If your PR is a refactor or speedup without any expected behavior change, It should include [pr] in the pull request title.

---

**Get started with tinygrad today and experience the power of a minimalist deep learning framework!**

[![GitHub Repo stars](https://img.shields.io/github/stars/tinygrad/tinygrad)](https://github.com/tinygrad/tinygrad/stargazers)
[![Unit Tests](https://github.com/tinygrad/tinygrad/actions/workflows/test.yml/badge.svg)](https://github.com/tinygrad/tinygrad/actions/workflows/test.yml)
[![Discord](https://img.shields.io/discord/1068976834382925865)](https://discord.gg/ZjZadyC7PK)