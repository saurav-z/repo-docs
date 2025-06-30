<div align="center">

<picture>
  <source media="(prefers-color-scheme: light)" srcset="/docs/logo_tiny_light.svg">
  <img alt="tiny corp logo" src="/docs/logo_tiny_dark.svg" width="50%" height="50%">
</picture>

</div>

# tinygrad: A Deep Learning Framework for Everyone

**tinygrad is a lightweight, fully-featured deep learning framework designed for simplicity and ease of use.** Dive into the world of deep learning without the complexity, and discover how easy it is to add new accelerators and run cutting-edge models.

[![GitHub Repo stars](https://img.shields.io/github/stars/tinygrad/tinygrad)](https://github.com/tinygrad/tinygrad/stargazers)
[![Unit Tests](https://github.com/tinygrad/tinygrad/actions/workflows/test.yml/badge.svg)](https://github.com/tinygrad/tinygrad/actions/workflows/test.yml)
[![Discord](https://img.shields.io/discord/1068976834382925865)](https://discord.gg/ZjZadyC7PK)

**Key Features:**

*   **Runs LLaMA and Stable Diffusion:** Experience state-of-the-art models with the power of tinygrad.
*   **Lazy Evaluation:** Witness the magic of fused kernels with lazy evaluation, optimizing your computations.
*   **Neural Network Support:** Build and train neural networks with an intuitive autograd/tensor library, optimizers, and data loaders.
*   **Extensive Accelerator Support:** Utilize a wide array of accelerators, including GPU (OpenCL), CPU (C Code), LLVM, METAL, CUDA, AMD, NV, QCOM, and WEBGPU, with more being added all the time.
*   **Easy to Extend:** Add support for your favorite accelerator with only ~25 low level ops.

**Get Started:**

*   **[Homepage](https://github.com/tinygrad/tinygrad)**
*   **[Documentation](https://docs.tinygrad.org/)**
*   **[Discord](https://discord.gg/ZjZadyC7PK)**

**Installation:**

Install tinygrad directly from source:

```bash
git clone https://github.com/tinygrad/tinygrad.git
cd tinygrad
python3 -m pip install -e .
```

Or install the latest master:

```bash
python3 -m pip install git+https://github.com/tinygrad/tinygrad.git
```

**Quick Example:**

```python
from tinygrad import Tensor

x = Tensor.eye(3, requires_grad=True)
y = Tensor([[2.0,0,-2.0]], requires_grad=True)
z = y.matmul(x).sum()
z.backward()

print(x.grad.tolist())  # dz/dx
print(y.grad.tolist())  # dz/dy
```

**[Find the original repository here](https://github.com/tinygrad/tinygrad)**

---

**Contributing:**

We welcome contributions! Please review the guidelines in the original README on the GitHub repository before submitting a pull request to ensure your contribution is aligned with the project's goals.