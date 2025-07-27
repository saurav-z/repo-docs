html
<div align="center">

<picture>
  <source media="(prefers-color-scheme: light)" srcset="/docs/logo_tiny_light.svg">
  <img alt="tiny corp logo" src="/docs/logo_tiny_dark.svg" width="50%" height="50%">
</picture>

</div>

# tinygrad: A Deep Learning Framework for Everyone

tinygrad is a lightweight deep learning framework, designed to be a simpler alternative to PyTorch, perfect for those seeking a deeper understanding of neural networks. Explore the [tinygrad repository](https://github.com/tinygrad/tinygrad) for more details.

**Key Features:**

*   **Runs LLaMA and Stable Diffusion:** Build and run complex models.
*   **Lazy Evaluation:** Experience efficient computation through kernel fusion.
*   **Neural Network Building:** Build, train, and optimize neural networks with ease.
*   **Accelerated Performance:** Supports a wide range of accelerators.
*   **Easy to Extend:** Add support for new accelerators with ~25 low-level ops.

[![GitHub Repo stars](https://img.shields.io/github/stars/tinygrad/tinygrad)](https://github.com/tinygrad/tinygrad/stargazers)
[![Unit Tests](https://github.com/tinygrad/tinygrad/actions/workflows/test.yml/badge.svg)](https://github.com/tinygrad/tinygrad/actions/workflows/test.yml)
[![Discord](https://img.shields.io/discord/1068976834382925865)](https://discord.gg/ZjZadyC7PK)

---

## What is tinygrad?

tinygrad is a fully-featured deep learning framework built for simplicity and ease of use. Whether you are just starting out or are an experienced deep learning engineer, tinygrad has a lot to offer. It's designed to be the easiest framework to add new accelerators to, with support for both inference and training.

## Key Advantages

*   **Simplicity:** Tinygrad is built on the principle of simplicity, making it easier to understand and modify.
*   **Flexibility:** Easily experiment with new architectures and training techniques.
*   **Accelerated Performance:** Supports a wide range of accelerators, including GPU, CPU, LLVM, METAL, CUDA, AMD, NV, QCOM, and WEBGPU.

## Getting Started

### Installation

Install tinygrad from source using these commands:

```bash
git clone https://github.com/tinygrad/tinygrad.git
cd tinygrad
python3 -m pip install -e .
```

Or, install directly from master:

```bash
python3 -m pip install git+https://github.com/tinygrad/tinygrad.git
```

### Quick Example

Here's a quick comparison to PyTorch to illustrate the framework's simplicity:

```python
from tinygrad import Tensor

x = Tensor.eye(3, requires_grad=True)
y = Tensor([[2.0,0,-2.0]], requires_grad=True)
z = y.matmul(x).sum()
z.backward()

print(x.grad.tolist())  # dz/dx
print(y.grad.tolist())  # dz/dy
```

## Contribute

Join the community and help shape the future of tinygrad! See the [Contributing](#contributing) section in the original README.

## Further Resources

*   [Homepage](https://github.com/tinygrad/tinygrad)
*   [Documentation](https://docs.tinygrad.org/)
*   [Discord](https://discord.gg/ZjZadyC7PK)