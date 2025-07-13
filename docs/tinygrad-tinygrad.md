<div align="center">

<picture>
  <source media="(prefers-color-scheme: light)" srcset="/docs/logo_tiny_light.svg">
  <img alt="tiny corp logo" src="/docs/logo_tiny_dark.svg" width="50%" height="50%">
</picture>

</div>

# tinygrad: A Deep Learning Framework for Everyone

**tinygrad is a simple yet powerful deep learning framework designed to be a lightweight alternative to frameworks like PyTorch.** [Explore the code on GitHub](https://github.com/tinygrad/tinygrad)!

## Key Features

*   **Lightweight and Simple:** Experience the power of deep learning with a framework that's easy to understand and extend.
*   **Supports LLaMA and Stable Diffusion:** Run cutting-edge models with tinygrad.
*   **Lazy Evaluation:** Witness the magic of kernel fusion through lazy evaluation for optimized performance.
*   **Neural Network Capabilities:** Build and train neural networks with ease, complete with autograd, optimizers, and more.
*   **Multi-Accelerator Support:** Run your models on a variety of hardware, from CPUs to GPUs, with support for:
    *   GPU (OpenCL)
    *   CPU (C Code)
    *   LLVM
    *   METAL
    *   CUDA
    *   AMD
    *   NV
    *   QCOM
    *   WEBGPU
*   **Easy to Extend:** Add support for new accelerators with only ~25 low-level operations to implement.

## Getting Started

### Installation

1.  **From Source (Recommended):**
    ```bash
    git clone https://github.com/tinygrad/tinygrad.git
    cd tinygrad
    python3 -m pip install -e .
    ```

2.  **Direct (Master):**
    ```bash
    python3 -m pip install git+https://github.com/tinygrad/tinygrad.git
    ```

### Example: Tinygrad vs. PyTorch

```python
from tinygrad import Tensor

x = Tensor.eye(3, requires_grad=True)
y = Tensor([[2.0,0,-2.0]], requires_grad=True)
z = y.matmul(x).sum()
z.backward()

print(x.grad.tolist())  # dz/dx
print(y.grad.tolist())  # dz/dy
```

Compare this to the PyTorch equivalent for a quick taste of the simplicity!

## Documentation

Find detailed documentation and a quick start guide on the [docs website](https://docs.tinygrad.org/).

## Contributing

We welcome contributions! Please review the [Contribution Guidelines](https://github.com/tinygrad/tinygrad#contributing) before submitting a pull request.

## Further Resources

*   [Homepage](https://github.com/tinygrad/tinygrad)
*   [Documentation](https://docs.tinygrad.org/)
*   [Discord](https://discord.gg/ZjZadyC7PK)
*   [GitHub Stars](https://github.com/tinygrad/tinygrad/stargazers)
*   [Unit Tests](https://github.com/tinygrad/tinygrad/actions/workflows/test.yml)