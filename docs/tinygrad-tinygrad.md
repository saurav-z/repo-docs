<div align="center">

<picture>
  <source media="(prefers-color-scheme: light)" srcset="/docs/logo_tiny_light.svg">
  <img alt="tiny corp logo" src="/docs/logo_tiny_dark.svg" width="50%" height="50%">
</picture>

</div>

# tinygrad: A Deep Learning Framework for Everyone

**tinygrad offers a lightweight, flexible, and fast deep learning experience, designed to be a simpler alternative to PyTorch.**

[View the project on GitHub](https://github.com/tinygrad/tinygrad)

**Key Features:**

*   **Runs LLaMA and Stable Diffusion:** Experience cutting-edge AI models with tinygrad.
*   **Lazy Evaluation:** Achieve efficient kernel fusion and optimized computation with lazy evaluation.
*   **Neural Network Capabilities:** Build and train neural networks with a minimal and intuitive API.
*   **Broad Accelerator Support:** Easily add new accelerators. tinygrad supports a wide range of hardware, including:
    *   GPU (OpenCL)
    *   CPU (C Code)
    *   LLVM
    *   METAL
    *   CUDA
    *   AMD
    *   NV
    *   QCOM
    *   WEBGPU
*   **Easy to Install:** Install from source or directly using pip.

## Get Started

### Installation

1.  **From Source:**

    ```bash
    git clone https://github.com/tinygrad/tinygrad.git
    cd tinygrad
    python3 -m pip install -e .
    ```

2.  **Direct (master):**

    ```bash
    python3 -m pip install git+https://github.com/tinygrad/tinygrad.git
    ```

### Example: Compare to PyTorch
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

## Documentation

Comprehensive documentation and a quick start guide are available on the [tinygrad docs website](https://docs.tinygrad.org/).

## Contributing

See the [Contributing](#contributing) section in the original README for details on how to contribute.