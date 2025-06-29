<div align="center">

<picture>
  <source media="(prefers-color-scheme: light)" srcset="/docs/logo_tiny_light.svg">
  <img alt="tiny corp logo" src="/docs/logo_tiny_dark.svg" width="50%" height="50%">
</picture>

</div>

# tinygrad: A Deep Learning Framework for Everyone

**tinygrad** is a minimalist deep learning framework, designed for simplicity and ease of use, offering a unique perspective on deep learning from [tiny corp](https://tinygrad.org/).  Access the original repo [here](https://github.com/tinygrad/tinygrad).

---

### Key Features:

*   **Fully Featured Deep Learning:** Despite its size, tinygrad supports both inference and training, making it a powerful tool for various deep learning tasks.
*   **Simplicity and Extensibility:**  tinygrad's design makes it exceptionally easy to add support for new accelerators.
*   **LLaMA and Stable Diffusion Support:** Run advanced models like LLaMA and Stable Diffusion with tinygrad.
*   **Lazy Evaluation:** Experience the power of lazy evaluation with fused kernels for efficient computation, reducing overhead and improving performance.
*   **Neural Network Capabilities:** Build and train neural networks with essential components like autograd, tensor libraries, optimizers, and data loaders.
*   **Broad Accelerator Support:**  Includes support for various accelerators:
    *   [x] GPU (OpenCL)
    *   [x] CPU (C Code)
    *   [x] LLVM
    *   [x] METAL
    *   [x] CUDA
    *   [x] AMD
    *   [x] NV
    *   [x] QCOM
    *   [x] WEBGPU
*   **Easy to Install:** Install from source or directly using pip.

### Installation

*   **From Source:**
    ```bash
    git clone https://github.com/tinygrad/tinygrad.git
    cd tinygrad
    python3 -m pip install -e .
    ```
*   **Direct (Master):**
    ```bash
    python3 -m pip install git+https://github.com/tinygrad/tinygrad.git
    ```

### Documentation

Comprehensive documentation and a quick start guide are available on the [docs website](https://docs.tinygrad.org/).

### Examples

A quick example comparing tinygrad with PyTorch, demonstrates how to perform the same matrix operations:

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

### Contributing

Guidelines for contributing to tinygrad are outlined in the original README.

### Running tests

```bash
python3 -m pip install -e '.[testing]'  # install extra deps for testing
python3 test/test_ops.py                # just the ops tests
python3 -m pytest test/                 # whole test suite