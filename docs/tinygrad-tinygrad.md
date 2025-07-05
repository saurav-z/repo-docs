<div align="center">

<picture>
  <source media="(prefers-color-scheme: light)" srcset="/docs/logo_tiny_light.svg">
  <img alt="tiny corp logo" src="/docs/logo_tiny_dark.svg" width="50%" height="50%">
</picture>

</div>

# tinygrad: A Deep Learning Framework for RISC Architectures

**tinygrad** is a minimalist deep learning framework, offering a PyTorch-like experience with a focus on simplicity and efficient hardware utilization.  Explore the [tinygrad repository](https://github.com/tinygrad/tinygrad) for more details.

---

## Key Features

*   **Runs LLaMA and Stable Diffusion:** Utilize cutting-edge models with this lightweight framework.
*   **Lazy Evaluation:** Experience optimized performance through intelligent kernel fusion.
*   **Simple Neural Network Creation:** Build and train models with a streamlined autograd/tensor library.
*   **Wide Accelerator Support:**  Deploy on various hardware platforms with ease.
    *   GPU (OpenCL)
    *   CPU (C Code)
    *   LLVM
    *   METAL
    *   CUDA
    *   AMD
    *   NV
    *   QCOM
    *   WEBGPU
*   **Easy to Extend:** Quickly add support for your preferred accelerator with only a few low-level operations.

## Getting Started

### Installation

Install tinygrad from source:

```bash
git clone https://github.com/tinygrad/tinygrad.git
cd tinygrad
python3 -m pip install -e .
```

or directly from the master branch:

```bash
python3 -m pip install git+https://github.com/tinygrad/tinygrad.git
```

## Documentation

Comprehensive documentation is available at the [docs website](https://docs.tinygrad.org/), including a quick start guide.

### Quick Example: Comparing to PyTorch

See how tinygrad mirrors PyTorch's functionality:

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

We welcome contributions! Please review our [contributing guidelines](https://github.com/tinygrad/tinygrad/blob/master/README.md#contributing) to ensure your PR is aligned with project goals.