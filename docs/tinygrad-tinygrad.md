<div align="center">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="/docs/logo_tiny_light.svg">
    <img alt="tiny corp logo" src="/docs/logo_tiny_dark.svg" width="50%" height="50%">
  </picture>
</div>

# tinygrad: A Deep Learning Framework for RISC-V Style Acceleration

**tinygrad** is a fully-featured deep learning framework designed for simplicity and ease of adding new hardware accelerators, offering a streamlined alternative to frameworks like PyTorch. Check out the [original repo](https://github.com/tinygrad/tinygrad) for more details!

## Key Features

*   **Runs LLaMA and Stable Diffusion:** Supports complex models with ease.
*   **Lazy Evaluation:** Optimizes operations by fusing them into efficient kernels.
*   **Neural Network Support:** Provides essential components like autograd, tensor library, optimizers, and data loaders.
*   **Broad Accelerator Support:** Includes GPU (OpenCL), CPU (C Code), LLVM, METAL, CUDA, AMD, NV, QCOM, and WEBGPU. Easily extendable to new hardware.
*   **Simple to Use**: Built with a core set of ~25 low-level ops which makes it easy to add new accelerators

## Get Started

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

### Example: Tensor Operations

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

Contributions are welcome!  Please review the [contributing guidelines](https://github.com/tinygrad/tinygrad#contributing) to ensure your pull requests are accepted quickly.

*   Bug fixes (with regression tests)
*   Solving bounties
*   Features
*   Refactors
*   Tests/fuzzers
*   Dead code removal from the core `tinygrad/` folder

```