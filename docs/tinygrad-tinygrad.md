<div align="center">

<picture>
  <source media="(prefers-color-scheme: light)" srcset="/docs/logo_tiny_light.svg">
  <img alt="tiny corp logo" src="/docs/logo_tiny_dark.svg" width="50%" height="50%">
</picture>

</div>

# tinygrad: A Minimal Deep Learning Framework for Maximum Flexibility

**tinygrad** is a surprisingly powerful, yet incredibly simple, deep learning framework designed to be a lean alternative to PyTorch, empowering you to build and deploy models with ease. Explore the original repository [here](https://github.com/tinygrad/tinygrad).

## Key Features

*   **Lightweight & Efficient:** tinygrad's minimalist design makes it easy to understand, modify, and deploy on diverse hardware.
*   **Versatile Accelerator Support:**  Seamlessly supports a wide range of accelerators, including CPU, GPU (OpenCL, CUDA, Metal, WebGPU), LLVM, and more!
*   **Easy Accelerator Integration:** Extend tinygrad to new hardware with a few lines of code by supporting ~25 low-level operations.
*   **Runs LLaMA and Stable Diffusion:** Build cutting-edge AI models with a streamlined framework.
*   **Lazy Evaluation:** Experience the power of automatic kernel fusion through lazy evaluation, leading to optimized performance.
*   **Neural Network Capabilities:** Develop neural networks using a core autograd/tensor library, an optimizer, and a data loader.

## Get Started

### Installation

Install from source:

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

```python
from tinygrad import Tensor

x = Tensor.eye(3, requires_grad=True)
y = Tensor([[2.0,0,-2.0]], requires_grad=True)
z = y.matmul(x).sum()
z.backward()

print(x.grad.tolist())  # dz/dx
print(y.grad.tolist())  # dz/dy
```

## Documentation

Comprehensive documentation is available on the [docs website](https://docs.tinygrad.org/).

## Contributing

Contribute to tinygrad! Review the [Contribution Guidelines](https://github.com/tinygrad/tinygrad/blob/master/README.md#contributing) before submitting a pull request.