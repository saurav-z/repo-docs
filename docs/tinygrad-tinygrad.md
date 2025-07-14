<div align="center">

<picture>
  <source media="(prefers-color-scheme: light)" srcset="/docs/logo_tiny_light.svg">
  <img alt="tiny corp logo" src="/docs/logo_tiny_dark.svg" width="50%" height="50%">
</picture>

</div>

# tinygrad: A Deep Learning Framework Designed for Simplicity and Efficiency

**tinygrad is a minimalist deep learning framework, offering a powerful alternative to PyTorch and micrograd, built for easy accelerator integration and efficient performance.** Explore the power of tinygrad and its ability to run LLaMA and Stable Diffusion, all while understanding how it works under the hood.

**[Explore the tinygrad Repository](https://github.com/tinygrad/tinygrad)**

### Key Features

*   **Runs LLaMA and Stable Diffusion:** Experience cutting-edge AI models with tinygrad.
*   **Laziness:** Optimize performance through fused kernels.
*   **Neural Network Support:** Build and train neural networks with a clean and concise API.
*   **Extensive Accelerator Support:** Seamlessly runs on various hardware including GPU (OpenCL, METAL, CUDA, AMD, NV, QCOM, WEBGPU), CPU (C Code, LLVM).
*   **Easy to Extend:** Add support for new accelerators with just ~25 low-level ops.

### Ready to get started?

Here's a quick example:
```python
from tinygrad import Tensor

x = Tensor.eye(3, requires_grad=True)
y = Tensor([[2.0,0,-2.0]], requires_grad=True)
z = y.matmul(x).sum()
z.backward()

print(x.grad.tolist())  # dz/dx
print(y.grad.tolist())  # dz/dy
```

### Installation

Install tinygrad from source:

```bash
git clone https://github.com/tinygrad/tinygrad.git
cd tinygrad
python3 -m pip install -e .
```
### Explore the Documentation
For in-depth information, consult the [docs website](https://docs.tinygrad.org/).

### Contributing

We welcome contributions! Please review the [contributing guidelines](https://github.com/tinygrad/tinygrad#contributing) to help your PR get accepted!