html
<div align="center">

<picture>
  <source media="(prefers-color-scheme: light)" srcset="/docs/logo_tiny_light.svg">
  <img alt="tiny corp logo" src="/docs/logo_tiny_dark.svg" width="50%" height="50%">
</picture>

</div>

# tinygrad: A Deep Learning Framework for Everyone

tinygrad is a lightweight deep learning framework designed for simplicity and ease of use, offering a streamlined approach to building and deploying neural networks. **Explore the power of tinygrad and build your own AI solutions with minimal overhead.** Learn more at the [tinygrad GitHub repository](https://github.com/tinygrad/tinygrad).

Key Features:

*   **Minimalist Design:** Get started quickly with a framework that prioritizes simplicity.
*   **Flexible Autograd:** Leverage powerful automatic differentiation capabilities for efficient model training.
*   **Multi-Accelerator Support:** Run your models on a variety of hardware, including:
    *   GPU (OpenCL)
    *   CPU (C Code)
    *   LLVM
    *   METAL
    *   CUDA
    *   AMD
    *   NV
    *   QCOM
    *   WEBGPU
*   **Runs LLaMA and Stable Diffusion**: Leverage the framework to run large-scale projects.
*   **Lazy Evaluation:** Optimize your computations with lazy tensor operations.
*   **Neural Network Capabilities:** Build and train neural networks with ease using autograd, tensor library, optimizers, and data loaders.
*   **Easy Installation:** Get up and running quickly with straightforward installation from source or direct install options.
*   **Comprehensive Documentation:** Access detailed documentation and a quick start guide on the [docs website](https://docs.tinygrad.org/).
*   **Community and Support:** Join the [Discord](https://discord.gg/ZjZadyC7PK) to connect with the community and get help.

## Getting Started

### Installation

Install tinygrad from source:

```bash
git clone https://github.com/tinygrad/tinygrad.git
cd tinygrad
python3 -m pip install -e .
```

Or install directly from master:

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

## Contributing

We welcome contributions! Please review the guidelines to ensure your pull requests are accepted.

### Running Tests

Install testing dependencies:

```bash
python3 -m pip install -e '.[testing]'
```

Run the full test suite:

```bash
python3 -m pytest test/
```

See the [CI workflow](.github/workflows/test.yml) for more testing examples.