<div align="center">

<picture>
  <source media="(prefers-color-scheme: light)" srcset="/docs/logo_tiny_light.svg">
  <img alt="tiny corp logo" src="/docs/logo_tiny_dark.svg" width="50%" height="50%">
</picture>

</div>

# tinygrad: A Deep Learning Framework for Everyone

**Tinygrad is a surprisingly complete deep learning framework that's small, fast, and easy to understand.** Inspired by PyTorch and micrograd, it's designed for simplicity and extensibility. Check out the [GitHub repository](https://github.com/tinygrad/tinygrad) for the latest updates!

### Key Features

*   **Lightweight and Efficient:** tinygrad is designed for minimal overhead, making it ideal for experimentation and resource-constrained environments.
*   **Easy Accelerator Support:** Quickly add support for new hardware accelerators with a few low-level operations.
*   **Full-Featured:** Includes automatic differentiation, a tensor library, and support for common neural network operations.
*   **Runs LLaMA and Stable Diffusion:** Supports complex models like LLaMA and Stable Diffusion.
*   **Lazy Evaluation:** Optimizes computations by fusing operations into efficient kernels, as seen in the matmul example below.
*   **Broad Hardware Support:** Supports a wide range of hardware platforms, including:
    *   GPU (OpenCL)
    *   CPU (C Code)
    *   LLVM
    *   METAL
    *   CUDA
    *   AMD
    *   NV
    *   QCOM
    *   WEBGPU

### Quick Start - Neural Network Example

Build and train neural networks with tinygrad's intuitive API:

```python
from tinygrad import Tensor, nn

class LinearNet:
  def __init__(self):
    self.l1 = Tensor.kaiming_uniform(784, 128)
    self.l2 = Tensor.kaiming_uniform(128, 10)
  def __call__(self, x:Tensor) -> Tensor:
    return x.flatten(1).dot(self.l1).relu().dot(self.l2)

model = LinearNet()
optim = nn.optim.Adam([model.l1, model.l2], lr=0.001)

x, y = Tensor.rand(4, 1, 28, 28), Tensor([2,4,3,7])  # replace with real mnist dataloader

with Tensor.train():
  for i in range(10):
    optim.zero_grad()
    loss = model(x).sparse_categorical_crossentropy(y).backward()
    optim.step()
    print(i, loss.item())
```

See [examples/beautiful_mnist.py](examples/beautiful_mnist.py) for a complete MNIST example.

### Installation

Install tinygrad from source using the following commands:

```sh
git clone https://github.com/tinygrad/tinygrad.git
cd tinygrad
python3 -m pip install -e .
```

Alternatively, install the master branch directly:

```sh
python3 -m pip install git+https://github.com/tinygrad/tinygrad.git
```

### Documentation and Community

*   [Homepage](https://github.com/tinygrad/tinygrad)
*   [Documentation](https://docs.tinygrad.org/)
*   [Discord](https://discord.gg/ZjZadyC7PK)

### Contributing

Contributions are welcome!  Please review the contribution guidelines in the original README.md, linked at the top, to ensure your pull requests are accepted quickly.

---

[![GitHub Repo stars](https://img.shields.io/github/stars/tinygrad/tinygrad)](https://github.com/tinygrad/tinygrad/stargazers)
[![Unit Tests](https://github.com/tinygrad/tinygrad/actions/workflows/test.yml/badge.svg)](https://github.com/tinygrad/tinygrad/actions/workflows/test.yml)
[![Discord](https://img.shields.io/discord/1068976834382925865)](https://discord.gg/ZjZadyC7PK)