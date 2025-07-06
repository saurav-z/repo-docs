<div align="center">

<picture>
  <source media="(prefers-color-scheme: light)" srcset="/docs/logo_tiny_light.svg">
  <img alt="tiny corp logo" src="/docs/logo_tiny_dark.svg" width="50%" height="50%">
</picture>

</div>

# tinygrad: A Deep Learning Framework for Everyone

**tinygrad is a lightweight, fully-featured deep learning framework designed for simplicity and ease of adding new accelerators.** Explore the power of deep learning without the complexity!  [Check out the original repo](https://github.com/tinygrad/tinygrad).

**Key Features:**

*   **Simplicity:** Easy to understand and modify, making it ideal for both beginners and experts.
*   **Accelerator Support:** Supports a wide range of accelerators, including CPU, GPU (OpenCL, METAL, CUDA, AMD, NV, QCOM, WEBGPU), and LLVM, with more easily added.
*   **LLaMA and Stable Diffusion Ready:** Run cutting-edge models like LLaMA and Stable Diffusion with ease.
*   **Lazy Evaluation:** Optimized performance through lazy evaluation, fusing operations into efficient kernels.
*   **Neural Network Capabilities:** Build and train neural networks with essential components like autograd, tensor libraries, and optimizers.

**Example: Build a Simple Neural Network**

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

**Ready to get started?**

*   **Homepage:** [https://github.com/tinygrad/tinygrad](https://github.com/tinygrad/tinygrad)
*   **Documentation:** [https://docs.tinygrad.org/](https://docs.tinygrad.org/)
*   **Discord:** [https://discord.gg/ZjZadyC7PK](https://discord.gg/ZjZadyC7PK)

[![GitHub Repo stars](https://img.shields.io/github/stars/tinygrad/tinygrad)](https://github.com/tinygrad/tinygrad/stargazers)
[![Unit Tests](https://github.com/tinygrad/tinygrad/actions/workflows/test.yml/badge.svg)](https://github.com/tinygrad/tinygrad/actions/workflows/test.yml)
[![Discord](https://img.shields.io/discord/1068976834382925865)](https://discord.gg/ZjZadyC7PK)

---

## Installation

### From Source

```bash
git clone https://github.com/tinygrad/tinygrad.git
cd tinygrad
python3 -m pip install -e .
```

### Direct (master)

```bash
python3 -m pip install git+https://github.com/tinygrad/tinygrad.git
```

## Contributing

We welcome contributions!  Please review the [Contributing Guidelines](https://github.com/tinygrad/tinygrad/blob/master/README.md#contributing) before submitting a pull request.