<div align="center">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="/docs/logo_tiny_light.svg">
    <img alt="tiny corp logo" src="/docs/logo_tiny_dark.svg" width="50%" height="50%">
  </picture>
</div>

# tinygrad: A Deep Learning Framework for Everyone

**tinygrad is a remarkably small deep learning framework, offering a unique balance of simplicity and power, making it ideal for both beginners and experienced practitioners.** Explore the cutting edge of machine learning with tinygrad! ([Original Repo](https://github.com/tinygrad/tinygrad))

### Key Features:

*   **Lightweight and Efficient:** Tinygrad is designed to be minimal, making it easy to understand, modify, and deploy.
*   **Broad Accelerator Support:** Supports a wide range of hardware, including CPU, GPU (OpenCL, METAL, CUDA, AMD, NV, WebGPU), and more, allowing for flexible deployment.
*   **Easy to Extend:** The framework's simplicity makes it straightforward to add support for new accelerators, paving the way for innovation.
*   **Fully Functional:** Despite its small size, tinygrad can run complex models like LLaMA and Stable Diffusion.
*   **Lazy Evaluation:** Smart lazy evaluation optimizes computations, fusing operations for maximum performance.

### Dive Deeper

*   **[Homepage](https://github.com/tinygrad/tinygrad)**
*   **[Documentation](https://docs.tinygrad.org/)**
*   **[Discord](https://discord.gg/ZjZadyC7PK)**

[![GitHub Repo stars](https://img.shields.io/github/stars/tinygrad/tinygrad)](https://github.com/tinygrad/tinygrad/stargazers)
[![Unit Tests](https://github.com/tinygrad/tinygrad/actions/workflows/test.yml/badge.svg)](https://github.com/tinygrad/tinygrad/actions/workflows/test.yml)
[![Discord](https://img.shields.io/discord/1068976834382925865)](https://discord.gg/ZjZadyC7PK)

---

## What You Can Do with tinygrad

### Run Complex Models

Tinygrad shines by running impressive models. For example, it can run:

*   **LLaMA**
*   **Stable Diffusion**

### Explore Lazy Evaluation

```sh
DEBUG=3 python3 -c "from tinygrad import Tensor; N = 1024; a, b = Tensor.empty(N, N), Tensor.empty(N, N); (a @ b).realize()"
```

### Build Neural Networks

Create and train neural networks with ease, leveraging the framework's tensor library, autograd, and optimizers.

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

See [examples/beautiful_mnist.py](examples/beautiful_mnist.py) for the full version that gets 98% in ~5 seconds

## Accelerators

tinygrad supports several accelerators:

*   [x] [GPU (OpenCL)](tinygrad/runtime/ops_gpu.py)
*   [x] [CPU (C Code)](tinygrad/runtime/ops_cpu.py)
*   [x] [LLVM](tinygrad/runtime/ops_llvm.py)
*   [x] [METAL](tinygrad/runtime/ops_metal.py)
*   [x] [CUDA](tinygrad/runtime/ops_cuda.py)
*   [x] [AMD](tinygrad/runtime/ops_amd.py)
*   [x] [NV](tinygrad/runtime/ops_nv.py)
*   [x] [QCOM](tinygrad/runtime/ops_qcom.py)
*   [x] [WEBGPU](tinygrad/runtime/ops_webgpu.py)

To check default accelerator run: `python3 -c "from tinygrad import Device; print(Device.DEFAULT)"`

## Installation

### From Source
```sh
git clone https://github.com/tinygrad/tinygrad.git
cd tinygrad
python3 -m pip install -e .
```

### Direct (master)
```sh
python3 -m pip install git+https://github.com/tinygrad/tinygrad.git
```

## Documentation and Quick Start

Detailed documentation and a quick start guide are available on the [docs website](https://docs.tinygrad.org/).

### Quick Comparison with PyTorch

```python
from tinygrad import Tensor

x = Tensor.eye(3, requires_grad=True)
y = Tensor([[2.0,0,-2.0]], requires_grad=True)
z = y.matmul(x).sum()
z.backward()

print(x.grad.tolist())  # dz/dx
print(y.grad.tolist())  # dz/dy
```

```python
import torch

x = torch.eye(3, requires_grad=True)
y = torch.tensor([[2.0,0,-2.0]], requires_grad=True)
z = y.matmul(x).sum()
z.backward()

print(x.grad.tolist())  # dz/dx
print(y.grad.tolist())  # dz/dy
```

## Contributing

We welcome contributions!  Please review the [contribution guidelines](https://github.com/tinygrad/tinygrad#contributing) before submitting a PR.