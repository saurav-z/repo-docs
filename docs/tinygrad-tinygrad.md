<div align="center">

<picture>
  <source media="(prefers-color-scheme: light)" srcset="/docs/logo_tiny_light.svg">
  <img alt="tiny corp logo" src="/docs/logo_tiny_dark.svg" width="50%" height="50%">
</picture>

</div>

# tinygrad: A Deep Learning Framework for the Curious

**Tinygrad is a lightweight deep learning framework designed for simplicity and ease of use, offering a fresh perspective on neural network development.** Explore the [original repository](https://github.com/tinygrad/tinygrad) for the latest updates.

[Homepage](https://github.com/tinygrad/tinygrad) | [Documentation](https://docs.tinygrad.org/) | [Discord](https://discord.gg/ZjZadyC7PK)

[![GitHub Repo stars](https://img.shields.io/github/stars/tinygrad/tinygrad)](https://github.com/tinygrad/tinygrad/stargazers)
[![Unit Tests](https://github.com/tinygrad/tinygrad/actions/workflows/test.yml/badge.svg)](https://github.com/tinygrad/tinygrad/actions/workflows/test.yml)
[![Discord](https://img.shields.io/discord/1068976834382925865)](https://discord.gg/ZjZadyC7PK)

---

## Key Features of tinygrad

*   **Simplicity and Ease of Use:**  Built for rapid prototyping and exploration, tinygrad provides a streamlined experience for both beginners and experienced deep learning practitioners.
*   **Supports LLaMA and Stable Diffusion:** Run complex models with ease.  Tinygrad is capable of running popular models like [LLaMA](/docs/showcase.md#llama) and [Stable Diffusion](/docs/showcase.md#stable-diffusion).
*   **Lazy Evaluation:**  Experience efficient computation with tinygrad's laziness, optimizing your operations into fused kernels.
*   **Neural Network Support:** A complete autograd/tensor library with an optimizer and data loader that covers most of the common deep learning use cases.
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
*   **Wide Accelerator Support:**  Easily deploy your models across a variety of hardware backends:
    *   [x] [GPU (OpenCL)](tinygrad/runtime/ops_gpu.py)
    *   [x] [CPU (C Code)](tinygrad/runtime/ops_cpu.py)
    *   [x] [LLVM](tinygrad/runtime/ops_llvm.py)
    *   [x] [METAL](tinygrad/runtime/ops_metal.py)
    *   [x] [CUDA](tinygrad/runtime/ops_cuda.py)
    *   [x] [AMD](tinygrad/runtime/ops_amd.py)
    *   [x] [NV](tinygrad/runtime/ops_nv.py)
    *   [x] [QCOM](tinygrad/runtime/ops_qcom.py)
    *   [x] [WEBGPU](tinygrad/runtime/ops_webgpu.py)

    Easily add new accelerators by supporting ~25 low-level ops.

*   **Easy to Install:** Install from source for the best experience.
    ```sh
    git clone https://github.com/tinygrad/tinygrad.git
    cd tinygrad
    python3 -m pip install -e .
    ```
    Alternatively, install the latest master build:
    ```sh
    python3 -m pip install git+https://github.com/tinygrad/tinygrad.git
    ```

## Getting Started

For detailed information on getting started with tinygrad, consult the [docs website](https://docs.tinygrad.org/).

### Quick Example: Comparing to PyTorch

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

## Contributing

Contributions are welcome!  Please review the [Contributing Guidelines](https://github.com/tinygrad/tinygrad/blob/master/README.md#contributing) to learn about how to contribute.