html
<div align="center">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="/docs/logo_tiny_light.svg">
    <img alt="tiny corp logo" src="/docs/logo_tiny_dark.svg" width="50%" height="50%">
  </picture>
</div>

<h1>tinygrad: A Deep Learning Framework That's Surprisingly Powerful</h1>

<p>tinygrad is a lightweight deep learning framework designed for simplicity and flexibility, offering a compelling alternative to larger frameworks.</p>

<h3>
  <a href="https://github.com/tinygrad/tinygrad">Homepage</a> | <a href="https://docs.tinygrad.org/">Documentation</a> | <a href="https://discord.gg/ZjZadyC7PK">Discord</a>
</h3>

[![GitHub Repo stars](https://img.shields.io/github/stars/tinygrad/tinygrad)](https://github.com/tinygrad/tinygrad/stargazers)
[![Unit Tests](https://github.com/tinygrad/tinygrad/actions/workflows/test.yml/badge.svg)](https://github.com/tinygrad/tinygrad/actions/workflows/test.yml)
[![Discord](https://img.shields.io/discord/1068976834382925865)](https://discord.gg/ZjZadyC7PK)

---

<h2>Key Features</h2>

<ul>
  <li><b>Lightweight & Simple:</b> Built for ease of use and understanding, perfect for exploring deep learning concepts.</li>
  <li><b>Supports LLaMA and Stable Diffusion:</b> Run cutting-edge models with tinygrad.</li>
  <li><b>Lazy Evaluation:</b> Experience the power of kernel fusion for optimized performance.</li>
  <li><b>Neural Network Capabilities:</b> Build and train neural networks with a clean and efficient autograd/tensor library, optimizer, and data loader.</li>
  <li><b>Extensive Accelerator Support:</b> Supports a wide range of accelerators including GPU (OpenCL), CPU, LLVM, METAL, CUDA, AMD, NV, QCOM, and WEBGPU, with easy extensibility.</li>
</ul>

<h2>Examples</h2>

<h3>Lazy Evaluation Example</h3>

See how matmul is fused into one kernel:

```bash
DEBUG=3 python3 -c "from tinygrad import Tensor; N = 1024; a, b = Tensor.empty(N, N), Tensor.empty(N, N); (a @ b).realize()"
```
And change `DEBUG` to `4` to see the generated code.

<h3>Neural Network Example</h3>

Build a basic linear neural network:
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

<h2>Accelerators</h2>

tinygrad supports:
- [x] GPU (OpenCL)
- [x] CPU (C Code)
- [x] LLVM
- [x] METAL
- [x] CUDA
- [x] AMD
- [x] NV
- [x] QCOM
- [x] WEBGPU

Check default accelerator: `python3 -c "from tinygrad import Device; print(Device.DEFAULT)"`

<h2>Installation</h2>

Install from source:

```bash
git clone https://github.com/tinygrad/tinygrad.git
cd tinygrad
python3 -m pip install -e .
```

Or install the master branch directly:

```bash
python3 -m pip install git+https://github.com/tinygrad/tinygrad.git
```

<h2>Documentation</h2>

Detailed documentation and a quick start guide are available on the <a href="https://docs.tinygrad.org/">docs website</a>.

<h3>Quick Comparison to PyTorch</h3>

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

<h2>Contributing</h2>

Refer to the [contribution guidelines](https://github.com/tinygrad/tinygrad#contributing) to help ensure your PRs are accepted.

<h2>Further Reading</h2>

*   [Original Repository](https://github.com/tinygrad/tinygrad)