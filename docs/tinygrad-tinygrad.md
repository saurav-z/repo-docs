<div align="center">

<picture>
  <source media="(prefers-color-scheme: light)" srcset="/docs/logo_tiny_light.svg">
  <img alt="tiny corp logo" src="/docs/logo_tiny_dark.svg" width="50%" height="50%">
</picture>

</div>

# tinygrad: A Deep Learning Framework for Everyone

**tinygrad is a minimalist deep learning framework designed for simplicity and easy accelerator integration, offering a fresh perspective on the landscape of AI.**

[View the original repository on GitHub](https://github.com/tinygrad/tinygrad)

### Key Features:

*   **Lightweight and Efficient:** Built for simplicity, tinygrad provides a fast and streamlined deep learning experience.
*   **Broad Accelerator Support:** Supports multiple accelerators including CPU, GPU (OpenCL, CUDA, Metal), LLVM, and more, making it easy to run models on various hardware.
*   **LLaMA and Stable Diffusion:** Run complex models with this lightweight framework.
*   **Lazy Evaluation:** Benefit from optimized kernel fusion through laziness, increasing performance and reducing overhead.
*   **Neural Network Ready:** Includes everything you need for building and training neural networks, including autograd, tensor operations, optimizers, and data loading capabilities.
*   **Easy to Extend:** Simple architecture makes it straightforward to add support for new accelerators.
*   **Easy to Install:** Can be installed from source or directly via pip.
    *   **From Source:** `git clone https://github.com/tinygrad/tinygrad.git` -> `cd tinygrad` -> `python3 -m pip install -e .`
    *   **Direct (master):** `python3 -m pip install git+https://github.com/tinygrad/tinygrad.git`

### Showcase

#### LLaMA and Stable Diffusion
Tinygrad can run [LLaMA](/docs/showcase.md#llama) and [Stable Diffusion](/docs/showcase.md#stable-diffusion)!

### Accelerators

tinygrad already supports numerous accelerators, including:

*   [x] [GPU (OpenCL)](tinygrad/runtime/ops_gpu.py)
*   [x] [CPU (C Code)](tinygrad/runtime/ops_cpu.py)
*   [x] [LLVM](tinygrad/runtime/ops_llvm.py)
*   [x] [METAL](tinygrad/runtime/ops_metal.py)
*   [x] [CUDA](tinygrad/runtime/ops_cuda.py)
*   [x] [AMD](tinygrad/runtime/ops_amd.py)
*   [x] [NV](tinygrad/runtime/ops_nv.py)
*   [x] [QCOM](tinygrad/runtime/ops_qcom.py)
*   [x] [WEBGPU](tinygrad/runtime/ops_webgpu.py)

### Quick Example:

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

### Documentation

Comprehensive documentation and quick start guides are available on the [docs website](https://docs.tinygrad.org/).

### Contributing

We welcome contributions! Here's what we look for:

*   **Bug fixes:** Submit bug fixes with regression tests.
*   **Solving bounties:** Contribute to the project and earn rewards by solving open bounties.
*   **New features:** Implement new features, ensuring they align with the project's simplicity goals, and include thorough tests.
*   **Refactors:** Refactor code for improved readability and efficiency, particularly focusing on clear wins.
*   **Tests and Fuzzers:** Add non-brittle tests and improve existing fuzzers to enhance the library's robustness.

**Please note:** Contributions should adhere to the guidelines outlined in the original README, which includes avoiding code golf, ensuring proper documentation, benchmarking performance improvements, and focusing on core library changes.