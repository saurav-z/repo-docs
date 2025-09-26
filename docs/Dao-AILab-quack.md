# QuACK: Unleash Blazing-Fast Kernels for Your AI Workloads 🚀

**QuACK (Quirky Assortment of CuTe Kernels)** provides optimized, high-performance kernels written in the CuTe-DSL, empowering you to accelerate your AI computations.  [View the original repository on GitHub](https://github.com/Dao-AILab/quack).

## Key Features

*   **Optimized Kernels:**  Leverage hand-tuned kernels designed for peak performance.
*   **CuTe-DSL Powered:** Kernels are built using the CuTe-DSL, a powerful language for kernel development.
*   **Easy Installation:**  Simple `pip install` for quick setup.
*   **Broad Kernel Support:** Includes essential kernels for modern AI:
    *   RMSNorm (forward & backward)
    *   Softmax (forward & backward)
    *   Cross Entropy (forward & backward)
    *   Layernorm (forward)
    *   Hopper GEMM + Epilogue
    *   Blackwell GEMM + Epilogue
*   **Python Integration:** Seamlessly integrate QuACK kernels into your Python projects.
*   **Performance Focused:** Optimized for the latest NVIDIA GPUs, specifically H100 and B200.

## Installation

```bash
pip install quack-kernels
```

## Requirements

*   NVIDIA H100 or B200 GPU
*   CUDA Toolkit 12.9+
*   Python 3.12

## Usage

```python
from quack import rmsnorm, softmax, cross_entropy
```

## Performance & Benchmarks

QuACK delivers exceptional performance. Explore the details in our blogpost for in-depth performance analysis and optimization strategies.

<div align="center">
<figure>
  <img
  src="media/bf16_kernel_benchmarks_single_row.svg"
  alt="QuACK Kernel Performance Benchmarks"
  >
</figure>
</div>

For detailed performance analysis and how to optimize memory-bound kernels, see our comprehensive [blogpost](media/2025-07-10-membound-sol.md).

## Development

To set up a development environment:

```bash
pip install -e '.[dev]'
pre-commit install
```

---

**Keywords:** QuACK, CuTe, CuTe-DSL, CUDA, kernels, AI, machine learning, deep learning, H100, B200, RMSNorm, Softmax, Cross Entropy, Layernorm, GEMM, Hopper, Blackwell, performance optimization, Python, accelerated computing.