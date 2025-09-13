# QuACK: Unleash Lightning-Fast GPU Kernels with CuTe 🦆

**Supercharge your deep learning projects with QuACK, a collection of high-performance, CUDA-accelerated kernels built using the innovative CuTe-DSL!**

[View the original repository on GitHub](https://github.com/Dao-AILab/quack)

## Key Features

*   **Optimized Kernels:** Benefit from a suite of meticulously crafted kernels for common deep learning operations.
*   **Built with CuTe-DSL:** Leveraging the power of the CuTe-DSL for efficient kernel development and optimization.
*   **Easy Installation:** Get up and running quickly with a simple pip install.
*   **Cutting-Edge Performance:** Achieve blazing-fast performance with kernels optimized for modern NVIDIA GPUs.
*   **Comprehensive Coverage:** Includes kernels for RMSNorm, Softmax, Cross Entropy, LayerNorm, GEMM (Hopper & Blackwell) and more.

## Included Kernels

*   RMSNorm forward + backward
*   Softmax forward + backward
*   Cross entropy forward + backward
*   Layernorm forward
*   Hopper gemm + epilogue
*   Blackwell gemm + epilogue

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

## Performance & Optimization

**Experience exceptional performance gains:**  QuACK kernels are designed to maximize GPU utilization and minimize memory bottlenecks.

*   **Memory-Bound Kernel Optimization:** Dive deep into the performance details and optimization techniques in our comprehensive [blogpost](media/2025-07-10-membound-sol.md) explaining how to achieve speed-of-light performance.

<div align="center">
<figure>
  <img
  src="media/bf16_kernel_benchmarks_single_row.svg"
  >
</figure>
</div>

## Development

To set up the development environment:

```bash
pip install -e '.[dev]'
pre-commit install