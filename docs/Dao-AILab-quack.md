# 🦆 QuACK: Supercharge Your Deep Learning with Lightning-Fast Kernels 🦆

**QuACK (Quirky Assortment of CuTe Kernels) is a collection of high-performance, CUDA-accelerated kernels, meticulously crafted for speed and efficiency, using the CuTe-DSL.**  Visit the [original repository on GitHub](https://github.com/Dao-AILab/quack) for more details and to contribute.

## Key Features

*   **Optimized Kernels:**  Leverages the CuTe-DSL for blazing-fast performance on NVIDIA GPUs.
*   **Variety of Functions:** Includes essential kernels for deep learning tasks.
*   **Easy Installation:**  Simple `pip` install for quick setup.
*   **Production Ready:** Ready to integrate into your existing deep learning pipelines.

## Kernels Currently Available

*   🦆 RMSNorm forward + backward
*   🦆 Softmax forward + backward
*   🦆 Cross entropy forward + backward
*   🦆 Layernorm forward

**Coming Soon:**

*   🦆 Rotary forward + backward

## Requirements

*   NVIDIA H100 or B200 GPU
*   CUDA Toolkit 12.9 or later
*   Python 3.12

## Installation

```bash
pip install quack-kernels
```

## Usage Example

```python
from quack import rmsnorm, softmax, cross_entropy
```

## Performance and Benchmarking

Explore detailed performance insights and benchmark results in our comprehensive [blog post](media/2025-07-10-membound-sol.md), where we delve into optimizing memory-bound kernels for peak efficiency, all within the Python environment, thanks to the CuTe-DSL.

<div align="center">
<figure>
  <img
  src="media/bf16_kernel_benchmarks_single_row.svg"
  alt="Kernel Performance Benchmarks"
  >
</figure>
</div>

## Development

To set up your development environment:

```bash
pip install -e '.[dev]'
pre-commit install
```