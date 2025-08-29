# 🦆 QuACK: Unleash Blazing-Fast Kernels for Your AI Workloads 🦆

Supercharge your AI projects with **QuACK**, a collection of high-performance, CUDA-accelerated kernels written in the cutting-edge [CuTe-DSL](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html).

[View the original repository on GitHub](https://github.com/Dao-AILab/quack)

## Key Features

*   **Optimized Kernels:** Designed for peak performance on NVIDIA GPUs, specifically targeting H100 and B200 architectures.
*   **CuTe-DSL Powered:** Kernels are written in the CuTe-DSL, leveraging its power and flexibility for optimized computation.
*   **Easy Integration:** Simple `pip install` for seamless integration into your Python projects.

## Kernels Included

*   🦆 RMSNorm (Forward & Backward)
*   🦆 Softmax (Forward & Backward)
*   🦆 Cross Entropy (Forward & Backward)
*   🦆 LayerNorm (Forward)
*   🦆 Rotary (Forward & Backward) - *Coming Soon!*

## Installation

Get started with QuACK quickly:

```bash
pip install quack-kernels
```

## Requirements

Ensure you have the following dependencies:

*   NVIDIA H100 or B200 GPU
*   CUDA Toolkit 12.9+
*   Python 3.12

## Usage Example

Integrate QuACK kernels effortlessly into your Python code:

```python
from quack import rmsnorm, softmax, cross_entropy
```

## Performance & Optimization

QuACK kernels are engineered for speed.

**[Blogpost](media/2025-07-10-membound-sol.md)** details our approach to building memory-bound kernels for optimal performance.

<div align="center">
<figure>
  <img
  src="media/bf16_kernel_benchmarks_single_row.svg"
  >
</figure>
</div>

## Development

Set up your development environment:

```bash
pip install -e '.[dev]'
pre-commit install
```