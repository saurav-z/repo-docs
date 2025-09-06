# QuACK: High-Performance Kernels for AI Acceleration

**Supercharge your AI workloads with QuACK, a collection of optimized kernels built with the CuTe-DSL.**  [View the original repository on GitHub](https://github.com/Dao-AILab/quack).

## Key Features

*   **Optimized for Speed:** Engineered for peak performance on modern NVIDIA GPUs.
*   **CuTe-DSL Powered:** Built using the CuTe-DSL for highly customizable and efficient kernels.
*   **Easy Integration:** Seamlessly integrate into your existing Python projects.

## Supported Kernels

QuACK currently offers high-performance implementations for the following operations:

*   RMSNorm (Forward and Backward)
*   Softmax (Forward and Backward)
*   Cross Entropy (Forward and Backward)
*   LayerNorm (Forward)

*   **Upcoming:** Rotary (Forward and Backward)

## Installation

Get started quickly with pip:

```bash
pip install quack-kernels
```

## Requirements

Ensure your environment meets these specifications:

*   **GPU:** NVIDIA H100 or B200
*   **CUDA:** CUDA Toolkit 12.9+
*   **Python:** Python 3.12

## Usage

Import and utilize QuACK kernels in your Python code:

```python
from quack import rmsnorm, softmax, cross_entropy
```

## Performance & Memory Bound Kernel Optimization

Explore our comprehensive blog post for in-depth performance analysis and techniques for memory-bound kernel optimization using the CuTe-DSL.

*   [Blogpost: Optimizing for speed of light (2025-07-10)](media/2025-07-10-membound-sol.md)

## Benchmarks

[Include the image here using the image tag]

```html
<div align="center">
<figure>
  <img
  src="media/bf16_kernel_benchmarks_single_row.svg"
  alt="Performance Benchmarks">
</figure>
</div>
```

*   For detailed benchmark results, please refer to the blog post.

## Development

Set up your development environment with these commands:

```bash
pip install -e '.[dev]'
pre-commit install
```