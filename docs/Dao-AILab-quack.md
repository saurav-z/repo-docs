# 🦆 QuACK: Accelerate Your AI Workloads with Optimized CuTe Kernels 🦆

**QuACK provides a suite of high-performance, memory-bound AI kernels, all crafted using the CuTe-DSL for ultimate efficiency.**

[View the original repository on GitHub](https://github.com/Dao-AILab/quack)

## Key Features

*   **Optimized Kernels:** Leverages the power of the CuTe-DSL to deliver lightning-fast kernel performance.
*   **Comprehensive Kernel Suite:**
    *   🦆 RMSNorm (forward + backward)
    *   🦆 Softmax (forward + backward)
    *   🦆 Cross Entropy (forward + backward)
    *   🦆 LayerNorm (forward)
    *   🦆 **Upcoming:** Rotary (forward + backward)
*   **Easy to Use:** Simple Python interface for seamless integration into your existing projects.

## Installation

Get started with QuACK in minutes:

```bash
pip install quack-kernels
```

## Requirements

Ensure your environment meets these prerequisites:

*   H100 or B200 GPU
*   CUDA toolkit 12.9+
*   Python 3.12

## Usage

Import and utilize QuACK kernels effortlessly:

```python
from quack import rmsnorm, softmax, cross_entropy
```

## Performance and Benchmarks

Explore the performance gains achieved with QuACK.  For detailed insights, see the blogpost.

<div align="center">
<figure>
  <img
  src="media/bf16_kernel_benchmarks_single_row.svg"
  >
</figure>
</div>

## Documentation and Blog Post

*   Dive deep into the technical details and performance optimizations in our blog post.

    *   [Blogpost](media/2025-07-10-membound-sol.md) (details on memory-bound kernels)
*   Learn more about CuTe-DSL: [CuTe-DSL Documentation](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html)

## Development

Set up your development environment with these commands:

```bash
pip install -e '.[dev]'
pre-commit install
```