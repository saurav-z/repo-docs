# QuACK: Unleash Lightning-Fast Kernel Performance with CuTe!

[View the original repository on GitHub](https://github.com/Dao-AILab/quack)

QuACK (Quirky Assortment of CuTe Kernels) provides optimized, high-performance kernels for your deep learning needs, built using the innovative CuTe-DSL.

## Key Features

*   **Optimized Kernels:** Achieve superior performance with kernels meticulously crafted for speed and efficiency.
*   **CuTe-DSL Powered:** Leverages the power of CuTe-DSL for concise, performant kernel development.
*   **Easy Installation:** Simple pip installation for seamless integration into your projects.
*   **Comprehensive Coverage:** Includes essential kernels for common deep learning operations.
*   **Cross-Entropy Support:** Optimized cross-entropy calculation for efficient loss computation.

## Kernels Included

*   RMSNorm (Forward & Backward)
*   Softmax (Forward & Backward)
*   Cross Entropy (Forward & Backward)
*   LayerNorm (Forward)
*   Hopper GEMM + Epilogue
*   Blackwell GEMM + Epilogue

## Installation

```bash
pip install quack-kernels
```

## Requirements

*   NVIDIA H100 or B200 GPU
*   CUDA Toolkit 12.9+
*   Python 3.12

## Usage

Easily integrate QuACK kernels into your Python code:

```python
from quack import rmsnorm, softmax, cross_entropy
```

## Performance & Benchmarks

<div align="center">
<figure>
  <img
  src="media/bf16_kernel_benchmarks_single_row.svg"
  alt="QuACK Kernel Performance Benchmarks"
  >
  <figcaption>Benchmark results showcasing the performance of QuACK kernels.</figcaption>
</figure>
</div>

**[Blog Post](media/2025-07-10-membound-sol.md):** Dive deep into the performance details and discover how to optimize memory-bound kernels.

## Development

Set up your development environment:

```bash
pip install -e '.[dev]'
pre-commit install
```

## Documentation

For in-depth information and insights, explore our comprehensive [blogpost](media/2025-07-10-membound-sol.md) detailing memory-bound kernel optimization using the CuTe-DSL.