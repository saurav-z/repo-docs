# QuACK: High-Performance CUDA Kernels for AI and Deep Learning 🦆

**Supercharge your AI and deep learning projects with QuACK, a collection of optimized CUDA kernels for lightning-fast performance.** (See the original repository here: [https://github.com/Dao-AILab/quack](https://github.com/Dao-AILab/quack))

## Key Features of QuACK

*   **Optimized for NVIDIA GPUs:** Built using the CuTe-DSL for maximum performance on H100 and B200 GPUs.
*   **Comprehensive Kernel Suite:** Includes crucial kernels for deep learning tasks.
    *   RMSNorm (forward & backward)
    *   Softmax (forward & backward)
    *   Cross Entropy (forward & backward)
    *   LayerNorm (forward)
    *   Hopper GEMM + Epilogue
    *   Blackwell GEMM + Epilogue
*   **Easy to Use:** Simple Python interface for seamless integration into your existing projects.

## Installation

Install QuACK kernels with a single command:

```bash
pip install quack-kernels
```

## Requirements

*   NVIDIA H100 or B200 GPU
*   CUDA Toolkit 12.9+
*   Python 3.12

## Usage Example

Import and utilize QuACK kernels effortlessly:

```python
from quack import rmsnorm, softmax, cross_entropy
```

## Performance & Benchmarks

QuACK kernels are designed for speed, offering significant performance gains. See detailed benchmarks and analysis in our blog post:

<div align="center">
<figure>
  <img
  src="media/bf16_kernel_benchmarks_single_row.svg"
  >
</figure>
</div>

For in-depth performance insights, check out our blog: [blogpost](media/2025-07-10-membound-sol.md).

## Development

To set up a development environment:

```bash
pip install -e '.[dev]'
pre-commit install
```

## Resources

*   **CuTe-DSL Documentation:** Learn more about the underlying DSL: [https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html)
*   **Original Repository:** [https://github.com/Dao-AILab/quack](https://github.com/Dao-AILab/quack)