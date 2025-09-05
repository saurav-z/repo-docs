# QuACK: Accelerate Your AI with Quirky, CuTe Kernels 🦆

**Supercharge your AI models with QuACK, a library of high-performance kernels crafted in the CuTe-DSL, optimized for NVIDIA GPUs.**

[View the original repository on GitHub](https://github.com/Dao-AILab/quack)

## Key Features

*   **Optimized Kernels:** Pre-built kernels for common AI operations, delivering significant performance gains.
*   **CuTe-DSL Implementation:** Leverages the CuTe-DSL for efficient and flexible kernel development.
*   **Easy Installation:** Simple pip installation for quick integration into your projects.
*   **Comprehensive Documentation:** Detailed blog post available to deep-dive into the kernel optimizations, and how to harness the power of CuTe-DSL.

## Kernels Included

QuACK provides optimized implementations for the following key operations:

*   🦆 RMSNorm (Forward and Backward)
*   🦆 Softmax (Forward and Backward)
*   🦆 Cross Entropy (Forward and Backward)
*   🦆 Layernorm (Forward)
*   🦆 Rotary (Forward and Backward) - *Coming Soon!*

## Requirements

To utilize the QuACK library, the following is required:

*   NVIDIA H100 or B200 GPU
*   CUDA Toolkit 12.9 or later
*   Python 3.12

## Installation

Install the QuACK library using pip:

```bash
pip install quack-kernels
```

## Usage

Get started with QuACK by importing and utilizing the available kernels:

```python
from quack import rmsnorm, softmax, cross_entropy
```

## Performance

[Insert a representative image of the `bf16_kernel_benchmarks_single_row.svg` here.]

For detailed performance analysis and optimization insights, please refer to our comprehensive [blogpost](media/2025-07-10-membound-sol.md).

## Development

To set up the development environment, execute the following commands:

```bash
pip install -e '.[dev]'
pre-commit install
```