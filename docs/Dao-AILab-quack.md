# 🦆 QuACK: Accelerate Your AI with CuTe Kernels 🦆

**Supercharge your AI models with QuACK, a library of high-performance kernels built with the CuTe DSL, designed for optimal performance on NVIDIA GPUs.**

[View the original repository on GitHub](https://github.com/Dao-AILab/quack)

## Key Features

*   **Optimized Kernels:** Pre-built kernels for common AI operations, delivering significant performance gains.
*   **CuTe-DSL Implementation:** Kernels are written using the CuTe DSL, leveraging its power for customization and efficiency.
*   **Easy Integration:** Simple installation and usage within your Python projects.
*   **Comprehensive Documentation:**  Includes a detailed blog post on memory-bound kernel optimization.
*   **High Performance:** Benchmarks demonstrate impressive speedups (see performance section below).

## Kernels Available

QuACK currently provides highly optimized kernels for the following operations:

*   🦆 RMSNorm (forward + backward)
*   🦆 Softmax (forward + backward)
*   🦆 Cross Entropy (forward + backward)
*   🦆 Layernorm (forward)

**Upcoming:**

*   🦆 Rotary (forward + backward)

## Installation

Installing QuACK is easy:

```bash
pip install quack-kernels
```

## Requirements

To use QuACK, you'll need:

*   NVIDIA H100 or B200 GPU
*   CUDA toolkit 12.9+
*   Python 3.12

## Usage

Import and use the kernels directly in your Python code:

```python
from quack import rmsnorm, softmax, cross_entropy
```

## Performance

<div align="center">
<figure>
  <img
  src="media/bf16_kernel_benchmarks_single_row.svg"
  >
</figure>
</div>

For detailed performance analysis and insights into kernel optimization, please refer to our [blogpost](media/2025-07-10-membound-sol.md).

## Development

Set up your development environment with the following commands:

```bash
pip install -e '.[dev]'
pre-commit install