# 🦆 QuACK: Accelerate Your AI with Blazing-Fast CuTe Kernels 🦆

**QuACK provides a collection of highly optimized AI kernels, written in the CuTe-DSL, designed to supercharge your deep learning workflows on NVIDIA GPUs.**

[View the original repository on GitHub](https://github.com/Dao-AILab/quack)

## Key Features

*   **Optimized Kernels:** Leverage cutting-edge performance with kernels hand-crafted in the CuTe-DSL.
*   **Comprehensive Coverage:** Includes essential kernels for common deep learning tasks.
*   **Easy Integration:** Simple Python installation and usage.
*   **High Performance:** Achieve significant speedups with memory-bound kernel optimizations.

## Included Kernels

QuACK currently supports the following kernels:

*   🦆 RMSNorm (forward + backward)
*   🦆 Softmax (forward + backward)
*   🦆 Cross Entropy (forward + backward)
*   🦆 Layernorm (forward)

### Upcoming Kernels

*   🦆 Rotary (forward + backward)

## Requirements

Ensure your system meets the following requirements for optimal performance:

*   **GPU:** NVIDIA H100 or B200
*   **CUDA:** CUDA Toolkit 12.9+
*   **Python:** Python 3.12

## Installation

Get started quickly by installing QuACK with pip:

```bash
pip install quack-kernels
```

## Usage

Import and use QuACK kernels directly within your Python code:

```python
from quack import rmsnorm, softmax, cross_entropy
```

## Performance & Benchmarks

<div align="center">
<figure>
  <img
  src="media/bf16_kernel_benchmarks_single_row.svg"
  alt="Benchmark Graph"
  >
</figure>
</div>

For detailed performance analysis and optimization strategies, refer to our comprehensive [blogpost](media/2025-07-10-membound-sol.md).

## Development

To set up the development environment:

```bash
pip install -e '.[dev]'
pre-commit install
```

## Resources

*   **CuTe-DSL Documentation:** Learn more about the CuTe Domain Specific Language: [CuTe-DSL](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html).
*   **Blogpost:** Explore in-depth performance insights and optimization techniques in our [blogpost](media/2025-07-10-membound-sol.md).