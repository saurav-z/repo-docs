# QuACK: Unleash Lightning-Fast Kernels for Your AI Workloads 🚀

Accelerate your AI computations with **QuACK**, a library of high-performance kernels written in the cutting-edge [CuTe-DSL](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html), optimized for speed and efficiency.

[View the original repository on GitHub](https://github.com/Dao-AILab/quack)

## Key Features of QuACK

*   **Optimized Kernels:** Leverage highly optimized kernels for key operations, accelerating your AI model training and inference.
*   **Built with CuTe-DSL:** Kernels are crafted using the CuTe-DSL, enabling efficient use of modern GPU hardware.
*   **Easy to Use:** Simple Python interface allows for quick integration into your existing projects.
*   **Focus on Performance:** Designed for maximum performance, with benchmarks demonstrating significant speedups.

## Kernels Included

QuACK currently offers the following kernels:

*   🦆 RMSNorm forward + backward
*   🦆 Softmax forward + backward
*   🦆 Cross entropy forward + backward
*   🦆 Layernorm forward
*   Upcoming: 🦆 Rotary forward + backward

## Installation

Get started quickly with a simple pip install:

```bash
pip install quack-kernels
```

## Requirements

Ensure you have the following to get the most out of QuACK:

*   H100 or B200 GPU
*   CUDA toolkit 12.9+
*   Python 3.12

## Usage Example

Import and use QuACK kernels directly in your Python code:

```python
from quack import rmsnorm, softmax, cross_entropy
```

## Performance and Benchmarks

QuACK is engineered for performance. See the following figure demonstrating the speedups achieved:

<div align="center">
<figure>
  <img
  src="media/bf16_kernel_benchmarks_single_row.svg"
  alt="Performance Benchmark Graph"
  >
</figure>
</div>

For detailed performance analysis, check out our blog post:

*   [Blogpost](media/2025-07-10-membound-sol.md)  (Details on memory-bound kernel optimization)

## Development

Contribute to QuACK's development with the following steps:

```bash
pip install -e '.[dev]'
pre-commit install
```