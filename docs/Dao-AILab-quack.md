# 🦆 QuACK: Accelerate Your AI Workflows with Quirky, CuTe Kernels! 🦆

**QuACK** provides a collection of high-performance CUDA kernels, written in the [CuTe-DSL](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html), designed to supercharge your deep learning applications.

[View the original repository on GitHub](https://github.com/Dao-AILab/quack)

## Key Features

*   **Optimized Kernels:** Leverages the power of CuTe-DSL for efficient CUDA kernel implementations.
*   **Ready-to-Use:** Easily integrate into your Python projects with a simple `pip install`.
*   **Focus on Core Operations:** Implements essential AI operations with a focus on speed.

## Kernels Included

*   🦆 RMSNorm forward + backward
*   🦆 Softmax forward + backward
*   🦆 Cross entropy forward + backward
*   🦆 Layernorm forward
*   *Upcoming:* 🦆 Rotary forward + backward

## Installation

Get started quickly with:

```bash
pip install quack-kernels
```

## Requirements

Ensure your environment meets these requirements for optimal performance:

*   H100 or B200 GPU
*   CUDA toolkit 12.9+
*   Python 3.12

## Usage Example

Integrate QuACK kernels seamlessly into your Python code:

```python
from quack import rmsnorm, softmax, cross_entropy
```

## Performance & Benchmarks

QuACK kernels are designed for speed.  Our kernels have been carefully crafted and benchmarked to provide maximum performance. For detailed performance analysis, see our blog post!

*   See our [blogpost](media/2025-07-10-membound-sol.md) for benchmarks and details.

## Documentation

Dive deeper into the inner workings and optimization strategies:

*   [Blogpost](media/2025-07-10-membound-sol.md) on memory-bound kernel optimization using CuTe-DSL.

## Development

Contribute to the project or customize your installation with the following steps:

```bash
pip install -e '.[dev]'
pre-commit install
```