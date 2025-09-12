# 🦆 QuACK: Unleash Lightning-Fast CUDA Kernels for AI with CuTe-DSL!

[View the original repository on GitHub](https://github.com/Dao-AILab/quack)

## What is QuACK?

QuACK is a collection of highly optimized CUDA kernels, crafted using the powerful [CuTe-DSL](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html) for peak performance on modern NVIDIA GPUs.

## Key Features:

*   **Optimized Kernels:** Pre-built kernels for common AI operations, designed for speed and efficiency.
*   **CuTe-DSL Powered:** Leveraging the CuTe-DSL for flexible and high-performance kernel implementations.
*   **Easy Integration:** Simple Python installation and usage for seamless integration into your AI workflows.
*   **Performance-Driven:** Built for optimal performance, particularly on H100 and B200 GPUs.
*   **Comprehensive Documentation:**  Detailed blog post on how to optimize memory-bound kernels.

## Supported Kernels:

*   RMSNorm (Forward & Backward)
*   Softmax (Forward & Backward)
*   Cross Entropy (Forward & Backward)
*   Layernorm (Forward)
*   Hopper GEMM + Epilogue
*   Blackwell GEMM + Epilogue

## Installation:

Install QuACK kernels with a single pip command:

```bash
pip install quack-kernels
```

## Requirements:

*   NVIDIA H100 or B200 GPU
*   CUDA Toolkit 12.9+
*   Python 3.12

## Usage:

Import and utilize the kernels directly in your Python code:

```python
from quack import rmsnorm, softmax, cross_entropy
```

## Performance:

[Include image of performance](media/bf16_kernel_benchmarks_single_row.svg)

See our [blogpost](media/2025-07-10-membound-sol.md) for in-depth performance analysis and optimization strategies.

## Development:

Set up your development environment with these commands:

```bash
pip install -e '.[dev]'
pre-commit install