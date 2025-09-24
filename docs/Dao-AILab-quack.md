# 🦆 QuACK: Unleash Blazing-Fast Kernels for Your AI Workloads 🦆

**QuACK (Quirky Assortment of CuTe Kernels) delivers highly optimized kernels, written in the CuTe-DSL, designed to accelerate your AI and deep learning applications.**  Explore the power of optimized performance on NVIDIA GPUs with QuACK!  [Visit the original repository](https://github.com/Dao-AILab/quack) for more details.

## Key Features

*   **Optimized Kernels:**  Leverages the CuTe-DSL for performance-critical operations.
*   **Comprehensive Coverage:** Includes a variety of kernels for common AI tasks.
*   **Easy to Use:**  Simple Python interface for seamless integration.
*   **High Performance:** Designed to maximize performance on NVIDIA H100 and B200 GPUs.

## Kernels Included

QuACK provides a diverse set of optimized kernels, including:

*   RMSNorm forward + backward
*   Softmax forward + backward
*   Cross entropy forward + backward
*   Layernorm forward
*   Hopper gemm + epilogue
*   Blackwell gemm + epilogue

## Installation

Get started with QuACK quickly using pip:

```bash
pip install quack-kernels
```

## Requirements

Ensure your environment meets these requirements for optimal performance:

*   NVIDIA H100 or B200 GPU
*   CUDA Toolkit 12.9+
*   Python 3.12

## Usage

Integrate QuACK kernels effortlessly into your Python code:

```python
from quack import rmsnorm, softmax, cross_entropy
```

## Performance & Benchmarks

QuACK is engineered for exceptional performance.

<div align="center">
<figure>
  <img
  src="media/bf16_kernel_benchmarks_single_row.svg"
  alt="Performance Benchmarks"
  >
</figure>
</div>

For detailed performance analysis and insights, please refer to our blogpost.

## Documentation & Resources

*   **CuTe-DSL Documentation:**  Explore the CuTe-DSL used to build these kernels:  [CuTe-DSL Documentation](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html)
*   **Blog Post:** Dive deep into the optimizations in our blogpost: [2025-07-10-membound-sol.md](media/2025-07-10-membound-sol.md)

## Development

To set up a development environment:

```bash
pip install -e '.[dev]'
pre-commit install
```