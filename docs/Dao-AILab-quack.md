# QuACK: Unleash Blazing-Fast GPU Kernels with CuTe-DSL!

Accelerate your deep learning workloads with **QuACK**, a library of highly optimized GPU kernels built using the cutting-edge [CuTe-DSL](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html), delivering exceptional performance on NVIDIA GPUs.

[View the QuACK repository on GitHub](https://github.com/Dao-AILab/quack)

## Key Features

*   **Optimized Kernels:**  Provides high-performance kernels for common deep learning operations.
*   **Built with CuTe-DSL:** Leverages the power and flexibility of the CuTe-DSL for efficient kernel development.
*   **Easy Installation:**  Simple pip installation for seamless integration into your projects.
*   **Focus on Performance:** Designed for speed, targeting memory-bound kernels for maximum throughput.
*   **Comprehensive Documentation:** Detailed blog post available for in-depth understanding of kernels and performance optimizations.

## Supported Kernels

QuACK offers a range of optimized kernels, including:

*   🦆 RMSNorm forward + backward
*   🦆 Softmax forward + backward
*   🦆 Cross entropy forward + backward
*   🦆 Layernorm forward
*   🦆 Hopper gemm + epilogue
*   🦆 Blackwell gemm + epilogue

## Installation

Get started quickly with a simple pip install:

```bash
pip install quack-kernels
```

## Requirements

Ensure your system meets the following requirements:

*   NVIDIA H100 or B200 GPU
*   CUDA Toolkit 12.9+
*   Python 3.12

## Usage

Integrate QuACK kernels effortlessly into your Python code:

```python
from quack import rmsnorm, softmax, cross_entropy
```

## Performance & Optimization

QuACK kernels are meticulously designed and optimized for maximum performance.  See our detailed [blogpost](media/2025-07-10-membound-sol.md) for insights into how we achieve exceptional speeds, particularly for memory-bound kernels.

[Include image here,  if you can]

<div align="center">
<figure>
  <img
  src="media/bf16_kernel_benchmarks_single_row.svg"
  >
</figure>
</div>

## Development

Contribute to QuACK! Set up your development environment with:

```bash
pip install -e '.[dev]'
pre-commit install
```