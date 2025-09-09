# QuACK: Unleash Blazing-Fast AI Kernel Performance with CuTe-DSL!

[View the original repository on GitHub](https://github.com/Dao-AILab/quack)

QuACK delivers a quirky assortment of CuTe-DSL-powered kernels, optimized for NVIDIA H100 and B200 GPUs, to accelerate your AI workloads.

## Key Features:

*   **High-Performance Kernels:**  Leverage highly optimized kernels for key AI operations.
*   **CuTe-DSL Powered:** Built using the powerful CuTe-DSL for maximum performance and flexibility.
*   **Easy Installation:** Simple `pip install` for quick setup and integration.
*   **Optimized for Modern GPUs:**  Designed to take full advantage of NVIDIA H100 and B200 GPU architectures.
*   **Comprehensive Documentation & Benchmarks:** Detailed blog post explaining memory bound kernels and performance.

## Kernels Included:

*   RMSNorm (Forward & Backward)
*   Softmax (Forward & Backward)
*   Cross Entropy (Forward & Backward)
*   Layernorm (Forward)
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

## Usage Example

```python
from quack import rmsnorm, softmax, cross_entropy
# Example usage of the library's kernels
```

## Performance & Benchmarks

[See our blogpost (media/2025-07-10-membound-sol.md) for detailed benchmarks and performance analysis!](media/2025-07-10-membound-sol.md)

<div align="center">
<figure>
  <img
  src="media/bf16_kernel_benchmarks_single_row.svg"
  alt="Benchmark Performance"
  >
  <figcaption>Kernel Performance Benchmarks (See blog post for details)</figcaption>
</figure>
</div>

## Development

To set up the development environment:

```bash
pip install -e '.[dev]'
pre-commit install
```