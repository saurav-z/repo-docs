# QuACK: High-Performance CuTe Kernels for NVIDIA GPUs 🦆

**Supercharge your AI workloads with QuACK, a collection of optimized kernels meticulously crafted using the CuTe-DSL for lightning-fast performance on NVIDIA H100 and B200 GPUs.**

[View the original repository on GitHub](https://github.com/Dao-AILab/quack)

## Key Features

*   **Optimized Kernels:**  Leverage highly optimized kernels for common AI operations.
*   **CuTe-DSL Based:**  Written in the CuTe-DSL for efficient GPU utilization.
*   **Easy Installation:** Simple pip installation for quick setup.
*   **Comprehensive Coverage:**  Includes kernels for:
    *   RMSNorm (forward & backward)
    *   Softmax (forward & backward)
    *   Cross Entropy (forward & backward)
    *   Layernorm (forward)
    *   Hopper GEMM + Epilogue
    *   Blackwell GEMM + Epilogue

## Installation

Install QuACK kernels using pip:

```bash
pip install quack-kernels
```

## Requirements

Ensure your system meets the following requirements:

*   **GPU:** NVIDIA H100 or B200
*   **CUDA:** CUDA Toolkit 12.9 or higher
*   **Python:** Python 3.12

## Usage

Import and use QuACK kernels directly in your Python code:

```python
from quack import rmsnorm, softmax, cross_entropy
```

## Performance & Benchmarks

QuACK kernels are designed for exceptional performance.  Detailed performance analysis and optimization strategies are available in our blog post.

[<img src="media/bf16_kernel_benchmarks_single_row.svg">](media/bf16_kernel_benchmarks_single_row.svg)

*   **Learn more**:  Read our comprehensive [blog post](media/2025-07-10-membound-sol.md) for in-depth insights into kernel optimization and memory-bound performance.

## Development

Set up your development environment:

```bash
pip install -e '.[dev]'
pre-commit install
```

---