# 🦆 QuACK: High-Performance GPU Kernels for Accelerated AI 🦆

**Supercharge your AI workloads with QuACK, a collection of highly optimized GPU kernels built with the CuTe-DSL, delivering blazing-fast performance for your most demanding tasks.** (Check out the original repository: [https://github.com/Dao-AILab/quack](https://github.com/Dao-AILab/quack))

## Key Features

*   **Optimized Kernels:** Leveraging the power of CuTe-DSL, QuACK provides highly optimized kernels for common AI operations.
*   **Easy Installation:** Simple pip installation for quick integration into your Python projects.
*   **High Performance:** Designed for speed, QuACK kernels provide significant performance gains.
*   **Comprehensive Documentation:** Detailed blog post available with memory-bound kernels speed improvements

## Kernels

QuACK currently includes the following kernels, with more on the way:

*   🦆 RMSNorm (forward + backward)
*   🦆 Softmax (forward + backward)
*   🦆 Cross Entropy (forward + backward)
*   🦆 Layernorm (forward)
*   🦆 Rotary (forward + backward) - *Coming Soon*

## Installation

Get started quickly with a simple pip install:

```bash
pip install quack-kernels
```

## Requirements

Ensure your system meets these requirements for optimal performance:

*   NVIDIA H100 or B200 GPU
*   CUDA Toolkit 12.9+
*   Python 3.12

## Usage

Integrate QuACK kernels seamlessly into your Python code:

```python
from quack import rmsnorm, softmax, cross_entropy
```

## Performance & Benchmarks

![Benchmark Image](media/bf16_kernel_benchmarks_single_row.svg)

For in-depth performance analysis and optimization strategies, please refer to our comprehensive [blogpost](media/2025-07-10-membound-sol.md).

## Development

Set up your development environment with these commands:

```bash
pip install -e '.[dev]'
pre-commit install
```