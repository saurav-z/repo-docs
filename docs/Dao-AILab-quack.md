# 🦆 QuACK: Accelerate Your AI Workflows with CuTe Kernels 🦆

**Unleash blazing-fast performance for your AI tasks with QuACK, a library of optimized kernels written in the CuTe-DSL.** ([View on GitHub](https://github.com/Dao-AILab/quack))

## Key Features

*   **Optimized Kernels:** Leverage highly-tuned kernels for common AI operations.
*   **CuTe-DSL:** Built using the CuTe Domain-Specific Language for performance and flexibility.
*   **Easy Installation:** Simple pip installation.
*   **Comprehensive Documentation:** Get detailed explanations and usage examples.
*   **High Performance:** Experience significant speed improvements with memory-bound kernels.

## Kernels

QuACK provides optimized kernels for the following operations:

*   RMSNorm forward + backward
*   Softmax forward + backward
*   Cross entropy forward + backward
*   Layernorm forward
*   (Upcoming) Rotary forward + backward

## Installation

Install QuACK using pip:

```bash
pip install quack-kernels
```

## Requirements

Ensure your environment meets these requirements:

*   H100 or B200 GPU
*   CUDA toolkit 12.9+
*   Python 3.12

## Usage

Import and utilize the kernels in your Python code:

```python
from quack import rmsnorm, softmax, cross_entropy
```

## Performance & Documentation

Learn more about the performance benefits and implementation details in our resources:

*   **Blog Post:** Detailed explanation on optimizing memory-bound kernels:  [blogpost](media/2025-07-10-membound-sol.md) (link in original)
*   **CuTe-DSL Documentation:** Explore the CuTe-DSL used to build these kernels: [CuTe-DSL](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html)

<!-- Image insertion with alt text and caption -->
<div align="center">
<figure>
  <img
  src="media/bf16_kernel_benchmarks_single_row.svg"
  alt="BF16 Kernel Performance Benchmarks"
  >
  <figcaption>BF16 Kernel Performance Benchmarks</figcaption>
</figure>
</div>

## Development

To set up a development environment:

```bash
pip install -e '.[dev]'
pre-commit install
```