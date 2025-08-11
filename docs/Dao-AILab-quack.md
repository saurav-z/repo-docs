# 🦆 QuACK: Blazing-Fast Deep Learning Kernels for Modern GPUs

**QuACK** provides a suite of high-performance, CUDA-accelerated kernels for deep learning, written using the CuTe-DSL, to optimize your model training and inference. [Check out the original repository](https://github.com/Dao-AILab/quack) for the source code!

## Key Features

*   **Optimized Kernels:** Includes highly optimized kernels for common operations, built using the CuTe-DSL.
*   **Ease of Use:** Simple Python interface for seamless integration into your existing workflows.
*   **High Performance:** Designed to leverage the power of modern NVIDIA GPUs for significant speedups.
*   **Memory-Bound Optimization**: Kernel optimizations to get memory bound kernels to run efficiently.

## Available Kernels

*   RMSNorm (forward and backward)
*   Softmax (forward and backward)
*   Cross Entropy (forward and backward)
*   Layernorm (forward)

**Upcoming:**

*   Rotary (forward and backward)

## Installation

Install QuACK kernels with a single pip command:

```bash
pip install quack-kernels
```

## Requirements

*   NVIDIA H100 or B200 GPU
*   CUDA Toolkit 12.9 or later
*   Python 3.12

## Usage

Import and utilize QuACK kernels directly in your Python code:

```python
from quack import rmsnorm, softmax, cross_entropy
```

## Documentation and Performance

For in-depth information and performance analysis, including detailed explanations of memory-bound optimizations, see our blog post [here](media/2025-07-10-membound-sol.md).

<div align="center">
<figure>
  <img
  src="media/bf16_kernel_benchmarks_single_row.svg"
  >
</figure>
</div>

## Development

Set up the development environment to contribute:

```bash
pip install -e '.[dev]'
pre-commit install
```