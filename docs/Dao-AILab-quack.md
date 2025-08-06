# QuACK: Accelerate Your AI with Optimized CUDA Kernels 🦆

**QuACK provides a collection of high-performance CUDA kernels, written in the CuTe-DSL, designed to boost the performance of your AI applications.**

[Visit the original repository on GitHub](https://github.com/Dao-AILab/quack)

## Key Features

*   **Optimized Kernels:**  Leverages the CuTe-DSL for efficient CUDA kernel implementation.
*   **Comprehensive Kernel Suite:** Includes essential kernels for common AI tasks.
*   **Easy to Use:** Simple Python interface for seamless integration into your projects.
*   **High Performance:** Designed for optimal performance on NVIDIA H100 and B200 GPUs.

## Available Kernels

*   🦆 RMSNorm forward + backward
*   🦆 Softmax forward + backward
*   🦆 Cross entropy forward + backward
*   🦆 Layernorm forward
*   🦆 Rotary forward + backward (Upcoming)

## Installation

```bash
pip install quack-kernels
```

## Requirements

*   NVIDIA H100 or B200 GPU
*   CUDA Toolkit 12.9+
*   Python 3.12

## Usage

```python
from quack import rmsnorm, softmax, cross_entropy
```

## Documentation & Performance

*   **Blogpost:** Explore a detailed blog post that describes how to optimize memory-bound kernels for peak performance with CuTe-DSL.
    *   [Blogpost Link](media/2025-07-10-membound-sol.md)
*   **Performance Benchmarks:**  See visual performance data in our benchmarks.
    <div align="center">
    <figure>
      <img
      src="media/bf16_kernel_benchmarks_single_row.svg"
      >
    </figure>
    </div>

## Development

To set up the development environment:

```bash
pip install -e '.[dev]'
pre-commit install
```