# QuACK: High-Performance CUDA Kernels for Deep Learning 🦆

**Accelerate your deep learning workloads with QuACK, a collection of highly optimized CUDA kernels written in CuTe-DSL.** [View the original repository on GitHub](https://github.com/Dao-AILab/quack)

## Key Features

*   **Optimized for NVIDIA GPUs:** Specifically designed to leverage the power of H100 and B200 GPUs.
*   **Written in CuTe-DSL:** Utilizes the CuTe Domain Specific Language for efficient kernel development.
*   **Easy to Install:** Get started quickly with a simple pip install.
*   **Comprehensive Kernel Suite:** Includes forward and backward kernels for essential deep learning operations.

## Available Kernels 🐥

QuACK currently offers optimized kernels for the following:

*   🦆 RMSNorm (forward + backward)
*   🦆 Softmax (forward + backward)
*   🦆 Cross Entropy (forward + backward)
*   🦆 LayerNorm (forward)

**Upcoming:**

*   🦆 Rotary (forward + backward)

## Requirements

*   NVIDIA H100 or B200 GPU
*   CUDA Toolkit 12.9+
*   Python 3.12

## Installation

```bash
pip install quack-kernels
```

## Usage

Integrate QuACK kernels seamlessly into your Python code:

```python
from quack import rmsnorm, softmax, cross_entropy
```

## Performance & Benchmarks

QuACK kernels are designed for maximum performance.  For detailed performance analysis and insights, including memory-bound kernel optimization strategies, please refer to our comprehensive [blog post](media/2025-07-10-membound-sol.md).

<div align="center">
<figure>
  <img
  src="media/bf16_kernel_benchmarks_single_row.svg"
  alt="Performance Benchmarks"
  >
  <figcaption>Example Performance Benchmark (See Blog Post for Details)</figcaption>
</figure>
</div>

## Development

Set up your development environment with the following commands:

```bash
pip install -e '.[dev]'
pre-commit install