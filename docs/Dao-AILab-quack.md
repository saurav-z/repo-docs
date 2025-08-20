# 🦆 QuACK: Unleash Blazing-Fast Kernels for Your Deep Learning Projects 🦆

**QuACK** provides a collection of high-performance, CUDA-accelerated kernels designed for speed and efficiency, all written in the powerful [CuTe-DSL](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html).

[Visit the original repository on GitHub](https://github.com/Dao-AILab/quack)

## Key Features

*   **Optimized Kernels:** Experience lightning-fast performance with kernels optimized for modern NVIDIA GPUs.
*   **CuTe-DSL Implementation:** Kernels are written using the CuTe-DSL, enabling significant performance gains.
*   **Easy Integration:** Seamlessly integrate QuACK kernels into your existing Python projects.

## Available Kernels

QuACK currently offers the following kernels:

*   🦆 RMSNorm (forward + backward)
*   🦆 Softmax (forward + backward)
*   🦆 Cross Entropy (forward + backward)
*   🦆 Layernorm (forward)

**Upcoming:**

*   🦆 Rotary (forward + backward)

## Installation

Get started quickly with a simple pip install:

```bash
pip install quack-kernels
```

## Requirements

Ensure your environment meets these prerequisites:

*   NVIDIA H100 or B200 GPU
*   CUDA Toolkit 12.9+
*   Python 3.12

## Usage

Import and utilize QuACK kernels directly in your Python code:

```python
from quack import rmsnorm, softmax, cross_entropy
```

## Performance

[View benchmark images here]
[Link to media/bf16_kernel_benchmarks_single_row.svg]

**For detailed performance analysis and optimization strategies, explore our comprehensive blog post.**

[Link to Blog Post, media/2025-07-10-membound-sol.md]

## Development

Set up your development environment with these commands:

```bash
pip install -e '.[dev]'
pre-commit install
```