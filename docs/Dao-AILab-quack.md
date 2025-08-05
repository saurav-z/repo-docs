# QuACK: Unleash Blazing-Fast Kernel Performance with CuTe-DSL!

**QuACK** offers a suite of high-performance kernels optimized with the [CuTe-DSL](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html) for your NVIDIA H100/B200 GPU.

[See the original repository on GitHub](https://github.com/Dao-AILab/quack)

## Key Features:

*   **Optimized Kernels:**  Leverages the power of CuTe-DSL for superior performance on modern NVIDIA GPUs.
*   **Ease of Use:** Simple installation and import for quick integration into your projects.
*   **Production-Ready:** Provides essential kernels for common deep learning operations.

## Kernels Included:

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
*   CUDA toolkit 12.9+
*   Python 3.12

## Usage

```python
from quack import rmsnorm, softmax, cross_entropy
```

## Performance & Optimization

Learn how to achieve incredible speeds with memory-bound kernels in our detailed [blog post](media/2025-07-10-membound-sol.md), demonstrating performance improvements achieved with QuACK!

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