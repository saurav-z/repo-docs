# QuACK: Unleash Blazing-Fast Kernels for Deep Learning 🚀

**Supercharge your deep learning workflows with QuACK, a collection of highly optimized kernels written in the CuTe-DSL, designed for peak performance on NVIDIA H100 and B200 GPUs.**  Find the original repo [here](https://github.com/Dao-AILab/quack).

## Key Features

*   **Optimized Kernels:**  Benefit from hand-tuned kernels for critical operations.
*   **CuTe-DSL:**  Leverages the power of the [CuTe-DSL](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html) for efficient kernel design.
*   **CUDA-Accelerated:**  Fully exploits the capabilities of NVIDIA H100 and B200 GPUs.
*   **Python Integration:** Seamlessly integrate QuACK kernels into your Python deep learning projects.

## Kernels Included

QuACK provides high-performance implementations for the following operations:

*   RMSNorm (forward & backward) 🦆
*   Softmax (forward & backward) 🦆
*   Cross Entropy (forward & backward) 🦆
*   LayerNorm (forward)
*   Hopper GEMM + Epilogue
*   Blackwell GEMM + Epilogue

## Installation

Get started with QuACK in seconds:

```bash
pip install quack-kernels
```

## Requirements

Ensure your system meets the following requirements:

*   **GPU:** NVIDIA H100 or B200
*   **CUDA:** CUDA Toolkit 12.9+
*   **Python:** Python 3.12

## Usage

Easily import and utilize QuACK kernels within your Python code:

```python
from quack import rmsnorm, softmax, cross_entropy
```

## Performance Benchmarks

QuACK kernels deliver exceptional performance, as demonstrated in our benchmarks.  See the [blogpost](media/2025-07-10-membound-sol.md) for details and in-depth analysis on the kernels performance.

<div align="center">
<figure>
  <img
  src="media/bf16_kernel_benchmarks_single_row.svg"
  >
</figure>
</div>

## Documentation

For a deep dive into the design and optimization strategies behind QuACK, check out our comprehensive [blogpost](media/2025-07-10-membound-sol.md) which discusses how to optimize memory bound kernels using CuTe-DSL.

## Development

Contribute to QuACK by setting up your development environment:

```bash
pip install -e '.[dev]'
pre-commit install
```