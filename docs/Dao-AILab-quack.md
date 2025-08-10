# 🦆 QuACK: Unleash Lightning-Fast CUDA Kernels with CuTe-DSL 🦆

**Accelerate your machine learning workloads with QuACK, a collection of high-performance CUDA kernels meticulously crafted using the CuTe-DSL!** ([Original Repo](https://github.com/Dao-AILab/quack))

## Key Features

*   **Optimized Kernels:** Experience blazing-fast performance with kernels optimized for modern NVIDIA GPUs.
*   **CuTe-DSL Powered:** Built with the CuTe-DSL for elegant and efficient kernel implementations.
*   **Easy Installation:**  Get started quickly with a simple pip install.
*   **Python-Friendly:**  Seamlessly integrate QuACK into your Python projects.

## Available Kernels 🐥

QuACK currently offers optimized implementations for the following:

*   🦆 RMSNorm (Forward & Backward)
*   🦆 Softmax (Forward & Backward)
*   🦆 Cross Entropy (Forward & Backward)
*   🦆 Layernorm (Forward)

**Upcoming:**

*   🦆 Rotary (Forward & Backward)

## Installation

Install QuACK using pip:

```bash
pip install quack-kernels
```

## Requirements

Ensure your system meets the following requirements:

*   NVIDIA H100 or B200 GPU
*   CUDA Toolkit 12.9+
*   Python 3.12

## Usage

Import and utilize QuACK kernels directly in your Python code:

```python
from quack import rmsnorm, softmax, cross_entropy
```

## Performance

QuACK kernels are engineered for peak performance.  Detailed benchmark results can be found in the blog post linked below.

[<img src="media/bf16_kernel_benchmarks_single_row.svg" alt="Performance Benchmarks" width="600">](media/bf16_kernel_benchmarks_single_row.svg)

## Documentation and Resources

*   **Blog Post:** Dive deeper into the performance optimizations and CuTe-DSL implementation details in our comprehensive [blog post](media/2025-07-10-membound-sol.md).
*   **CuTe-DSL Documentation:** Learn more about the CuTe-DSL at [https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html).

## Development

To set up a development environment for QuACK:

```bash
pip install -e '.[dev]'
pre-commit install
```