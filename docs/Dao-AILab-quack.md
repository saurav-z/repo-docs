# 🦆 QuACK: Unleash Lightning-Fast AI Kernels with CuTe! 🦆

Accelerate your AI workloads with QuACK, a collection of high-performance kernels meticulously crafted using the [CuTe-DSL](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html).  

[**View the original repository on GitHub**](https://github.com/Dao-AILab/quack)

## Key Features

*   **Optimized Kernels:** Featuring a suite of fast kernels engineered for peak performance.
*   **CuTe-DSL Powered:** Built using the CuTe-DSL for efficient kernel development and optimization.
*   **Easy Installation:** Simple pip installation for quick setup.
*   **Comprehensive Documentation:**  Benefit from detailed documentation and insightful blog posts.
*   **Performance Benchmarks:**  See impressive performance gains with our benchmark visualizations.

## Available Kernels

*   RMSNorm (Forward & Backward)
*   Softmax (Forward & Backward)
*   Cross Entropy (Forward & Backward)
*   LayerNorm (Forward)
*   **Upcoming:** Rotary (Forward & Backward)

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

Dive deeper into the inner workings and performance benefits of QuACK through our comprehensive resources:

*   **Blog Post:** Explore how to optimize memory-bound kernels for speed-of-light performance, developed using the [CuTe-DSL](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html).
*   **Performance Benchmarks:**  See our visual benchmarks demonstrating the speed of our kernels:

<div align="center">
<figure>
  <img
  src="media/bf16_kernel_benchmarks_single_row.svg"
  alt="Performance Benchmarks"
  >
</figure>
</div>

## Development

Set up your development environment with:

```bash
pip install -e '.[dev]'
pre-commit install
```