# 🦆 QuACK: Accelerate Your AI with Quirky, CuTe Kernels 🦆

**QuACK (Quirky Assortment of CuTe Kernels) provides highly optimized, custom kernels written in the CuTe-DSL, designed to boost the performance of your AI workloads.**

[Visit the original repository on GitHub](https://github.com/Dao-AILab/quack)

## Key Features

*   **Optimized Kernels:** Leveraging the power of CuTe-DSL for maximum performance.
*   **Wide Kernel Coverage:** Includes essential kernels for modern AI tasks.
    *   RMSNorm (forward & backward)
    *   Softmax (forward & backward)
    *   Cross Entropy (forward & backward)
    *   Layernorm (forward)
    *   Hopper GEMM + Epilogue
    *   Blackwell GEMM + Epilogue
*   **Easy Integration:** Simple Python API for seamless integration into your existing projects.

## Installation

Install QuACK with a single pip command:

```bash
pip install quack-kernels
```

## Requirements

*   **GPU:** H100 or B200
*   **CUDA Toolkit:** CUDA toolkit 12.9+
*   **Python:** Python 3.12

## Usage

Import and use QuACK kernels directly in your Python code:

```python
from quack import rmsnorm, softmax, cross_entropy
```

## Performance & Benchmarks

QuACK kernels are designed for peak performance.  Our detailed benchmarks and optimization strategies are available in our blog post.

[Link to Performance Blogpost (example: media/2025-07-10-membound-sol.md)]

<!-- Include an inline image or link to the benchmarking image here -->

## Development

Set up your development environment with the following commands:

```bash
pip install -e '.[dev]'
pre-commit install
```