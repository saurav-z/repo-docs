# 🦆 QuACK: Accelerate Your AI Workflows with Optimized CuTe Kernels 🦆

**QuACK (Quirky Assortment of CuTe Kernels) provides a collection of high-performance kernels written in the CuTe-DSL, designed to boost the speed of your AI and deep learning applications.** [Learn more at the original repository](https://github.com/Dao-AILab/quack).

## Key Features

*   **Optimized Kernels:** Leverages the CuTe-DSL for efficient performance on NVIDIA GPUs.
*   **Wide Range of Kernels:** Includes essential kernels for deep learning tasks.
*   **Easy Integration:** Simple `pip` installation and straightforward usage within your Python code.
*   **Performance-Focused:** Designed and optimized for peak performance on supported hardware.

## Kernels Included

*   RMSNorm (forward and backward)
*   Softmax (forward and backward)
*   Cross Entropy (forward and backward)
*   LayerNorm (forward)
*   Hopper GEMM with Epilogue
*   Blackwell GEMM with Epilogue

## Installation

Get started quickly with a simple `pip` install:

```bash
pip install quack-kernels
```

## Requirements

Ensure your system meets these requirements for optimal performance:

*   NVIDIA H100 or B200 GPU
*   CUDA Toolkit 12.9 or newer
*   Python 3.12

## Usage

Import the kernels directly into your Python code:

```python
from quack import rmsnorm, softmax, cross_entropy
```

## Performance

[Include a representative image/chart here - if available]

See the details in our comprehensive blogpost ([2025-07-10-membound-sol.md])!

## Documentation

For in-depth information on how to optimize your memory-bound kernels, check out our detailed blogpost.

*   [Blog Post on Memory-Bound Kernel Optimization](media/2025-07-10-membound-sol.md)

## Development

To contribute to the QuACK project, set up your development environment:

```bash
pip install -e '.[dev]'
pre-commit install
```