# QuACK: High-Performance CuTe Kernels for NVIDIA GPUs 🦆

**Supercharge your deep learning workloads with QuACK, a library of optimized kernels written in the CuTe-DSL for NVIDIA GPUs!** This library provides a collection of high-performance kernels designed to accelerate common operations in modern deep learning models.  Check out the original repository for more details: [https://github.com/Dao-AILab/quack](https://github.com/Dao-AILab/quack)

## Key Features

*   **Optimized Kernels:**  QuACK provides highly optimized kernels for critical deep learning operations, leading to faster training and inference times.
*   **CuTe-DSL Implementation:** Kernels are implemented using the CuTe-DSL, enabling efficient and flexible code generation.
*   **Easy Installation:**  Install QuACK with a simple pip command.
*   **User-Friendly API:**  The library offers an intuitive Python API for seamless integration into your existing projects.

*   **Comprehensive Coverage:** Kernels include:
    *   RMSNorm (forward + backward)
    *   Softmax (forward + backward)
    *   Cross Entropy (forward + backward)
    *   Layernorm (forward)
    *   Hopper GEMM + Epilogue
    *   Blackwell GEMM + Epilogue

## Installation

Install QuACK using pip:

```bash
pip install quack-kernels
```

## Requirements

Ensure your environment meets these requirements for optimal performance:

*   NVIDIA H100 or B200 GPU
*   CUDA Toolkit 12.9 or later
*   Python 3.12

## Usage

Import and utilize the kernels directly in your Python code:

```python
from quack import rmsnorm, softmax, cross_entropy
```

## Documentation and Resources

*   **Blog Post:** Dive deep into the performance optimizations and memory-bound kernel strategies with our comprehensive [blogpost](media/2025-07-10-membound-sol.md).  Learn how to maximize speed in Python using CuTe-DSL.
*   **CuTe-DSL Documentation:** Learn more about the CuTe-DSL used to build these kernels: [CuTe-DSL Documentation](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html)

## Performance

[Image of performance benchmarks - See original README for image details and context]

For detailed performance benchmarks and analysis, please refer to our [blogpost](media/2025-07-10-membound-sol.md).

## Development

To set up a development environment:

```bash
pip install -e '.[dev]'
pre-commit install
```