## 🦆 QuACK: Unleash Blazing-Fast Kernel Performance with CuTe-DSL 🦆

Supercharge your CUDA applications with **QuACK**, a collection of optimized kernels designed for speed and efficiency, all crafted with the power of the CuTe-DSL.  For more details, visit the original repository: [https://github.com/Dao-AILab/quack](https://github.com/Dao-AILab/quack).

### Key Features:

*   **High-Performance Kernels:** Pre-optimized kernels for critical operations, maximizing performance on NVIDIA GPUs.
*   **CuTe-DSL Powered:** Built using the [CuTe-DSL](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html) for efficient and flexible kernel development.
*   **Easy Installation:** Simple pip install for quick setup and integration.
*   **Comprehensive Coverage:** Kernels for essential operations like RMSNorm, Softmax, Cross Entropy, LayerNorm, and GEMM (Hopper & Blackwell architectures).

### Installation

Get started quickly with the following pip command:

```bash
pip install quack-kernels
```

### Requirements

Ensure your system meets the following prerequisites:

*   **GPU:** NVIDIA H100 or B200
*   **CUDA:** CUDA Toolkit 12.9+
*   **Python:** Python 3.12

### Available Kernels

QuACK provides a range of optimized kernels for various compute-intensive tasks:

*   🦆 RMSNorm (forward and backward)
*   🦆 Softmax (forward and backward)
*   🦆 Cross Entropy (forward and backward)
*   🦆 Layernorm (forward)
*   🦆 Hopper GEMM + Epilogue
*   🦆 Blackwell GEMM + Epilogue

### Usage

Import and utilize the kernels directly in your Python code:

```python
from quack import rmsnorm, softmax, cross_entropy
```

### Documentation and Resources

Explore the performance gains and implementation details in our comprehensive resources:

*   **Blog Post:** Learn how to optimize memory-bound kernels with CuTe-DSL:  [media/2025-07-10-membound-sol.md](media/2025-07-10-membound-sol.md)
*   **CuTe-DSL Documentation:** Refer to the official documentation: [https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html)

### Performance Benchmarks

[Insert Image]
[Image of performance benchmarks]

### Development

Set up a development environment for contributing to QuACK:

```bash
pip install -e '.[dev]'
pre-commit install