# QuACK: Unleash Blazing-Fast CUDA Kernels with CuTe 🚀

**Supercharge your deep learning projects with QuACK, a library of high-performance CUDA kernels meticulously crafted using the CuTe-DSL.**  [See the original repository here](https://github.com/Dao-AILab/quack).

## Key Features:

*   **Optimized Kernels:** Benefit from highly optimized CUDA kernels for common deep learning operations.
*   **CuTe-DSL Powered:**  Leverages the CuTe-DSL for efficient kernel development and customization.
*   **Broad Coverage:** Includes kernels for RMSNorm, Softmax, Cross Entropy, LayerNorm, and GEMM operations.
*   **Hopper and Blackwell Architecture Support:** Optimized for NVIDIA Hopper and Blackwell GPUs.
*   **Easy Integration:**  Simple Python API for seamless integration into your existing workflows.

## Installation:

Get started with QuACK in seconds:

```bash
pip install quack-kernels
```

## Requirements:

Ensure your system meets the following requirements:

*   NVIDIA H100 or B200 GPU
*   CUDA Toolkit 12.9+
*   Python 3.12

## Available Kernels:

QuACK provides a suite of high-performance kernels:

*   🦆 RMSNorm (forward + backward)
*   🦆 Softmax (forward + backward)
*   🦆 Cross Entropy (forward + backward)
*   🦆 Layernorm (forward)
*   🦆 Hopper GEMM + Epilogue
*   🦆 Blackwell GEMM + Epilogue

## Usage Example:

Easily import and use QuACK kernels in your Python code:

```python
from quack import rmsnorm, softmax, cross_entropy
```

## Performance & Benchmarks:

[**Read the blog post**](media/2025-07-10-membound-sol.md)  to learn how to speed up memory-bound kernels thanks to the CuTe-DSL.

<div align="center">
<figure>
  <img
  src="media/bf16_kernel_benchmarks_single_row.svg"
  alt="QuACK Kernel Benchmarks"
  >
</figure>
</div>

## Development:

Contribute to QuACK by setting up your development environment:

```bash
pip install -e '.[dev]'
pre-commit install
```

---
**Keywords:** CUDA, CuTe-DSL, deep learning, machine learning, kernels, optimization, H100, B200, Hopper, Blackwell, RMSNorm, Softmax, Cross Entropy, LayerNorm, GEMM.