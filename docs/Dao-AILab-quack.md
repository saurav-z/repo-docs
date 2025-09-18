# 🦆 QuACK: Unleash Blazing-Fast Performance with CuTe Kernels 🦆

**Supercharge your AI workloads with QuACK, a library of high-performance kernels meticulously crafted using the CuTe-DSL.** ([See original repo](https://github.com/Dao-AILab/quack))

## Key Features

*   **Optimized Kernels:** Leverage a suite of highly optimized kernels for critical AI operations.
*   **CuTe-DSL Powered:** Built using the powerful CuTe-DSL for superior performance and flexibility.
*   **Easy Installation:** Get started quickly with a simple pip install.
*   **Comprehensive Coverage:** Includes kernels for RMSNorm, Softmax, Cross Entropy, LayerNorm, and GEMM (Hopper & Blackwell).
*   **Seamless Integration:** Designed for straightforward integration into your existing Python projects.

## Installation

Get started with QuACK in seconds:

```bash
pip install quack-kernels
```

## Requirements

Ensure your environment meets these requirements:

*   NVIDIA H100 or B200 GPU
*   CUDA toolkit 12.9+
*   Python 3.12

## Available Kernels

QuACK provides a rich set of high-performance kernels:

*   🦆 RMSNorm (forward + backward)
*   🦆 Softmax (forward + backward)
*   🦆 Cross Entropy (forward + backward)
*   🦆 Layernorm (forward)
*   🦆 Hopper GEMM + Epilogue
*   🦆 Blackwell GEMM + Epilogue

## Usage

Integrate QuACK kernels into your code effortlessly:

```python
from quack import rmsnorm, softmax, cross_entropy
```

## Performance & Benchmarks

QuACK delivers exceptional performance. Detailed performance analysis and optimization strategies are discussed in our comprehensive blog post.

[Image of benchmark results]

See our [blog post](media/2025-07-10-membound-sol.md) for in-depth analysis.

## Development

Set up your development environment:

```bash
pip install -e '.[dev]'
pre-commit install
```

## Documentation

For detailed information on how to use the CuTe-DSL and our kernels, see our [blogpost](media/2025-07-10-membound-sol.md).