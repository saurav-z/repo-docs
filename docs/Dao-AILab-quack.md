# QuACK: High-Performance CUDA Kernels for AI Acceleration 🚀

**Supercharge your AI workflows with QuACK, a collection of optimized CUDA kernels built using the CuTe-DSL!** [View the original repository](https://github.com/Dao-AILab/quack).

## Key Features

*   **Optimized Kernels:** Provides high-performance implementations of essential AI operations.
*   **Built with CuTe-DSL:** Leverages the power and flexibility of the CuTe Domain-Specific Language for optimal kernel performance.
*   **Easy Installation:** Simple pip install for quick setup.
*   **Comprehensive Coverage:** Includes forward and backward passes for common operations.

## Available Kernels 🐥

*   🦆 RMSNorm (forward + backward)
*   🦆 Softmax (forward + backward)
*   🦆 Cross Entropy (forward + backward)
*   🦆 Layernorm (forward)
*   🦆 Rotary (forward + backward) - *Upcoming*

## Installation

Install QuACK using pip:

```bash
pip install quack-kernels
```

## Requirements

Ensure your system meets these requirements:

*   **GPU:** H100 or B200
*   **CUDA:** CUDA toolkit 12.9+
*   **Python:** Python 3.12

## Usage

Import and utilize QuACK kernels in your Python code:

```python
from quack import rmsnorm, softmax, cross_entropy
```

## Performance & Benchmarks

QuACK kernels are engineered for peak performance. Dive into our benchmark results and understand the optimization strategies in our comprehensive blogpost:

<div align="center">
<figure>
  <img
  src="media/bf16_kernel_benchmarks_single_row.svg"
  >
</figure>
</div>

## Documentation & Blog Post

For detailed insights into the optimization process and performance analysis, check out our blog post: [2025-07-10-membound-sol.md](media/2025-07-10-membound-sol.md) which is all about achieving speed-of-light for memory-bound kernels in Python with CuTe-DSL.

## Development

To set up your development environment:

```bash
pip install -e '.[dev]'
pre-commit install
```