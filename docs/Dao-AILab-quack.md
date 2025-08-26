# QuACK: Accelerate Your Deep Learning with Cute Kernels 🦆

**QuACK** offers a collection of high-performance, CUDA kernels crafted with the CuTe-DSL to supercharge your deep learning workloads. [Visit the original repository](https://github.com/Dao-AILab/quack) for more details.

## Key Features

*   **Optimized Kernels:** Leverage highly optimized kernels written in the CuTe-DSL for peak performance.
*   **Variety of Kernels:** Includes implementations for:
    *   RMSNorm (forward and backward)
    *   Softmax (forward and backward)
    *   Cross Entropy (forward and backward)
    *   LayerNorm (forward)
*   **Easy Installation:** Simple installation via pip.
*   **Memory Bound Kernel Optimizations**: Achieve impressive speeds for memory-bound kernels right within your Python environment
*   **Upcoming Features:** Rotary forward and backward kernels are planned.

## Installation

```bash
pip install quack-kernels
```

## Requirements

*   NVIDIA H100 or B200 GPU
*   CUDA Toolkit 12.9 or later
*   Python 3.12

## Usage

```python
from quack import rmsnorm, softmax, cross_entropy
```

## Performance & Optimization

[Link to performance graph]

Our kernels are designed for optimal performance.  Refer to our detailed [blog post](media/2025-07-10-membound-sol.md) for in-depth benchmarks and memory bound kernel optimization strategies.

## Development

To contribute to QuACK, set up your development environment:

```bash
pip install -e '.[dev]'
pre-commit install