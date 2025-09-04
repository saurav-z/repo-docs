# 🦆 QuACK: Unleash Blazing-Fast Kernels with CuTe-DSL 🦆

**Supercharge your deep learning workloads with QuACK, a collection of high-performance kernels optimized using the CuTe-DSL!**  For the original source, visit the [Quack GitHub Repository](https://github.com/Dao-AILab/quack).

## Key Features

*   **Optimized Kernels:** Enjoy significant performance gains with kernels meticulously crafted using the CuTe-DSL.
*   **Ready-to-Use:** Simple installation and integration with your Python projects.
*   **Cutting-Edge:** Leverages the latest advancements in CUDA and hardware capabilities.
*   **Comprehensive Support:** Includes essential kernels for common deep learning operations.

## Kernels Included

QuACK offers a selection of high-performance kernels, including:

*   🦆 RMSNorm (forward + backward)
*   🦆 Softmax (forward + backward)
*   🦆 Cross Entropy (forward + backward)
*   🦆 LayerNorm (forward)

## Upcoming Kernels

*   🦆 Rotary (forward + backward)

## Installation

Get started quickly with a simple pip install:

```bash
pip install quack-kernels
```

## Requirements

Ensure your environment meets these specifications for optimal performance:

*   **GPU:** H100 or B200
*   **CUDA:** CUDA Toolkit 12.9+
*   **Python:** Python 3.12

## Usage Example

Integrate QuACK kernels seamlessly into your Python code:

```python
from quack import rmsnorm, softmax, cross_entropy
```

## Performance & Benchmarks

[Link to image of benchmark]

For in-depth performance analysis and optimization strategies, explore our detailed [blog post](media/2025-07-10-membound-sol.md) (coming soon).  Learn how we leverage the CuTe-DSL to achieve memory-bound kernel performance.

## Development Environment Setup

Contribute to QuACK development with these steps:

```bash
pip install -e '.[dev]'
pre-commit install
```