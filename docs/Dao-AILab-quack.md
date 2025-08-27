# QuACK: High-Performance CUDA Kernels for AI 🦆

**Accelerate your AI workloads with QuACK, a library of optimized CUDA kernels built using the CuTe-DSL.** ([See the original repo](https://github.com/Dao-AILab/quack))

## Key Features

*   **Optimized Kernels:** Provides highly optimized CUDA kernels for common AI operations.
*   **CuTe-DSL Implementation:** Kernels are written using the CuTe-DSL for efficient performance.
*   **Easy Installation:** Simple `pip install` for quick setup.
*   **Comprehensive Documentation:** Detailed documentation and a blog post are available.

## Available Kernels

*   RMSNorm (forward + backward)
*   Softmax (forward + backward)
*   Cross Entropy (forward + backward)
*   LayerNorm (forward)
*   **Upcoming:** Rotary (forward + backward)

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

## Performance

[Include a brief summary of the performance section. Consider the following:]

*   QuACK kernels deliver significant performance gains, as demonstrated in our benchmarks.
*   See the [blogpost](media/2025-07-10-membound-sol.md) for details on achieving optimal performance.
*   [Include a short description of the image if possible - i.e., graph of performance on different kernels]

## Development

To set up the development environment:

```bash
pip install -e '.[dev]'
pre-commit install