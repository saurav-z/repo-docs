# QuACK: High-Performance CUDA Kernels for AI 🦆

**Supercharge your AI workloads with QuACK, a collection of optimized CUDA kernels written in the CuTe-DSL.** This library provides a suite of efficient implementations for common AI operations, designed to run blazing fast on NVIDIA GPUs. ([Original Repository](https://github.com/Dao-AILab/quack))

## Key Features

*   **Optimized Kernels:** Implementations written in the CuTe-DSL for maximum performance.
*   **Ease of Use:** Simple Python interface for seamless integration into your projects.
*   **Comprehensive Coverage:** Includes essential kernels for various AI tasks.
*   **Memory-Bound Kernel Optimization:** The project is designed with memory-bound kernels in mind.

### Available Kernels 🐥

*   🦆 RMSNorm forward + backward
*   🦆 Softmax forward + backward
*   🦆 Cross entropy forward + backward
*   🦆 Layernorm forward
*   🦆 Rotary forward + backward (Upcoming)

## Installation

Install the `quack-kernels` package using pip:

```bash
pip install quack-kernels
```

## Requirements

Ensure your system meets the following requirements:

*   NVIDIA H100 or B200 GPU
*   CUDA toolkit 12.9+
*   Python 3.12

## Usage

Import and utilize the kernels in your Python code:

```python
from quack import rmsnorm, softmax, cross_entropy
```

## Documentation and Performance

For detailed information on the CuTe-DSL and the performance optimizations, refer to our comprehensive blog post: [2025-07-10-membound-sol.md](media/2025-07-10-membound-sol.md).

### Performance Benchmarks

[Include image of benchmark result here - as per original README.]

## Development

Set up your development environment with the following commands:

```bash
pip install -e '.[dev]'
pre-commit install
```