# QuACK: Unleash Blazing-Fast Kernel Performance with CuTe Kernels 🚀

**Accelerate your AI and deep learning workloads with QuACK, a collection of high-performance kernels written in the CuTe-DSL, optimized for modern NVIDIA GPUs.** (See the original repository: [Dao-AILab/quack](https://github.com/Dao-AILab/quack))

## Key Features 🌟

*   **Optimized Kernels:** Harness the power of carefully crafted kernels for critical operations.
*   **CuTe-DSL Based:** Built using the NVIDIA CuTe-DSL for exceptional performance and efficiency.
*   **Easy Integration:** Seamlessly integrate QuACK kernels into your Python projects.
*   **Performance-Driven:** Designed for speed, with benchmarks showcasing significant performance gains.

## Available Kernels 🐥

QuACK currently offers a suite of optimized kernels:

*   🦆 RMSNorm (Forward + Backward)
*   🦆 Softmax (Forward + Backward)
*   🦆 Cross Entropy (Forward + Backward)
*   🦆 LayerNorm (Forward)

**Coming Soon:**

*   🦆 Rotary (Forward + Backward)

## Installation ⚙️

Get started quickly with a simple pip install:

```bash
pip install quack-kernels
```

## Requirements 💡

Ensure your environment meets these specifications:

*   H100 or B200 GPU
*   CUDA Toolkit 12.9+
*   Python 3.12

## Usage 💻

Import the kernels directly into your Python code:

```python
from quack import rmsnorm, softmax, cross_entropy
```

## Performance and Benchmarks 📈

[Include a link to the image file directly here. The HTML is unnecessary.]

See our comprehensive [blogpost](media/2025-07-10-membound-sol.md) for detailed performance analysis and optimization insights.

## Development Setup 🛠️

Set up your development environment with these commands:

```bash
pip install -e '.[dev]'
pre-commit install