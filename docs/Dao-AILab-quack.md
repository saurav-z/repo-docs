# QuACK: Accelerate Your AI Workflows with CuTe Kernels 🦆

**Supercharge your AI computations with QuACK, a collection of high-performance kernels written in the CuTe-DSL, optimized for speed and efficiency.**  Explore the QuACK kernels to unlock the full potential of your AI hardware.  See the original repository [here](https://github.com/Dao-AILab/quack).

## Key Features of QuACK

*   **Optimized Performance:**  Leverage kernels written in the CuTe-DSL for blazing-fast AI computations.
*   **Diverse Kernel Suite:** Includes a growing library of essential kernels for deep learning tasks.
*   **Easy Installation:** Install QuACK with a simple pip command.
*   **Python Integration:** Seamlessly integrate QuACK kernels into your Python projects.
*   **Optimized for NVIDIA Hardware:** Designed to take advantage of NVIDIA H100 and B200 GPUs.

## Available Kernels

QuACK offers a range of optimized kernels, with more on the way:

*   🦆 RMSNorm (Forward and Backward)
*   🦆 Softmax (Forward and Backward)
*   🦆 Cross Entropy (Forward and Backward)
*   🦆 Layernorm (Forward)
*   **Upcoming:** 🦆 Rotary (Forward and Backward)

## Installation

Getting started with QuACK is simple:

```bash
pip install quack-kernels
```

## Requirements

Ensure your environment meets the following requirements:

*   NVIDIA H100 or B200 GPU
*   CUDA Toolkit 12.9+
*   Python 3.12

## Usage

Import and utilize QuACK kernels effortlessly within your Python code:

```python
from quack import rmsnorm, softmax, cross_entropy
```

## Documentation & Performance Insights

Dive deeper into the performance benefits and technical details:

*   **Blog Post:**  Explore our comprehensive [blogpost](media/2025-07-10-membound-sol.md) to see how to get memory-bound kernels to speed-of-light, right in the comfort of Python thanks to the [CuTe-DSL](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html).
*   **Performance Benchmarks:**

    <div align="center">
    <figure>
      <img
      src="media/bf16_kernel_benchmarks_single_row.svg"
      >
    </figure>
    </div>

## Development

To set up your development environment:

```bash
pip install -e '.[dev]'
pre-commit install
```