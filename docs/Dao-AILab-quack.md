# QuACK: Accelerate Your AI with CuTe Kernels 🦆

**Supercharge your AI workflows with QuACK, a collection of high-performance kernels written in the innovative CuTe-DSL.**  [View the original repository](https://github.com/Dao-AILab/quack).

## Key Features

*   **Optimized Kernels:** Leveraging the power of CuTe-DSL for blazing-fast performance on modern NVIDIA GPUs.
*   **Comprehensive Coverage:** Includes essential kernels for deep learning applications, with more on the way.
*   **Easy Installation:** Get started in seconds with a simple pip install.
*   **Seamless Integration:** Designed for direct use within your Python projects.

## Available Kernels

QuACK currently offers optimized kernels for:

*   🦆 RMSNorm forward + backward
*   🦆 Softmax forward + backward
*   🦆 Cross entropy forward + backward
*   🦆 Layernorm forward
*   🦆 Upcoming: Rotary forward + backward

## Installation

Installing QuACK is a breeze:

```bash
pip install quack-kernels
```

## Requirements

Ensure you have the following to run QuACK kernels:

*   H100 or B200 GPU (or compatible)
*   CUDA toolkit 12.9+
*   Python 3.12

## Usage

Integrate QuACK kernels directly into your Python code:

```python
from quack import rmsnorm, softmax, cross_entropy
```

## Performance & Benchmarks

QuACK kernels are designed for exceptional performance.  For detailed benchmarks and insights into optimizing memory-bound kernels, check out our comprehensive blog post.

<!-- Example of embedding an image.  Replace with an image URL if it isn't working.  I did not include a link. -->

<div align="center">
  <figure>
    <img src="media/bf16_kernel_benchmarks_single_row.svg" alt="QuACK Kernel Performance Benchmarks">
  </figure>
</div>

## Documentation & Resources

*   **CuTe-DSL Documentation:**  Learn more about the underlying CuTe-DSL technology: [https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html)
*   **Blog Post:** Dive deeper into the performance benefits and implementation details:  [media/2025-07-10-membound-sol.md] (Placeholder. Needs a real link)

## Development

To set up a development environment:

```bash
pip install -e '.[dev]'
pre-commit install
```