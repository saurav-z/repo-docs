# 🦆 QuACK: Unleash Lightning-Fast Kernels with CuTe 🦆

**Supercharge your AI workloads with QuACK, a collection of optimized kernels written in the CuTe-DSL, delivering unparalleled performance for your H100 and B200 GPUs!**  For the original repository, see [Dao-AILab/quack](https://github.com/Dao-AILab/quack).

## Key Features

*   **Optimized Kernels:** Pre-built kernels for common AI operations, including:
    *   RMSNorm (forward & backward)
    *   Softmax (forward & backward)
    *   Cross Entropy (forward & backward)
    *   LayerNorm (forward)
*   **CuTe-DSL Powered:** Kernels are built using the CuTe-DSL, enabling high performance and customization.
*   **Easy Installation:** Simple pip installation for quick setup.
*   **Python Integration:** Seamlessly integrate QuACK kernels into your Python projects.
*   **Future-Proof:** Actively developed with upcoming features, including Rotary kernels.

## Installation

Get started with QuACK in seconds:

```bash
pip install quack-kernels
```

## Requirements

Ensure your environment meets the following:

*   NVIDIA H100 or B200 GPU
*   CUDA Toolkit 12.9 or later
*   Python 3.12

## Usage

Easily integrate QuACK kernels into your Python code:

```python
from quack import rmsnorm, softmax, cross_entropy
```

## Performance and Benchmarks

QuACK kernels are designed for exceptional speed. Detailed performance insights are available in our blog post.

<div align="center">
<figure>
  <img
  src="media/bf16_kernel_benchmarks_single_row.svg"
  alt="QuACK Kernel Benchmarks"
  >
</figure>
</div>

## Documentation and Blog Post

Dive deeper into the architecture and performance of QuACK in our comprehensive [blogpost](media/2025-07-10-membound-sol.md) which discusses how to leverage the CuTe-DSL for memory-bound kernels to deliver lightning-fast performance.

## Development

Contribute and customize QuACK with ease:

```bash
pip install -e '.[dev]'
pre-commit install
```