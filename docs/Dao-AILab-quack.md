# QuACK: Supercharge Your AI with Optimized CuTe Kernels 🦆

**Accelerate your AI workloads with QuACK, a collection of high-performance kernels crafted with the CuTe-DSL for NVIDIA GPUs.** [See the original repository](https://github.com/Dao-AILab/quack).

## Key Features:

*   **Optimized Kernels:** Built with the CuTe-DSL for maximum performance on NVIDIA H100 and B200 GPUs.
*   **Ready-to-Use:** Easily integrate QuACK kernels into your Python projects.
*   **Current Kernels:**
    *   🦆 RMSNorm (forward + backward)
    *   🦆 Softmax (forward + backward)
    *   🦆 Cross Entropy (forward + backward)
    *   🦆 Layernorm (forward)
*   **Upcoming:**
    *   🦆 Rotary (forward + backward)
*   **CuTe-DSL Power:** Kernels are written in the [CuTe-DSL](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html), enabling fine-grained control and optimization.

## Installation

Get started quickly with a simple pip install:

```bash
pip install quack-kernels
```

## Requirements

Ensure your environment meets these specifications:

*   NVIDIA H100 or B200 GPU
*   CUDA Toolkit 12.9+
*   Python 3.12

## Usage

Import and utilize QuACK kernels in your Python code:

```python
from quack import rmsnorm, softmax, cross_entropy
```

## Performance and Optimization

Achieve blazing-fast performance for your AI tasks.  For in-depth details, including benchmark results and optimization strategies, refer to our comprehensive [blog post](media/2025-07-10-membound-sol.md) explaining how we optimized memory-bound kernels.

### Performance Benchmarks

```
<div align="center">
<figure>
  <img
  src="media/bf16_kernel_benchmarks_single_row.svg"
  >
</figure>
</div>
```

## Development

Set up your development environment with these steps:

```bash
pip install -e '.[dev]'
pre-commit install
```