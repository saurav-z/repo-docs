# 🦆 QuACK: Blazing-Fast Kernels for Your AI Projects 🦆

**Supercharge your AI workloads with QuACK, a collection of high-performance kernels meticulously crafted using the CuTe-DSL.**  [View the original repo](https://github.com/Dao-AILab/quack).

## Key Features

*   **Optimized for NVIDIA GPUs:** Leverages the power of H100 and B200 GPUs for peak performance.
*   **CuTe-DSL Implementation:**  Kernels are written in the CuTe-DSL for efficient execution.
*   **Easy Installation:** Install with a simple `pip install`.
*   **Ready-to-Use Kernels:** Includes essential AI kernels for common tasks.

## Kernels Included

*   RMSNorm (Forward + Backward)
*   Softmax (Forward + Backward)
*   Cross Entropy (Forward + Backward)
*   Layernorm (Forward)
*   Upcoming: Rotary (Forward + Backward)

## Requirements

*   NVIDIA H100 or B200 GPU
*   CUDA Toolkit 12.9+
*   Python 3.12

## Installation

```bash
pip install quack-kernels
```

## Usage Example

```python
from quack import rmsnorm, softmax, cross_entropy
```

## Performance & Benchmarks

QuACK kernels are designed for optimal performance. See the following for detailed benchmarks:

*   **Blog Post:**  [Our blog post](media/2025-07-10-membound-sol.md) on optimizing memory-bound kernels with CuTe-DSL.

<div align="center">
<figure>
  <img
  src="media/bf16_kernel_benchmarks_single_row.svg"
  alt="Benchmark graph"
  >
</figure>
</div>

## Development

To contribute or develop, set up your environment with:

```bash
pip install -e '.[dev]'
pre-commit install
```