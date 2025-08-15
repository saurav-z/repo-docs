# 🦆 QuACK: Unleash Blazing-Fast Kernel Performance with CuTe! 🦆

Looking for optimized, high-performance kernels for your CUDA-based applications? [QuACK](https://github.com/Dao-AILab/quack) provides a collection of hand-crafted kernels written in the CuTe-DSL, designed for peak efficiency on modern NVIDIA GPUs.

## Key Features:

*   **Optimized Kernels:**  Benefit from highly optimized kernels for common operations, boosting the performance of your applications.
*   **CuTe-DSL Powered:** Built using the CuTe-DSL ([CuTe-DSL Documentation](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html)) for efficient and expressive kernel development.
*   **Easy Installation:** Simple pip installation for quick integration into your projects.
*   **Comprehensive Coverage:** Includes essential kernels with forward and backward passes.

## Kernels Included:

*   🦆 RMSNorm (Forward & Backward)
*   🦆 Softmax (Forward & Backward)
*   🦆 Cross Entropy (Forward & Backward)
*   🦆 Layernorm (Forward)
*   🦆 Rotary (Forward & Backward) - *Coming Soon*

## System Requirements

To leverage the full potential of QuACK, ensure your environment meets these requirements:

*   **GPU:** H100 or B200
*   **CUDA:** CUDA toolkit 12.9+
*   **Python:** Python 3.12

## Installation

Install QuACK using pip:

```bash
pip install quack-kernels
```

## Usage Example

Quickly integrate QuACK into your Python code:

```python
from quack import rmsnorm, softmax, cross_entropy
```

## Performance

[Detailed performance analysis can be found in the blogpost](media/2025-07-10-membound-sol.md) for further details on the performance advantages of QuACK.

<div align="center">
<figure>
  <img
  src="media/bf16_kernel_benchmarks_single_row.svg"
  >
</figure>
</div>

## Development

Set up your development environment:

```bash
pip install -e '.[dev]'
pre-commit install
```

**[Back to the QuACK Repository](https://github.com/Dao-AILab/quack)**