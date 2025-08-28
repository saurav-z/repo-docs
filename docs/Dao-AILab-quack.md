# 🦆 QuACK: Unleash Lightning-Fast Kernels with CuTe-DSL! 🦆

**Accelerate your AI workloads with QuACK, a collection of optimized kernels crafted using the powerful CuTe-DSL, delivering exceptional performance on NVIDIA GPUs.**  ([View the original repo](https://github.com/Dao-AILab/quack))

## Key Features

*   **Optimized Kernels:** Provides high-performance implementations of essential AI kernels.
*   **CuTe-DSL Based:** Kernels are written in the CuTe-DSL for maximum efficiency and flexibility.
*   **Easy Installation:**  Simple pip installation for quick setup.
*   **Python Integration:** Seamlessly integrate QuACK kernels into your Python projects.

## Kernels Included

QuACK currently offers the following optimized kernels:

*   🦆 RMSNorm forward + backward
*   🦆 Softmax forward + backward
*   🦆 Cross entropy forward + backward
*   🦆 Layernorm forward
*   🦆 Rotary forward + backward (Upcoming)

## Requirements

Ensure your system meets the following requirements for optimal performance:

*   **GPU:** NVIDIA H100 or B200
*   **CUDA:** CUDA Toolkit 12.9+
*   **Python:** Python 3.12

## Installation

Install QuACK kernels with a single pip command:

```bash
pip install quack-kernels
```

## Usage Example

Easily integrate QuACK kernels into your Python code:

```python
from quack import rmsnorm, softmax, cross_entropy
```

## Performance & Benchmarks

[Insert Image of benchmark](media/bf16_kernel_benchmarks_single_row.svg)

For detailed performance analysis and optimization strategies, explore our comprehensive blog post:  [Blog Post Link (replace with the actual link)]

## Development

To set up the development environment, use the following commands:

```bash
pip install -e '.[dev]'
pre-commit install
```