# QuACK: High-Performance CuTe Kernels for Your AI Workloads 🦆

**Supercharge your AI applications with QuACK, a collection of blazing-fast kernels optimized for NVIDIA GPUs using the CuTe-DSL.** (Original repository: [https://github.com/Dao-AILab/quack](https://github.com/Dao-AILab/quack))

## Key Features

*   **Optimized Kernels:** Leverage highly-optimized kernels written in the CuTe-DSL for maximum performance.
*   **Broad Compatibility:** Designed to run on H100 and B200 GPUs.
*   **Ease of Use:** Simple installation and integration into your Python projects.
*   **Comprehensive Coverage:** Includes a range of essential AI kernels.

## Available Kernels

QuACK currently offers optimized kernels for:

*   🦆 RMSNorm (forward + backward)
*   🦆 Softmax (forward + backward)
*   🦆 Cross Entropy (forward + backward)
*   🦆 Layernorm (forward)

**Upcoming:**

*   🦆 Rotary (forward + backward)

## Installation

Get started with QuACK in seconds:

```bash
pip install quack-kernels
```

## Requirements

Ensure your environment meets these requirements:

*   NVIDIA H100 or B200 GPU
*   CUDA Toolkit 12.9 or later
*   Python 3.12

## Usage

Integrate QuACK kernels seamlessly into your Python code:

```python
from quack import rmsnorm, softmax, cross_entropy
```

## Performance & Benchmarking

[Include relevant image of performance from the repo]

For detailed performance analysis and insights, please refer to our blog post: [link to blogpost from the original repo, if available]

## Documentation & Resources

*   **CuTe-DSL:** Learn more about the underlying technology: [CuTe-DSL Documentation](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html)
*   **Blog Post:** (If available, link to the blog post from original README)

## Development

To set up a development environment:

```bash
pip install -e '.[dev]'
pre-commit install
```