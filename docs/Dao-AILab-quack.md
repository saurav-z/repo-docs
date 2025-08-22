# QuACK: Unleash Lightning-Fast CUDA Kernels for Your AI Projects 🚀

Speed up your AI computations with QuACK, a library of high-performance CUDA kernels written in the elegant [CuTe-DSL](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html). Explore how QuACK can boost your model training and inference with optimized kernels.

**[View the original repository on GitHub](https://github.com/Dao-AILab/quack)**

## Key Features

*   **Optimized Kernels:**  Leverage highly performant kernels designed for speed and efficiency.
*   **CuTe-DSL Implementation:** Written in CuTe-DSL for readability and performance.
*   **Easy Installation:** Get started quickly with a simple `pip` installation.
*   **Python Integration:** Seamlessly integrate QuACK kernels into your Python workflows.

## Available Kernels

QuACK currently offers the following kernels, with more on the way:

*   🦆 RMSNorm (Forward and Backward)
*   🦆 Softmax (Forward and Backward)
*   🦆 Cross Entropy (Forward and Backward)
*   🦆 Layernorm (Forward)
*   🦆 Rotary (Forward and Backward) - *Coming Soon*

## Installation

Install QuACK with a single command:

```bash
pip install quack-kernels
```

## Requirements

Ensure your environment meets the following requirements:

*   **GPU:** H100 or B200
*   **CUDA:** CUDA toolkit 12.9+
*   **Python:** Python 3.12

## Usage

Import and use QuACK kernels in your Python code:

```python
from quack import rmsnorm, softmax, cross_entropy
```

## Performance & Benchmarks

QuACK kernels are designed for maximum performance. See our [blogpost](media/2025-07-10-membound-sol.md) for detailed benchmarks and performance analysis.

[Benchmarking image to be inserted here]