# QuACK: Accelerate Your AI Workflows with CuTe Kernels 🦆

**Supercharge your AI models with QuACK, a collection of high-performance kernels written in the CuTe-DSL, optimized for NVIDIA GPUs.**  (Check out the [original repository](https://github.com/Dao-AILab/quack)!)

## Key Features

*   **Optimized Kernels:**  Leverage pre-built kernels for common AI operations.
*   **CuTe-DSL:**  Built using the CuTe-DSL, ensuring performance and flexibility.
*   **Easy Installation:**  Install and integrate QuACK kernels with a simple `pip` command.
*   **Ready for Production:**  Designed for speed and efficiency on supported hardware.

## Available Kernels

*   🦆 RMSNorm (forward and backward)
*   🦆 Softmax (forward and backward)
*   🦆 Cross Entropy (forward and backward)
*   🦆 Layernorm (forward)

**Upcoming:**

*   🦆 Rotary (forward and backward)

## Installation

```bash
pip install quack-kernels
```

## Requirements

*   NVIDIA H100 or B200 GPU
*   CUDA Toolkit 12.9+
*   Python 3.12

## Usage

```python
from quack import rmsnorm, softmax, cross_entropy
```

## Performance and Optimization

QuACK kernels are designed for optimal performance on supported hardware.  For detailed benchmarks and memory-bound optimization strategies, see our comprehensive blog post:

[Blogpost](media/2025-07-10-membound-sol.md)

<div align="center">
<figure>
  <img
  src="media/bf16_kernel_benchmarks_single_row.svg"
  alt="Performance Benchmarks">
</figure>
</div>

## Development

To set up the development environment:

```bash
pip install -e '.[dev]'
pre-commit install
```
```

Key improvements and SEO considerations:

*   **Concise and Engaging Hook:** Immediately grabs the reader's attention.
*   **Targeted Keywords:**  Includes relevant terms like "AI," "kernels," "CuTe-DSL," and "NVIDIA GPUs."
*   **Clear Headings:**  Organizes the information logically.
*   **Bulleted Key Features:**  Highlights the benefits of using QuACK.
*   **Stronger Call to Action:** Implies the benefits in the opening sentence.
*   **Alt text for image:**  Provides context for accessibility and SEO.
*   **Hyperlinked relevant URLs:**  Links to the blogpost and original repo.
*   **Clear Requirements Section:**  Specifies the hardware and software prerequisites.