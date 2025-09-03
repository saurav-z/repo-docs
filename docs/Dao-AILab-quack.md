# 🦆 QuACK: Unleash the Power of Lightning-Fast Kernels with CuTe-DSL 🦆

**Supercharge your AI workloads with QuACK, a collection of high-performance kernels meticulously crafted using the CuTe-DSL for NVIDIA GPUs.**  Find the original repository [here](https://github.com/Dao-AILab/quack).

## Key Features

*   **Optimized Kernels:**  Leverage highly optimized kernels for common AI operations.
*   **CuTe-DSL Powered:** Built using the CuTe-DSL, enabling exceptional performance on NVIDIA GPUs.
*   **Easy Installation:** Simple `pip install` for seamless integration into your projects.
*   **Memory-Bound Kernel Optimization:**  Achieve remarkable speedups for memory-bound operations, as detailed in our blog post.

## Supported Kernels

QuACK currently offers optimized kernels for the following:

*   🦆 RMSNorm (forward + backward)
*   🦆 Softmax (forward + backward)
*   🦆 Cross Entropy (forward + backward)
*   🦆 Layernorm (forward)

**Upcoming:**

*   🦆 Rotary (forward + backward)

## Requirements

*   NVIDIA H100 or B200 GPU
*   CUDA Toolkit 12.9+
*   Python 3.12

## Installation

```bash
pip install quack-kernels
```

## Usage

```python
from quack import rmsnorm, softmax, cross_entropy
```

## Performance

[Include your performance graph here, preferably with alt text.]

**[Optional: Add a link to the image if you're not displaying it directly.]**

For detailed performance analysis, please refer to our comprehensive [blog post](media/2025-07-10-membound-sol.md).

## Development

To set up a development environment:

```bash
pip install -e '.[dev]'
pre-commit install
```

## Documentation

Explore the [CuTe-DSL documentation](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html) for more information on the underlying technology.
```

Key improvements and explanations:

*   **SEO Optimization:** Includes relevant keywords like "kernels," "CuTe-DSL," "NVIDIA GPUs," "AI workloads," and "performance" in the title and body.
*   **One-Sentence Hook:**  Grabs attention immediately.
*   **Clear Headings:**  Uses Markdown headings for better organization and readability.
*   **Bulleted Key Features:** Highlights the core benefits of using QuACK.
*   **Concise Language:** Streamlines the text for clarity.
*   **Call to Action (Implied):**  Encourages users to explore the kernels and blog post.
*   **Explicit GPU Requirements:**  Clarifies GPU support.
*   **Clear Usage Example:**  Provides a simple code snippet.
*   **Image Placeholder:** Adds a note about where to include the performance image and suggests alt text for SEO.
*   **Development Section:** Includes the development setup instructions.
*   **Link Back to Original Repo:** Adds a link to the original repository.
*   **Documentation Links:** Includes links to both the CuTe-DSL and the blogpost.