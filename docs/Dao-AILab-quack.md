# QuACK: High-Performance Kernels for NVIDIA GPUs 🦆

**Supercharge your AI workloads with QuACK, a library of highly optimized kernels designed for NVIDIA H100 and B200 GPUs, written in the powerful CuTe-DSL.** ([Original Repository](https://github.com/Dao-AILab/quack))

## Key Features

*   **Optimized Kernels:**  Leverage hand-tuned kernels for critical operations, delivering significant performance gains.
*   **CuTe-DSL Based:**  Built using the CuTe-DSL, enabling efficient and flexible kernel development.
*   **Easy Installation:** Simple pip installation makes getting started a breeze.
*   **Diverse Kernel Suite:** Includes kernels for essential deep learning operations, like:
    *   RMSNorm (forward + backward)
    *   Softmax (forward + backward)
    *   Cross Entropy (forward + backward)
    *   LayerNorm (forward)
    *   Hopper GEMM + Epilogue
    *   Blackwell GEMM + Epilogue

## Installation

```bash
pip install quack-kernels
```

## Requirements

*   NVIDIA H100 or B200 GPU
*   CUDA Toolkit 12.9+
*   Python 3.12

## Usage

Get started quickly in your Python code:

```python
from quack import rmsnorm, softmax, cross_entropy
```

## Performance & Benchmarks

[Include an image of bf16_kernel_benchmarks_single_row.svg here]

>   [Link to the original image on github](https://github.com/Dao-AILab/quack/blob/main/media/bf16_kernel_benchmarks_single_row.svg)

See our comprehensive [blogpost](media/2025-07-10-membound-sol.md) for detailed performance analysis and optimization strategies.

## Documentation & Resources

*   **CuTe-DSL:** Learn more about the underlying framework:  [CuTe-DSL Documentation](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html)
*   **Blog Post:** Dive into the details of our memory-bound kernel optimization:  [Blog Post](media/2025-07-10-membound-sol.md)

## Development

Set up your development environment:

```bash
pip install -e '.[dev]'
pre-commit install
```
```
**SEO Improvements and Explanation:**

*   **Title:**  "QuACK: High-Performance Kernels for NVIDIA GPUs" -  Includes the key keywords ("QuACK", "Kernels", "NVIDIA GPUs") to target relevant search queries.
*   **One-Sentence Hook:** Grabs the user's attention immediately and highlights the core value proposition.
*   **Bulleted Key Features:** Makes the information easy to scan and highlights the main benefits.
*   **Usage Example:** Provides a quick "how-to-get-started" snippet.
*   **Clear Headings:** Organizes the content logically for readability.
*   **Requirements Section:** Directly states the needed hardware and software.
*   **Links:** Includes links to the original repository, documentation, and the blog post.
*   **Keywords:** Utilizes keywords like "kernels," "NVIDIA," "CUDA," "deep learning," and the specific kernel names to improve searchability.
*   **Call to Action:** Encourages the user to start using the library.
*   **Image Placeholder:** Includes a clear placeholder to insert the provided performance image and adds a link to it.