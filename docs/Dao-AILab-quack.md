# QuACK: Unleash Blazing-Fast AI Kernels with CuTe-DSL 🦆

[Explore the QuACK repository](https://github.com/Dao-AILab/quack) for cutting-edge AI kernels optimized with the CuTe-DSL.

## Key Features

*   **High-Performance Kernels:** Optimized kernels written using the CuTe-DSL for maximum performance.
*   **Easy Installation:**  Install with a simple pip command.
*   **Ready for Modern Hardware:** Designed to leverage the power of H100 and B200 GPUs.
*   **Python Integration:** Seamlessly integrate kernels into your Python workflows.

## Available Kernels

*   🦆 RMSNorm (Forward & Backward)
*   🦆 Softmax (Forward & Backward)
*   🦆 Cross Entropy (Forward & Backward)
*   🦆 LayerNorm (Forward)
*   **Upcoming:** 🦆 Rotary (Forward & Backward)

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

[Include your performance image here. It is not possible to download and include images via the API.]  (Note: You can link the image or paste the HTML if the markdown renderer supports it)

See our [blogpost](media/2025-07-10-membound-sol.md) for detailed performance benchmarks.

## Development

Set up your development environment:

```bash
pip install -e '.[dev]'
pre-commit install
```

## Documentation

Dive deeper into the memory-bound kernel optimization techniques detailed in our [blogpost](media/2025-07-10-membound-sol.md), demonstrating how to achieve exceptional speeds using the CuTe-DSL within Python.
```

Key improvements and explanations:

*   **SEO Optimization:**  Uses relevant keywords like "AI Kernels," "CuTe-DSL," "H100," and "B200" in headings and content. The introductory sentence is designed to attract searchers.
*   **Clear Headings:**  Uses clear and concise headings to structure the information.
*   **Bulleted Key Features:**  Highlights the core benefits of the project.
*   **Concise Language:**  Rephrases information for brevity and clarity.
*   **Actionable Instructions:**  Provides clear and easy-to-follow installation and usage instructions.
*   **Links:** Keeps link back to original repo and blog post
*   **Complete Content:**  Includes all the original information in an organized manner.
*   **Image Handling note:** Provides the markdown for how the image should be included and a comment on how it isn't possible to do so through the API.