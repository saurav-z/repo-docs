<div align="center">
  <img src="./docs/assets/images/logo.png" alt="Pruna AI Logo" width=400>
  <img src="./docs/assets/images/element.png" alt="Element" width=10>
  **Supercharge your AI models with Pruna: making them faster, cheaper, smaller, and greener!**
  <img src="./docs/assets/images/element.png" alt="Element" width=10>
<br>

[![Documentation](https://img.shields.io/badge/Pruna_documentation-purple?style=for-the-badge)][documentation]

<br>

![GitHub License](https://img.shields.io/github/license/prunaai/pruna?style=flat-square)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/prunaai/pruna/build.yaml?style=flat-square)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/prunaai/pruna/tests.yaml?label=tests&style=flat-square)
![GitHub Release](https://img.shields.io/github/v/release/prunaai/pruna?style=flat-square)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/PrunaAI/pruna?style=flat-square)
![PyPI - Downloads](https://img.shields.io/pypi/dm/pruna?style=flat-square)
![Codacy](https://app.codacy.com/project/badge/Grade/092392ec4be846928a7c5978b6afe060)

[![Website](https://img.shields.io/badge/Pruna.ai-purple?style=flat-square)][website]
[![X (formerly Twitter) URL](https://img.shields.io/twitter/url?url=https%3A%2F%2Fx.com%2FPrunaAI)][x]
[![Devto](https://img.shields/badge/dev-to-black?style=flat-square)][devto]
[![Reddit](https://img.shields.io/badge/Follow-r%2FPrunaAI-orange?style=social)][reddit]
[![Discord](https://img.shields.io/badge/Discord-join_us-purple?style=flat-square)][discord]
[![Huggingface](https://img.shields.io/badge/Huggingface-models-yellow?style=flat-square)][huggingface]
[![Replicate](https://img.shields.io/badge/replicate-black?style=flat-square)][replicate]

<br>

<img src="./docs/assets/images/triple_line.png" alt="Pruna AI Logo" width=600, height=30>
</div>

## Table of Contents

*   [Introduction](#introduction)
*   [Key Features](#key-features)
*   [Installation](#installation)
*   [Quick Start](#quick-start)
*   [Pruna Pro](#pruna-pro)
*   [Algorithm Overview](#algorithm-overview)
*   [FAQ and Troubleshooting](#faq-and-troubleshooting)
*   [Contributors](#contributors)
*   [Citation](#citation)

## Introduction

Pruna is a powerful model optimization framework designed to help developers improve the efficiency of their AI models. Achieve significant gains in speed, size, and cost-effectiveness with minimal code changes. Pruna supports a wide range of model types and offers a comprehensive suite of compression techniques.

## Key Features

*   **Faster Inference:** Accelerate model execution with advanced optimization techniques.
*   **Smaller Model Size:** Reduce model footprint without sacrificing performance.
*   **Lower Computational Costs:** Minimize resource requirements and expenses.
*   **Greener AI:** Decrease energy consumption and environmental impact.
*   **Comprehensive Suite:** Includes caching, quantization, pruning, distillation, and compilation.
*   **User-Friendly:** Designed for simplicity, requiring just a few lines of code to optimize models.
*   **Broad Compatibility:** Supports LLMs, Diffusion Models, Vision Transformers, Speech Recognition models, and more.

## Installation

Pruna is available for Linux, MacOS, and Windows. Ensure you have Python 3.9+ and, optionally, the CUDA toolkit for GPU support.

**Installation options:**

*   **Using pip:**

    ```bash
    pip install pruna
    ```

*   **From source:**

    ```bash
    git clone https://github.com/pruna-ai/pruna.git
    cd pruna
    pip install -e .
    ```

## Quick Start

Optimize your models in a few simple steps!

1.  **Load a pre-trained model:**

    ```python
    from diffusers import StableDiffusionPipeline
    base_model = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
    ```

2.  **Use Pruna's `smash` function:**

    ```python
    from pruna import smash, SmashConfig

    smash_config = SmashConfig()
    smash_config["cacher"] = "deepcache"
    smash_config["compiler"] = "stable_fast"
    smashed_model = smash(model=base_model, smash_config=smash_config)
    ```

3.  **Use the optimized model:**

    ```python
    smashed_model("An image of a cute prune.").images[0]
    ```

4.  **Evaluate Model Performance:**

    ```python
    from pruna.evaluation.task import Task
    from pruna.evaluation.evaluation_agent import EvaluationAgent
    from pruna.data.pruna_datamodule import PrunaDataModule

    datamodule = PrunaDataModule.from_string("LAION256")
    datamodule.limit_datasets(10)
    task = Task("image_generation_quality", datamodule=datamodule)
    eval_agent = EvaluationAgent(task)
    eval_agent.evaluate(smashed_model)
    ```

See our [documentation][documentation] for a complete overview of algorithms, use cases, and tutorials.

## Pruna Pro

For advanced features and optimization, consider Pruna Pro, our enterprise solution. Experience greater efficiency gains with features like Auto Caching and proprietary algorithms.  See performance improvements for Stable Diffusion XL, FLUX [dev], and HunyuanVideo.

## Algorithm Overview

Pruna offers a diverse set of optimization algorithms. Detailed descriptions are in the [documentation](https://docs.pruna.ai/en/stable/).

| Technique    | Description                                                                                   | Speed | Memory | Quality |
|--------------|-----------------------------------------------------------------------------------------------|:-----:|:------:|:-------:|
| `batcher`    | Groups multiple inputs together to be processed simultaneously, improving computational efficiency and reducing processing time. | ✅    | ❌     | ➖      |
| `cacher`     | Stores intermediate results of computations to speed up subsequent operations.               | ✅    | ➖     | ➖      |
| `compiler`   | Optimises the model with instructions for specific hardware.                                 | ✅    | ➖     | ➖      |
| `distiller`  | Trains a smaller, simpler model to mimic a larger, more complex model.                       | ✅    | ✅     | ❌      |
| `quantizer`  | Reduces the precision of weights and activations, lowering memory requirements.              | ✅    | ✅     | ❌      |
| `pruner`     | Removes less important or redundant connections and neurons, resulting in a sparser, more efficient network. | ✅    | ✅     | ❌      |
| `recoverer`  | Restores the performance of a model after compression.                                       | ➖    | ➖     | ✅      |
| `factorizer` | Factorization batches several small matrix multiplications into one large fused operation. | ✅ | ➖ | ➖ |
| `enhancer`   | Enhances the model output by applying post-processing algorithms such as denoising or upscaling. | ❌ | ➖ | ✅ |
| `distributer`   | Distributes the inference, the model or certain calculations across multiple devices. | ✅ | ❌ | ➖ |
| `kernel`   | Kernels are specialized GPU routines that speed up parts of the computation.  | ✅ | ➖ | ➖ |

✅ (improves), ➖ (approx. the same), ❌ (worsens)

<br><br>

<p align="center"><img src="./docs/assets/images/single_line.png" alt="Pruna AI Logo" width=600, height=30></img></p>

<br>

## FAQ and Troubleshooting

Find answers to common questions and solutions to problems in our [documentation][documentation], [FAQs][docs-faq], or existing issues.  Get help from the community on [Discord][discord], join our [Office Hours][docs-office-hours], or open an issue on GitHub.

## Contributors

Pruna is built with 💜 by the Pruna AI team and our amazing contributors.  Join the Pruna family by [contributing to the repository][docs-contributing]!

[![Contributors](https://contrib.rocks/image?repo=PrunaAI/pruna)](https://github.com/PrunaAI/pruna/graphs/contributors)

## Citation

If you use Pruna in your research, please cite us!

```
@misc{pruna,
    title = {Efficient Machine Learning with Pruna},
    year = {2023},
    note = {Software available from pruna.ai},
    url={https://www.pruna.ai/}
}
```

<br>

<p align="center"><img src="./docs/assets/images/triple_line.png" alt="Pruna AI Logo" width=600, height=30></img></p>

[discord]: https://discord.gg/JFQmtFKCjd
[reddit]: https://www.reddit.com/r/PrunaAI/
[x]: https://x.com/PrunaAI
[devto]: https://dev.to/pruna-ai
[website]: https://pruna.ai
[huggingface]: https://huggingface.co/PrunaAI
[replicate]: https://replicate.com/prunaai
[documentation]: https://docs.pruna.ai/en/stable
[docs-algorithms]: https://docs.pruna.ai/en/stable/compression.html
[docs-faq]: https://docs.pruna.ai/en/stable/resources/faq.html
[docs-office-hours]: https://docs.pruna.ai/en/stable/resources/office_hours.html
[docs-contributing]: https://docs.pruna.ai/en/stable/docs_pruna/contributions/how_to_contribute.html
```
Key improvements and explanations:

*   **SEO-Optimized Hook:** Added a concise, benefit-driven sentence to immediately grab attention and highlight the value proposition.
*   **Clear Headings:**  Used proper Markdown headings for better organization and readability.
*   **Table of Contents:** Added a table of contents for easy navigation.
*   **Bulleted Key Features:**  Presented key features in a clear, scannable bulleted list.
*   **Concise and Focused:** Streamlined the text to be more direct and impactful.
*   **Emphasis on Benefits:**  Repeatedly highlighted the advantages of using Pruna (speed, size, cost, green AI).
*   **Improved Formatting:** Used bolding and better spacing for readability.
*   **Call to Action:** Included a direct call to action to install and try the library.
*   **Contextual Links:** Made sure links were placed strategically for easier access to information.
*   **Removed Redundancy:** Eliminated repetitive phrasing.
*   **Keyword Optimization:**  Used relevant keywords throughout the README, such as "AI models," "optimization," "speed," "size," "cost," and "green AI."
*   **Clarity:** Rewrote certain sections for improved clarity and conciseness.
*   **Revised Quickstart:** Improved the Quickstart section with the complete code example for better understanding and demonstration.