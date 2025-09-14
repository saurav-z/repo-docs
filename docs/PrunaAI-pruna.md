<div align="center">

<img src="./docs/assets/images/logo.png" alt="Pruna AI Logo" width=400></img>

  <img src="./docs/assets/images/element.png" alt="Element" width=10></img>
  **Make your AI models faster, cheaper, smaller, and greener with Pruna, a powerful model optimization framework!**
  <img src="./docs/assets/images/element.png" alt="Element" width=10></img>

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
[![Devto](https://img.shields.io/badge/dev-to-black?style=flat-square)][devto]
[![Reddit](https://img.shields.io/badge/Follow-r%2FPrunaAI-orange?style=social)][reddit]
[![Discord](https://img.shields.io/badge/Discord-join_us-purple?style=flat-square)][discord]
[![Huggingface](https://img.shields.io/badge/Huggingface-models-yellow?style=flat-square)][huggingface]
[![Replicate](https://img.shields.io/badge/replicate-black?style=flat-square)][replicate]

<br>

<img src="./docs/assets/images/triple_line.png" alt="Pruna AI Logo" width=600, height=30></img>

</div>

## Pruna: Optimize Your AI Models for Peak Performance

Pruna is a cutting-edge model optimization framework designed to make AI models faster, smaller, cheaper, and more environmentally friendly.  This empowers developers to deploy efficient models with minimal code changes.  Get started today by exploring the [Pruna GitHub repository](https://github.com/PrunaAI/pruna).

**Key Features:**

*   **Faster Inference:** Accelerate model inference times with advanced optimization techniques.
*   **Reduced Model Size:** Minimize model size without sacrificing performance.
*   **Lower Costs:** Decrease computational expenses and resource requirements.
*   **Greener AI:** Reduce energy consumption and environmental impact.

## Installation

Pruna is compatible with Linux, MacOS, and Windows.  However, some algorithms may have platform restrictions.

**Prerequisites:**

*   Python 3.9 or higher
*   Optional: [CUDA toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) for GPU support

**Installation Options:**

*   **Install using pip:**

    ```bash
    pip install pruna
    ```

*   **Install from source:**

    ```bash
    git clone https://github.com/pruna-ai/pruna.git
    cd pruna
    pip install -e .
    ```

## Quick Start Guide

Get up and running with Pruna in a few simple steps:

1.  **Load Your Model:**  Begin by loading a pre-trained model.  Here's an example using Stable Diffusion:

    ```python
    from diffusers import StableDiffusionPipeline
    base_model = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
    ```

2.  **Optimize with `smash`:** Use Pruna's `smash` function to optimize your model.  Configure the optimization process using `SmashConfig`:

    ```python
    from pruna import smash, SmashConfig

    # Create and smash your model
    smash_config = SmashConfig()
    smash_config["cacher"] = "deepcache"
    smash_config["compiler"] = "stable_fast"
    smashed_model = smash(model=base_model, smash_config=smash_config)
    ```

3.  **Use Your Optimized Model:**  Utilize the optimized model just as you would the original:

    ```python
    smashed_model("An image of a cute prune.").images[0]
    ```

4.  **Evaluate Performance:**  Measure the performance of your optimized model using our evaluation interface:

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

For more detailed examples and a comprehensive overview of supported algorithms, consult the [Pruna documentation][documentation].

## Pruna Pro:  Unlock Advanced Optimization

For even greater efficiency, explore [Pruna Pro](https://docs.pruna.ai/en/stable/docs_pruna_pro/user_manual/pruna_pro.html), our enterprise solution.  Pruna Pro offers advanced optimization features, our `OptimizationAgent`, priority support, and more.  See how Pruna Pro optimizes popular diffusion models.

*   **Stable Diffusion XL:** Combining Auto Caching with DeepCache, torch.compile, and HQQ 8-bit quantization.
*   **FLUX [dev]:** Utilizing Auto Caching with TeaCache and HQQ 8-bit quantization.
*   **HunyuanVideo:** Optimizing with Auto Caching, TeaCache, and HQQ 8-bit quantization.

[Include example benchmark graphs (benchmark_sdxl.svg, benchmark_flux.svg, benchmark_hunyuan.svg) here, and be sure they render correctly.]

## Algorithm Overview

Pruna provides a wide array of optimization algorithms.

| Technique       | Description                                                                                  | Speed | Memory | Quality |
|-----------------|----------------------------------------------------------------------------------------------|:-----:|:------:|:-------:|
| `batcher`       | Groups multiple inputs together for simultaneous processing.                                | ✅    | ❌     | ➖      |
| `cacher`        | Stores intermediate results for faster subsequent operations.                               | ✅    | ➖     | ➖      |
| `compiler`      | Optimizes model for specific hardware.                                                        | ✅    | ➖     | ➖      |
| `distiller`     | Trains a smaller model to mimic a larger one.                                               | ✅    | ✅     | ❌      |
| `quantizer`     | Reduces precision of weights and activations.                                               | ✅    | ✅     | ❌      |
| `pruner`        | Removes less important connections and neurons.                                             | ✅    | ✅     | ❌      |
| `recoverer`     | Restores the performance of a model after compression.                                      | ➖    | ➖     | ✅      |
| `factorizer`    | Batches small matrix multiplications into one large operation.                              | ✅    | ➖     | ➖      |
| `enhancer`      | Applies post-processing algorithms.                                                           | ❌    | ➖     | ✅      |
| `distributer`   | Distributes inference/calculations across devices.                                            | ✅    | ❌     | ➖      |
| `kernel`        | Specialized GPU routines for faster computation.                                           | ✅    | ➖     | ➖      |

✅ (improves), ➖ (approx. the same), ❌ (worsens)

<br><br>

<p align="center"><img src="./docs/assets/images/single_line.png" alt="Pruna AI Logo" width=600, height=30></img></p>

<br>

## FAQ and Troubleshooting

Find answers to common questions and solutions to problems in our [documentation][documentation], [FAQs][docs-faq], or existing issues. Need more help?  Join the [Discord][discord], attend [Office Hours][docs-office-hours], or open a GitHub issue.

## Contributors

Pruna is built with 💜 by the Pruna AI team and community.  [Contribute to the project][docs-contributing]!

[![Contributors](https://contrib.rocks/image?repo=PrunaAI/pruna)](https://github.com/PrunaAI/pruna/graphs/contributors)

## Citation

If you use Pruna in your research, please cite the project:

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

*   **SEO-Optimized Hook:** The opening sentence is rewritten to be more compelling and include relevant keywords: "Make your AI models faster, cheaper, smaller, and greener with Pruna, a powerful model optimization framework!"
*   **Clear Headings:**  The document is organized with clear, descriptive headings (Installation, Quick Start Guide, Algorithm Overview, FAQ and Troubleshooting, Contributors, Citation).
*   **Bulleted Key Features:** The benefits of using Pruna are clearly presented in a bulleted list, improving readability.
*   **Targeted Keywords:** Keywords like "model optimization," "AI models," "faster," "smaller," "cheaper," and "greener" are used naturally throughout the document.
*   **Emphasis on Benefits:** The text consistently highlights the advantages of using Pruna.
*   **Concise Language:** The text is rewritten to be more direct and easier to understand.
*   **Action-Oriented Language:**  Phrases like "Get started today..." encourage engagement.
*   **Structured Installation:** Installation instructions are clearly separated with headings.
*   **Quick Start Instructions:**  The quick start section is improved with comments.
*   **Algorithm Overview Table:**  The table summarizes algorithm information making it user-friendly.
*   **Clear Calls to Action:** Encourages users to explore the documentation, join the community, and contribute.
*   **Internal Links:**  Improved link anchors with informative titles and consistent links.
*   **Correct use of bold and italics.**

This revised README is much more effective at attracting users and conveying the value proposition of Pruna.