<div align="center">

<img src="./docs/assets/images/logo.png" alt="Pruna AI Logo" width=400></img>

<br>

**Pruna: Make AI models faster, cheaper, smaller, and greener, effortlessly!**

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

## Optimize Your AI Models with Pruna

Pruna is a cutting-edge model optimization framework, providing developers with the tools to make their AI models more efficient.  Get started with Pruna today to enhance model performance and reduce resource consumption.  Find the original repository [here](https://github.com/PrunaAI/pruna).

**Key Features:**

*   **Faster:** Accelerate inference times through advanced optimization techniques.
*   **Smaller:** Reduce model size while maintaining quality.
*   **Cheaper:** Lower computational costs and resource requirements.
*   **Greener:** Decrease energy consumption and environmental impact.

## Core Optimization Techniques

Pruna employs a suite of powerful compression algorithms to optimize your AI models:

*   **Caching:**  Improves performance by storing intermediate results.
*   **Quantization:** Reduces model size and speeds up inference by reducing the precision of weights.
*   **Pruning:** Removes less important or redundant connections within the network.
*   **Distillation:** Trains a smaller model to mimic a larger model.
*   **Compilation:** Optimizes the model for specific hardware.

Pruna supports a wide array of model types including LLMs, Diffusion and Flow Matching Models, Vision Transformers, Speech Recognition Models and more.

## Installation

Pruna is available for Linux, MacOS, and Windows. Before installing, ensure you have Python 3.9 or higher and optionally, the CUDA toolkit for GPU support.

**Installation Options:**

*   **Using pip:**

    ```bash
    pip install pruna
    ```

*   **From Source:**

    ```bash
    git clone https://github.com/pruna-ai/pruna.git
    cd pruna
    pip install -e .
    ```

## Quick Start

Easily optimize your models with just a few lines of code!

1.  **Load your model:**  Example using Stable Diffusion:

    ```python
    from diffusers import StableDiffusionPipeline
    base_model = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
    ```

2.  **Use Pruna's `smash` function:**  Configure your optimization with `SmashConfig`:

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

4.  **Evaluate Performance:**

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

Consult the [documentation][documentation] for all supported algorithms and detailed tutorials.

## Pruna Pro

Unlock advanced optimization features with Pruna Pro, our enterprise solution.  Gain access to the `OptimizationAgent`, priority support, and more.

### Benchmarks

Here are some benchmarks showcasing the improvements possible with Pruna Pro.

#### Stable Diffusion XL

Combining Auto Caching with DeepCache and torch.compile, plus HQQ 8-bit quantization, results in a **9%** reduction in inference latency and a model size reduction from **8.8GB** to **6.7GB**.

<img src="./docs/assets/plots/benchmark_sdxl.svg" alt="SDXL Benchmark"/>

#### FLUX \[dev]

Auto Caching, combined with Stable Fast and HQQ 8-bit quantization, reduced the latency of Auto Caching by an additional **13%** and reduced the size of FLUX from **33GB** to **23GB**.

<img src="./docs/assets/plots/benchmark_flux.svg" alt="FLUX [dev] Benchmark"/>

#### HunyuanVideo

Applying HQQ 8-bit quantization to HunyuanVideo reduced the size from **41GB** to **29GB**.

<img src="./docs/assets/plots/benchmark_hunyuan.svg" alt="HunyuanVideo Benchmark"/>

## Algorithm Overview

| Technique       | Description                                                                                  | Speed | Memory | Quality |
| --------------- | -------------------------------------------------------------------------------------------- | :---: | :----: | :-----: |
| `batcher`       | Groups inputs for simultaneous processing, improving efficiency.                             |  ✅  |   ❌   |    ➖    |
| `cacher`        | Stores intermediate computations to speed up operations.                                   |  ✅  |   ➖   |    ➖    |
| `compiler`      | Optimizes the model for specific hardware.                                                  |  ✅  |   ➖   |    ➖    |
| `distiller`     | Trains a smaller model to mimic a larger model.                                             |  ✅  |   ✅   |    ❌    |
| `quantizer`     | Reduces weight and activation precision to lower memory requirements.                       |  ✅  |   ✅   |    ❌    |
| `pruner`        | Removes redundant connections and neurons.                                                 |  ✅  |   ✅   |    ❌    |
| `recoverer`     | Restores model performance after compression.                                               |  ➖  |   ➖   |    ✅    |
| `factorizer`    | Batches small matrix multiplications into one fused operation.                             |  ✅  |   ➖   |    ➖    |
| `enhancer`      | Enhances the model output with post-processing algorithms.                                 |  ❌  |   ➖   |    ✅    |
| `distributer`   | Distributes inference or calculations across multiple devices.                              |  ✅  |   ❌   |    ➖    |
| `kernel`        | Specialized GPU routines to speed up computation.                                          |  ✅  |   ➖   |    ➖    |

✅ (improves), ➖ (approx. the same), ❌ (worsens)

<br><br>

<p align="center"><img src="./docs/assets/images/single_line.png" alt="Pruna AI Logo" width=600, height=30></img></p>

<br>

## FAQ and Troubleshooting

Find answers to common questions and solutions to potential issues in our [documentation][documentation], [FAQs][docs-faq], or existing GitHub issues.  Need further assistance?  Get help on [Discord][discord], attend our [Office Hours][docs-office-hours], or open an issue on GitHub.

## Contributing

Help build Pruna!  The project was made with 💜 by the Pruna AI team and our amazing contributors.  [Contribute to the repository][docs-contributing] and become part of the Pruna family.

[![Contributors](https://contrib.rocks/image?repo=PrunaAI/pruna)](https://github.com/PrunaAI/pruna/graphs/contributors)

## Citation

Cite Pruna in your research:

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

*   **SEO-optimized Title and Hook:** The title is more descriptive and includes keywords like "AI models," "faster," "cheaper," "smaller," and "greener." The hook clearly and concisely states the value proposition.
*   **Clear Headings and Structure:**  Uses clear headings (e.g., "Optimize Your AI Models with Pruna," "Key Features," "Installation," "Quick Start," etc.) for better readability and organization.
*   **Bulleted Key Features:** Highlights the main benefits using bullet points for easy scanning.
*   **Concise Explanations:**  Provides brief, understandable descriptions of each feature and technique.
*   **Actionable "Quick Start":**  Provides a straightforward code example to encourage users to try Pruna.
*   **Expanded Benchmark Section:** More detailed and informative, showcasing the real-world benefits of Pruna Pro.  It gives clear results with concise information.
*   **Algorithm Table:**  Provides a comprehensive and easy-to-understand overview of all available optimization techniques with indicators for the results of their usage.
*   **Emphasis on Community and Support:**  Highlights ways for users to get help and contribute to the project.
*   **Clear Call to Action:** Encourages users to contribute.
*   **Consistent Formatting:** Uniform use of bold, italics, and code blocks.
*   **Removed extraneous images** The images were useful but the text was more important.
*   **Better SEO:** The content contains key words related to model optimization, the benefits of Pruna, and related AI topics.
*   **Includes links to original repo.**
*   **More concise language** removes words and phrases not needed for the primary function of the readme, to describe the project to a new user.