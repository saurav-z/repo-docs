<div align="center">

<img src="./docs/assets/images/logo.png" alt="Pruna AI Logo" width=400>

</div>

# Pruna: Optimize Your AI Models for Speed, Efficiency, and Sustainability

**Pruna is a model optimization framework designed to make AI models faster, cheaper, smaller, and greener.** 

[![Documentation](https://img.shields.io/badge/Pruna_documentation-purple?style=for-the-badge)][documentation]

[![License](https://img.shields.io/github/license/prunaai/pruna?style=flat-square)]
[![Build Status](https://img.shields.io/github/actions/workflow/status/prunaai/pruna/build.yaml?style=flat-square)]
[![Tests Status](https://img.shields.io/github/actions/workflow/status/prunaai/pruna/tests.yaml?label=tests&style=flat-square)]
[![Release](https://img.shields.io/github/v/release/prunaai/pruna?style=flat-square)]
[![Commit Activity](https://img.shields.io/github/commit-activity/m/PrunaAI/pruna?style=flat-square)]
[![PyPI Downloads](https://img.shields.io/pypi/dm/pruna?style=flat-square)]
[![Codacy](https://app.codacy.com/project/badge/Grade/092392ec4be846928a7c5978b6afe060)]

[![Website](https://img.shields.io/badge/Pruna.ai-purple?style=flat-square)][website]
[![X (formerly Twitter) URL](https://img.shields.io/twitter/url?url=https%3A%2F%2Fx.com%2FPrunaAI)][x]
[![Devto](https://img.shields.io/badge/dev-to-black?style=flat-square)][devto]
[![Reddit](https://img.shields.io/badge/Follow-r%2FPrunaAI-orange?style=social)][reddit]
[![Discord](https://img.shields.io/badge/Discord-join_us-purple?style=flat-square)][discord]
[![Huggingface](https://img.shields.io/badge/Huggingface-models-yellow?style=flat-square)][huggingface]
[![Replicate](https://img.shields.io/badge/replicate-black?style=flat-square)][replicate]

<br>

## Key Features

*   **Speed up inference:** Accelerate your models with advanced optimization techniques.
*   **Reduce model size:** Compress models without sacrificing quality.
*   **Lower costs:** Decrease computational expenses and resource needs.
*   **Enhance sustainability:** Minimize energy consumption and environmental impact.
*   **Easy to Use:** Optimize models with minimal code changes.
*   **Broad Compatibility:** Supports various model types like LLMs, Vision Transformers, and more.

## Getting Started

Pruna offers a straightforward way to optimize your models.

### Installation

Pruna is available for Linux, MacOS, and Windows.

**Prerequisites:**
*   Python 3.9+
*   Optional: CUDA toolkit (for GPU support)

**Installation Options:**

1.  **Install via pip:**
    ```bash
    pip install pruna
    ```

2.  **Install from source:**
    ```bash
    git clone https://github.com/pruna-ai/pruna.git
    cd pruna
    pip install -e .
    ```

### Quickstart Guide

1.  **Load a pre-trained model:**
    ```python
    from diffusers import StableDiffusionPipeline
    base_model = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
    ```

2.  **Optimize with `smash`:**
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

4.  **Evaluate the performance:**
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

[See the full documentation for details on supported algorithms and examples.](https://docs.pruna.ai/en/stable)

## Pruna Pro

Unlock advanced features and enterprise-level support with **Pruna Pro**, our premium solution.

### Benchmark results with Pruna Pro

The following tables summarize the performance gains we achieve with Pruna Pro for the most popular diffusion models.

### Stable Diffusion XL

| Optimization Type | Latency Reduction | Model Size Reduction |
|---|---|---|
| Auto Caching + [DeepCache](https://github.com/horseee/DeepCache) +  torch.compile | 9% | 8.8GB -> 6.7GB |

<img src="./docs/assets/plots/benchmark_sdxl.svg" alt="SDXL Benchmark"/>

### FLUX [dev]

| Optimization Type | Latency Reduction | Model Size Reduction |
|---|---|---|
| Auto Caching + [TeaCache](https://github.com/ali-vilab/TeaCache) + [Stable Fast](https://github.com/chengzeyi/stable-fast) | 13% | 33GB -> 23GB |

<img src="./docs/assets/plots/benchmark_flux.svg" alt="FLUX [dev] Benchmark"/>

### HunyuanVideo

| Optimization Type | Model Size Reduction |
|---|---|---|
| Auto Caching + [TeaCache](https://github.com/ali-vilab/TeaCache) | 41GB -> 29GB |

<img src="./docs/assets/plots/benchmark_hunyuan.svg" alt="HunyuanVideo Benchmark"/>

## Algorithm Overview

Pruna offers a diverse set of optimization algorithms.

| Technique     | Description                                                                                   | Speed | Memory | Quality |
|--------------|-----------------------------------------------------------------------------------------------|:-----:|:------:|:-------:|
| `batcher`     | Groups multiple inputs together for simultaneous processing.                               | ✅    | ❌     | ➖      |
| `cacher`      | Stores intermediate computation results for faster execution.               | ✅    | ➖     | ➖      |
| `compiler`    | Optimizes the model for specific hardware.                                 | ✅    | ➖     | ➖      |
| `distiller`   | Trains a smaller model to mimic a larger one.                       | ✅    | ✅     | ❌      |
| `quantizer`   | Reduces precision for lower memory usage.              | ✅    | ✅     | ❌      |
| `pruner`      | Removes unimportant connections. | ✅    | ✅     | ❌      |
| `recoverer`   | Restores model performance after compression.                                       | ➖    | ➖     | ✅      |
| `factorizer`  | Fuses matrix multiplications into one large operation. | ✅ | ➖ | ➖ |
| `enhancer`    | Enhances output through post-processing. | ❌ | ➖ | ✅ |
| `distributer` | Distributes the inference/model/calculations across multiple devices. | ✅ | ❌ | ➖ |
| `kernel`      | Specialized GPU routines for faster computation.  | ✅ | ➖ | ➖ |

✅ (improves), ➖ (approx. the same), ❌ (worsens)

<br>

## FAQ and Troubleshooting

Find answers to common questions in our [documentation][documentation] and [FAQs][docs-faq].  For further assistance, ask the Pruna community on [Discord][discord], attend our [Office Hours][docs-office-hours], or open a [GitHub issue](https://github.com/PrunaAI/pruna/issues).

## Contribute

Help make Pruna even better!  Join our community by [contributing to the repository][docs-contributing].

[![Contributors](https://contrib.rocks/image?repo=PrunaAI/pruna)](https://github.com/PrunaAI/pruna/graphs/contributors)

## Citation

If you use Pruna in your research, please cite us:

```
@misc{pruna,
    title = {Efficient Machine Learning with Pruna},
    year = {2023},
    note = {Software available from pruna.ai},
    url={https://www.pruna.ai/}
}
```

<br>

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

[**Original Repo**](https://github.com/PrunaAI/pruna)