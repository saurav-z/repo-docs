<div align="center">

<img src="./docs/assets/images/logo.png" alt="Pruna AI Logo" width=400></img>

<br>

  **Unlock the power of AI with Pruna: Make your AI models faster, cheaper, smaller, and greener!**

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

Pruna is an open-source model optimization framework that empowers developers to build and deploy more efficient AI models. Achieve significant improvements in speed, size, cost, and energy consumption with minimal code changes.  [Check out the original repository](https://github.com/PrunaAI/pruna)!

**Key Features:**

*   **Faster Inference:** Accelerate model execution with advanced optimization techniques.
*   **Smaller Model Sizes:** Reduce model footprints without sacrificing quality.
*   **Reduced Costs:** Lower computational expenses and resource needs.
*   **Greener AI:** Minimize energy usage and environmental impact.
*   **Simplified Optimization:**  Easily optimize models with just a few lines of code.
*   **Broad Model Support:** Compatible with LLMs, Diffusion Models, Vision Transformers, Speech Recognition models, and more.

## Installation

Pruna is compatible with Linux, MacOS, and Windows.

**Prerequisites:**

*   Python 3.9 or higher
*   Optional: [CUDA Toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) for GPU acceleration

**Installation Options:**

1.  **Using pip:**

    ```bash
    pip install pruna
    ```

2.  **From Source:**

    ```bash
    git clone https://github.com/pruna-ai/pruna.git
    cd pruna
    pip install -e .
    ```

## Quick Start

Optimize your models in a few simple steps!

1.  **Load your pre-trained model:**
    ```python
    from diffusers import StableDiffusionPipeline
    base_model = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
    ```

2.  **Use Pruna's `smash` function:**
    ```python
    from pruna import smash, SmashConfig

    # Create and smash your model
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

    Explore the [documentation][documentation] for a comprehensive overview of supported algorithms and more advanced examples.

## Pruna Pro: Enterprise-Grade Optimization

Unlock advanced features and support with Pruna Pro, our enterprise solution.

*   **Benefits:** Explore Auto Caching for popular Diffusers pipelines.
*   **Stable Diffusion XL:** Combines Auto Caching with DeepCache and torch.compile to reduce latency and model size (from 8.8GB to 6.7GB using HQQ quantization).
    <img src="./docs/assets/plots/benchmark_sdxl.svg" alt="SDXL Benchmark"/>

*   **FLUX [dev]:** Compares Auto Caching with TeaCache. Utilizes Stable Fast and HQQ quantization, reducing the size of FLUX from 33GB to 23GB.
    <img src="./docs/assets/plots/benchmark_flux.svg" alt="FLUX [dev] Benchmark"/>

*   **HunyuanVideo:**  Compares Auto Caching with TeaCache.  Applying HQQ 8-bit quantization reduced the size from 41GB to 29GB.
    <img src="./docs/assets/plots/benchmark_hunyuan.svg" alt="HunyuanVideo Benchmark"/>

## Algorithm Overview

Pruna provides a wide array of optimization algorithms.  Refer to the [documentation][documentation] for in-depth details.

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

Find answers to common questions and troubleshooting tips in our [documentation][documentation], [FAQs][docs-faq], or existing issues.  Get community help on [Discord][discord], attend our [Office Hours][docs-office-hours], or open a GitHub issue.

##  Contribute

Join the Pruna AI community!  Contribute to the project and become part of the family.

[![Contributors](https://contrib.rocks/image?repo=PrunaAI/pruna)](https://github.com/PrunaAI/pruna/graphs/contributors)

## Citation

If you use Pruna in your research, please cite the project!

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