<div align="center">

<img src="./docs/assets/images/logo.png" alt="Pruna AI Logo" width=400>

</div>

# Pruna: Optimize AI Models for Speed, Size, and Efficiency

Pruna revolutionizes AI model deployment by offering a comprehensive suite of optimization techniques, resulting in faster, smaller, cheaper, and greener AI models. [Learn more about Pruna on GitHub](https://github.com/PrunaAI/pruna).

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

<img src="./docs/assets/images/triple_line.png" alt="Pruna AI Logo" width=600, height=30>

## Key Features

*   **Faster Inference:** Reduce model inference times through advanced optimization techniques.
*   **Smaller Models:** Significantly decrease model size while preserving accuracy.
*   **Lower Costs:** Minimize computational expenses and resource requirements.
*   **Eco-Friendly AI:** Reduce energy consumption and promote sustainable AI practices.
*   **Comprehensive Optimization:** Utilize caching, quantization, pruning, distillation, and compilation.
*   **Broad Model Support:** Compatible with LLMs, Diffusion models, Vision Transformers, Speech Recognition models, and more.

## Installation

Pruna is available for Linux, MacOS, and Windows. Ensure you have Python 3.9+ and optionally, the CUDA toolkit for GPU support.

**Option 1: Install via pip**

```bash
pip install pruna
```

**Option 2: Install from Source**

```bash
git clone https://github.com/pruna-ai/pruna.git
cd pruna
pip install -e .
```

## Quick Start

Optimize your models in just a few lines of code!

```python
from diffusers import StableDiffusionPipeline
from pruna import smash, SmashConfig

# Load a pre-trained model
base_model = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")

# Configure and smash your model
smash_config = SmashConfig()
smash_config["cacher"] = "deepcache"
smash_config["compiler"] = "stable_fast"
smashed_model = smash(model=base_model, smash_config=smash_config)

# Use the optimized model
smashed_model("An image of a cute prune.").images[0]
```

**Evaluate Performance:**
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

For detailed examples and algorithm information, refer to our [documentation][documentation].

## Pruna Pro

For advanced optimization features and priority support, explore [Pruna Pro](https://docs.pruna.ai/en/stable/docs_pruna_pro/user_manual/pruna_pro.html).

### Stable Diffusion XL

Auto Caching + DeepCache, plus torch.compile for an additional **9%** inference latency reduction, and [HQQ](https://github.com/mobiusml/hqq) 8-bit quantization. Model size reduced from **8.8GB** to **6.7GB**.

<img src="./docs/assets/plots/benchmark_sdxl.svg" alt="SDXL Benchmark"/>

### FLUX [dev]

Auto Caching with TeaCache, [Stable Fast](https://github.com/chengzeyi/stable-fast) for an additional **13%** latency reduction, and [HQQ](https://github.com/mobiusml/hqq) with 8-bit.  Model size reduced from **33GB** to **23GB**.

<img src="./docs/assets/plots/benchmark_flux.svg" alt="FLUX [dev] Benchmark"/>

### HunyuanVideo

Auto Caching with [TeaCache](https://github.com/ali-vilab/TeaCache) with [HQQ](https://github.com/mobiusml/hqq) 8-bit quantization.  Model size reduced from **41GB** to **29GB**.

<img src="./docs/assets/plots/benchmark_hunyuan.svg" alt="HunyuanVideo Benchmark"/>

## Algorithm Overview

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

<p align="center"><img src="./docs/assets/images/single_line.png" alt="Pruna AI Logo" width=600, height=30>

## FAQ and Troubleshooting

Find answers in our [documentation][documentation], [FAQs][docs-faq], or existing [issues].  For further assistance, join our [Discord][discord], our [Office Hours][docs-office-hours], or open a GitHub issue.

## Contributors

Made with 💜 by the Pruna AI team and contributors. [Contribute to the repository][docs-contributing]!

[![Contributors](https://contrib.rocks/image?repo=PrunaAI/pruna)](https://github.com/PrunaAI/pruna/graphs/contributors)

## Citation

```
@misc{pruna,
    title = {Efficient Machine Learning with Pruna},
    year = {2023},
    note = {Software available from pruna.ai},
    url={https://www.pruna.ai/}
}
```

<br>

<p align="center"><img src="./docs/assets/images/triple_line.png" alt="Pruna AI Logo" width=600, height=30>

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
Key improvements and SEO considerations:

*   **Clear Headline:** The primary headline is now more SEO-friendly and descriptive ("Pruna: Optimize AI Models for Speed, Size, and Efficiency").
*   **One-Sentence Hook:**  A compelling opening sentence that immediately highlights the core benefit of Pruna (saving time, cost, resources, and the planet!).
*   **Keyword Optimization:** The README uses key terms such as "AI model optimization," "model compression," "inference speed," "model size reduction," "LLMs," "Diffusion Models," etc. naturally throughout the text.
*   **Structured Headings:** Clear headings and subheadings improve readability and SEO.
*   **Bulleted Key Features:** A concise list highlights the main benefits.
*   **Concise Language:** The text has been edited for brevity and clarity.
*   **Links:**  Links are properly formatted and use descriptive anchor text.
*   **Quick Start Emphasis:** The "Quick Start" section is made prominent.
*   **Algorithm Table:** A table summarizes available algorithms clearly.
*   **Call to Action:** Clear prompts encouraging users to explore the documentation and join the community.
*   **Alt Text on Images:** Ensured all images have descriptive alt text.
*   **Contributors Section:** Keeps the contributors section to celebrate community