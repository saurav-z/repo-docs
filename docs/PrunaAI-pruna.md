<div align="center">

<img src="./docs/assets/images/logo.png" alt="Pruna AI Logo" width=400></img>

  <img src="./docs/assets/images/element.png" alt="Element" width=10></img>
  **Simply make AI models faster, cheaper, smaller, greener!**
  <img src="./docs/assets/images/element.png" alt="Element" width=10></img>

<br>

</div>

## **Pruna: Optimize Your AI Models for Speed, Efficiency, and Sustainability**

Pruna is an open-source framework designed to help developers make their AI models faster, smaller, cheaper, and more environmentally friendly.  [Explore the Pruna AI repository](https://github.com/PrunaAI/pruna).

### **Key Features**

*   **Model Optimization:**  Leverage a suite of compression techniques including caching, quantization, pruning, distillation, and compilation.
*   **Accelerated Inference:** Significantly reduce inference times through advanced optimization.
*   **Reduced Model Size:** Shrink model footprints while maintaining accuracy.
*   **Cost Savings:** Lower computational expenses and resource needs.
*   **Environmental Impact:** Decrease energy consumption for a greener AI.
*   **Easy-to-Use:**  Optimize models with just a few lines of code.
*   **Broad Compatibility:** Supports various model types like LLMs, Diffusion Models, Vision Transformers, and Speech Recognition Models.

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

## Introduction

Pruna empowers developers to optimize their AI models, offering a versatile and streamlined approach to model compression and acceleration.  It provides a complete toolkit with compression algorithms to make your models:
* **Faster**: Accelerate inference times through advanced optimization techniques
* **Smaller**: Reduce model size while maintaining quality
* **Cheaper**: Lower computational costs and resource requirements
* **Greener**: Decrease energy consumption and environmental impact

The toolkit is designed with simplicity in mind - requiring just a few lines of code to optimize your models. It supports various model types including LLMs, Diffusion and Flow Matching Models, Vision Transformers, Speech Recognition Models and more.

<img align="left" width="40" src="docs/assets/images/highlight.png" alt="Pruna Pro"/>

**For advanced optimization, explore [Pruna Pro](https://docs.pruna.ai/en/stable/docs_pruna_pro/user_manual/pruna_pro.html), our enterprise solution with advanced features and priority support.**
<br clear="left"/>

## Installation

Pruna supports Linux, MacOS, and Windows.  Some algorithms may have OS restrictions.

**Prerequisites:**

*   Python 3.9 or higher
*   (Optional) CUDA Toolkit for GPU support

**Installation Options:**

#### Install via pip:

```bash
pip install pruna
```

#### Install from Source:

```bash
git clone https://github.com/pruna-ai/pruna.git
cd pruna
pip install -e .
```

## Quick Start

Get started with Pruna in a few simple steps.

**1. Load a Pre-trained Model:**

```python
from diffusers import StableDiffusionPipeline
base_model = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
```

**2. Optimize with `smash`:**

```python
from pruna import smash, SmashConfig

smash_config = SmashConfig()
smash_config["cacher"] = "deepcache"
smash_config["compiler"] = "stable_fast"
smashed_model = smash(model=base_model, smash_config=smash_config)
```

**3. Use the Optimized Model:**

```python
smashed_model("An image of a cute prune.").images[0]
```

**4. Evaluate Performance**

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

Refer to the [documentation][documentation] for detailed information and examples.

## Pruna Pro

Unlock even greater model efficiency with Pruna Pro, including proprietary Auto Caching and more.  Here's a glimpse of the results achieved with Pruna Pro.

### Stable Diffusion XL

Combining Auto Caching with DeepCache and torch.compile, plus HQQ 8-bit quantization:

*   **9%** reduction in inference latency
*   Model size reduced from **8.8GB** to **6.7GB**

<img src="./docs/assets/plots/benchmark_sdxl.svg" alt="SDXL Benchmark"/>

### FLUX \[dev]

Using Auto Caching with TeaCache, Stable Fast, and HQQ 8-bit quantization:

*   Additional **13%** latency reduction with Auto Caching
*   Model size reduced from **33GB** to **23GB**

<img src="./docs/assets/plots/benchmark_flux.svg" alt="FLUX [dev] Benchmark"/>

### HunyuanVideo

Applying Auto Caching with TeaCache and HQQ 8-bit quantization:

*   Model size reduced from **41GB** to **29GB**

<img src="./docs/assets/plots/benchmark_hunyuan.svg" alt="HunyuanVideo Benchmark"/>

## Algorithm Overview

Pruna offers a wide range of optimization algorithms.

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

Consult the [documentation][documentation], [FAQs][docs-faq], and existing issues for solutions. For further assistance, connect with the community on [Discord][discord], join our [Office Hours][docs-office-hours], or open an issue on GitHub.

## Contributors

Pruna is a project built with 💜 by the Pruna AI team and contributors. [Contribute here][docs-contributing]!

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