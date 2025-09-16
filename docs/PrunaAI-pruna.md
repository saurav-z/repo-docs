<div align="center">

<img src="./docs/assets/images/logo.png" alt="Pruna AI Logo" width=400>
<br>
<img src="./docs/assets/images/element.png" alt="Element" width=10>
**Pruna: Make AI Models Faster, Cheaper, and Greener!**
<img src="./docs/assets/images/element.png" alt="Element" width=10>
<br>
<br>

[![Documentation](https://img.shields.io/badge/Pruna_documentation-purple?style=for-the-badge)][documentation]
<br>
<br>

![GitHub License](https://img.shields.io/github/license/prunaai/pruna?style=flat-square)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/prunaai/pruna/build.yaml?style=flat-square)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/prunaai/pruna/tests.yaml?label=tests&style=flat-square)
![GitHub Release](https://img.shields.io/github/v/release/prunaai/pruna?style=flat-square)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/PrunaAI/pruna?style=flat-square)
![PyPI - Downloads](https://img.shields.io/pypi/dm/pruna?style=flat-square)
![Codacy](https://app.codacy.com/project/badge/Grade/092392ec4be846928a7c5978b6afe060)
<br>
[![Website](https://img.shields.io/badge/Pruna.ai-purple?style=flat-square)][website]
[![X (formerly Twitter) URL](https://img.shields.io/twitter/url?url=https%3A%2F%2Fx.com%2FPrunaAI)][x]
[![Devto](https://img.shields.io/badge/dev-to-black?style=flat-square)][devto]
[![Reddit](https://img.shields.io/badge/Follow-r%2FPrunaAI-orange?style=social)][reddit]
[![Discord](https://img.shields.io/badge/Discord-join_us-purple?style=flat-square)][discord]
[![Huggingface](https://img.shields.io/badge/Huggingface-models-yellow?style=flat-square)][huggingface]
[![Replicate](https://img.shields.io/badge/replicate-black?style=flat-square)][replicate]

<br>
<img src="./docs/assets/images/triple_line.png" alt="Pruna AI Logo" width=600, height=30>
</div>

## 🚀 **Pruna: Supercharge Your AI Models**

Pruna is a powerful open-source framework designed to optimize AI models, making them faster, smaller, cheaper, and more environmentally friendly.  [Explore the Pruna AI repository](https://github.com/PrunaAI/pruna).

## ✨ Key Features

*   **Speed Up Inference:** Accelerate model performance using advanced optimization techniques.
*   **Reduce Model Size:** Significantly decrease model size without sacrificing quality.
*   **Lower Costs:** Minimize computational expenses and resource requirements.
*   **Go Green:** Reduce energy consumption and minimize the environmental impact of your AI models.
*   **Comprehensive Optimization:** Offers caching, quantization, pruning, distillation, and compilation methods.
*   **User-Friendly:** Optimize your models with just a few lines of code.
*   **Broad Compatibility:** Supports various model types, including LLMs, Diffusion Models, Vision Transformers, and Speech Recognition Models.

## 🛠️ Installation

Pruna is available for Linux, macOS, and Windows. Ensure you have Python 3.9+ and, optionally, the CUDA toolkit for GPU support.

**Install with pip:**

```bash
pip install pruna
```

**Install from source:**

```bash
git clone https://github.com/pruna-ai/pruna.git
cd pruna
pip install -e .
```

## 💡 Quick Start

Optimize your models quickly with Pruna. Here's a basic example:

```python
from diffusers import StableDiffusionPipeline
from pruna import smash, SmashConfig

# Load a pre-trained model (e.g., Stable Diffusion)
base_model = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")

# Configure and apply Pruna's optimization
smash_config = SmashConfig()
smash_config["cacher"] = "deepcache"
smash_config["compiler"] = "stable_fast"
smashed_model = smash(model=base_model, smash_config=smash_config)

# Use the optimized model
smashed_model("An image of a cute prune.").images[0]
```

For detailed examples and available algorithms, see the [documentation][documentation].

## 🌟 Pruna Pro

Unlock advanced features and priority support with Pruna Pro, our enterprise solution.  Learn more in our documentation.

## 📊 Algorithm Overview

Pruna provides a suite of optimization techniques.

| Technique      | Description                                                                                             | Speed | Memory | Quality |
|----------------|---------------------------------------------------------------------------------------------------------|:-----:|:------:|:-------:|
| `batcher`      | Groups multiple inputs for simultaneous processing, improving efficiency and reducing processing time. | ✅    | ❌     | ➖      |
| `cacher`       | Stores intermediate computation results to speed up subsequent operations.                              | ✅    | ➖     | ➖      |
| `compiler`     | Optimizes model for specific hardware instructions.                                                     | ✅    | ➖     | ➖      |
| `distiller`    | Trains a smaller model to mimic a larger one.                                                           | ✅    | ✅     | ❌      |
| `quantizer`    | Reduces precision of weights/activations, lowering memory needs.                                        | ✅    | ✅     | ❌      |
| `pruner`       | Removes less significant connections and neurons.                                                      | ✅    | ✅     | ❌      |
| `recoverer`    | Restores model performance after compression.                                                             | ➖    | ➖     | ✅      |
| `factorizer`   | Fuses several small matrix multiplications into one large operation.                                   | ✅    | ➖     | ➖      |
| `enhancer`     | Enhances model output via post-processing algorithms (denoising, upscaling).                          | ❌ | ➖ | ✅ |
| `distributer`  | Distributes inference/calculations/model across devices.                                                                     | ✅ | ❌ | ➖ |
| `kernel`       | Specialized GPU routines to speed up computations.  | ✅ | ➖ | ➖ |

✅ (improves), ➖ (approx. the same), ❌ (worsens)

<br><br>

<p align="center"><img src="./docs/assets/images/single_line.png" alt="Pruna AI Logo" width=600, height=30></img></p>

<br>

## ❓ FAQ and Troubleshooting

Visit the [documentation][documentation], [FAQs][docs-faq] or open an issue on GitHub to get help!

## 🤝 Contributors

Pruna is made with 💜 by the Pruna AI team and the community.  [Contribute to the repository][docs-contributing]!

[![Contributors](https://contrib.rocks/image?repo=PrunaAI/pruna)](https://github.com/PrunaAI/pruna/graphs/contributors)

## 📝 Citation

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