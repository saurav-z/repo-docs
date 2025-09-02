<div align="center">

<img src="./docs/assets/images/logo.png" alt="Pruna AI Logo" width=400></img>

  <img src="./docs/assets/images/element.png" alt="Element" width=10></img>
  **Optimize your AI models for speed, cost, size, and sustainability with Pruna!**
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

## 🚀 Pruna: Supercharge Your AI Models

Pruna is a powerful model optimization framework designed to make your AI models faster, smaller, cheaper, and greener, empowering developers to deploy highly efficient models with minimal effort.  Check out the [Pruna](https://github.com/PrunaAI/pruna) repository!

**Key Features:**

*   **Speed Up Inference:** Accelerate model performance with advanced optimization techniques.
*   **Reduce Model Size:** Decrease model footprint while preserving quality.
*   **Lower Costs:** Minimize computational expenses and resource demands.
*   **Enhance Sustainability:** Reduce energy consumption and environmental impact.
*   **Easy to Use:** Optimize your models with just a few lines of code.
*   **Wide Model Support:** Compatible with LLMs, Diffusion Models, Vision Transformers, Speech Recognition Models, and more.

## ⚙️ Installation

Pruna is available for Linux, macOS, and Windows.

**Prerequisites:**

*   Python 3.9 or higher
*   Optional: [CUDA toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) for GPU support

**Installation Options:**

1.  **Install from PyPI (Recommended):**

    ```bash
    pip install pruna
    ```

2.  **Install from Source:**

    ```bash
    git clone https://github.com/pruna-ai/pruna.git
    cd pruna
    pip install -e .
    ```

## 🏁 Quick Start

Optimize your models with these simple steps!

```python
from diffusers import StableDiffusionPipeline
from pruna import smash, SmashConfig

# 1. Load a pre-trained model
base_model = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")

# 2. Configure and optimize with 'smash'
smash_config = SmashConfig()
smash_config["cacher"] = "deepcache"
smash_config["compiler"] = "stable_fast"
smashed_model = smash(model=base_model, smash_config=smash_config)

# 3. Use the optimized model
smashed_model("An image of a cute prune.").images[0]
```

Explore our [documentation][documentation] for detailed algorithm information and tutorials!

## ✨ Pruna Pro

For advanced optimization and features, explore [Pruna Pro](https://docs.pruna.ai/en/stable/docs_pruna_pro/user_manual/pruna_pro.html), offering proprietary optimization algorithms, an `OptimizationAgent`, and priority support.

### Performance Highlights (Examples):

*   **Stable Diffusion XL:** Combining Auto Caching with DeepCache, torch.compile, and HQQ 8-bit quantization can provide significant speedups and size reductions.
    <img src="./docs/assets/plots/benchmark_sdxl.svg" alt="SDXL Benchmark"/>
*   **FLUX [dev]:** Auto Caching, Stable Fast, and HQQ 8-bit quantization can lead to substantial improvements.
    <img src="./docs/assets/plots/benchmark_flux.svg" alt="FLUX [dev] Benchmark"/>
*   **HunyuanVideo:** Auto Caching and HQQ 8-bit quantization can also result in meaningful size reduction.
    <img src="./docs/assets/plots/benchmark_hunyuan.svg" alt="HunyuanVideo Benchmark"/>

## 🛠️ Algorithm Overview

Pruna provides a wide range of optimization algorithms.  Refer to our [documentation](https://docs.pruna.ai/en/stable/) for detailed descriptions.

| Technique     | Description                                                                                   | Speed | Memory | Quality |
|---------------|-----------------------------------------------------------------------------------------------|:-----:|:------:|:-------:|
| `batcher`     | Groups inputs for simultaneous processing.                                                     | ✅    | ❌     | ➖      |
| `cacher`      | Stores intermediate results for faster reuse.                                                 | ✅    | ➖     | ➖      |
| `compiler`    | Optimizes models for specific hardware.                                                        | ✅    | ➖     | ➖      |
| `distiller`   | Trains a smaller model to mimic a larger one.                                                 | ✅    | ✅     | ❌      |
| `quantizer`   | Reduces precision of weights and activations.                                                | ✅    | ✅     | ❌      |
| `pruner`      | Removes less important connections and neurons.                                               | ✅    | ✅     | ❌      |
| `recoverer`   | Restores model performance after compression.                                                | ➖    | ➖     | ✅      |
| `factorizer`  | Factorization batches several small matrix multiplications into one large fused operation. | ✅ | ➖ | ➖ |
| `enhancer`    | Enhances the model output by applying post-processing algorithms.                                | ❌ | ➖ | ✅ |
| `distributer`    | Distributes inference or calculations across multiple devices.                              | ✅ | ❌ | ➖ |
| `kernel`      | Utilizes specialized GPU routines.                                                              | ✅ | ➖ | ➖ |

✅ (improves), ➖ (approx. the same), ❌ (worsens)

<br><br>

<p align="center"><img src="./docs/assets/images/single_line.png" alt="Pruna AI Logo" width=600, height=30></img></p>

<br>

## ❓ FAQ and Troubleshooting

Find answers in our [documentation][documentation], [FAQs][docs-faq], and existing issues. Get help from the community on [Discord][discord], attend our [Office Hours][docs-office-hours], or [open an issue](https://github.com/PrunaAI/pruna/issues).

## 💖 Contributing

Contribute to Pruna and be part of our community!  See the [contributing guidelines][docs-contributing].

[![Contributors](https://contrib.rocks/image?repo=PrunaAI/pruna)](https://github.com/PrunaAI/pruna/graphs/contributors)

## ✍️ Citation

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
Key improvements:

*   **SEO-optimized Title & Introduction:**  Added a strong hook and targeted keywords like "AI model optimization," "faster," "smaller," "cheaper," "greener."
*   **Clear Headings:**  Used clear and concise headings for readability and SEO.
*   **Bulleted Key Features:**  Highlights the core benefits.
*   **Concise Language:**  Streamlined text for clarity.
*   **Emphasis on Benefits:**  Focused on what users gain (speed, cost savings, sustainability).
*   **Clearer Installation Instructions:**  Simplified and emphasized installation.
*   **Quick Start Example:**  Kept the essential quick start but improved the formatting.
*   **Pruna Pro Section:** Highlighted benefits of the Pro version, including benchmarks.
*   **Algorithm Overview Table:** Added a table for quick reference.
*   **FAQ/Troubleshooting & Contributing Sections:** Encourages user interaction and community involvement.
*   **Citation:** Included the citation information.
*   **Link back to Original Repo:**  The link is included at the beginning and the end of the text.
*   **Images**: Maintained image links.
*   **Clean Formatting:** Improved formatting and readability.
*   **Added a direct link to the Github Repo.**