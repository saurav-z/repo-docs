<div align="center">

<img src="./docs/assets/images/logo.png" alt="Pruna AI Logo" width=400></img>

  <img src="./docs/assets/images/element.png" alt="Element" width=10></img>
  **Unlock AI efficiency: Make your AI models faster, cheaper, smaller, and greener with Pruna.**
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

Pruna is an open-source model optimization framework designed to make AI models faster, smaller, cheaper, and more environmentally friendly.  Empower your AI models with advanced techniques. This framework is available on [GitHub](https://github.com/PrunaAI/pruna).

**Key Features:**

*   **Faster Inference:** Accelerate model performance with cutting-edge optimization.
*   **Smaller Model Size:** Reduce model size without sacrificing quality.
*   **Lower Costs:** Minimize computational expenses and resource needs.
*   **Greener AI:** Decrease energy consumption and environmental impact.
*   **Comprehensive Techniques:** Utilize caching, quantization, pruning, distillation, and compilation.
*   **Broad Compatibility:** Supports LLMs, Diffusion models, Vision Transformers, Speech Recognition models, and more.
*   **Easy Integration:** Optimize models with just a few lines of code.

## Installation

Pruna is available for Linux, MacOS, and Windows.

**Prerequisites:**
*   Python 3.9 or higher
*   Optional: [CUDA toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) for GPU support

**Install via pip:**

```bash
pip install pruna
```

**Install from Source:**

```bash
git clone https://github.com/pruna-ai/pruna.git
cd pruna
pip install -e .
```

## Quick Start

Get started by loading a pre-trained model, configuring optimization, and then running your optimized model.

```python
from diffusers import StableDiffusionPipeline
from pruna import smash, SmashConfig

# Load a pre-trained model
base_model = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")

# Configure optimization
smash_config = SmashConfig()
smash_config["cacher"] = "deepcache"
smash_config["compiler"] = "stable_fast"

# Apply optimization
smashed_model = smash(model=base_model, smash_config=smash_config)

# Use the optimized model
smashed_model("An image of a cute prune.").images[0]
```

For comprehensive examples and tutorials, consult the [documentation][documentation].

## Pruna Pro: Advanced Optimization

For enhanced efficiency, consider [Pruna Pro](https://docs.pruna.ai/en/stable/docs_pruna_pro/user_manual/pruna_pro.html), the enterprise solution that delivers advanced optimization features, including the `OptimizationAgent` and priority support.

**Benchmarking Results with Pruna Pro**
The following graphs provides benchmark results of Stable Diffusion XL, FLUX, and HunyuanVideo
### Stable Diffusion XL
<img src="./docs/assets/plots/benchmark_sdxl.svg" alt="SDXL Benchmark"/>
### FLUX [dev]
<img src="./docs/assets/plots/benchmark_flux.svg" alt="FLUX [dev] Benchmark"/>
### HunyuanVideo
<img src="./docs/assets/plots/benchmark_hunyuan.svg" alt="HunyuanVideo Benchmark"/>

## Algorithm Overview

Pruna offers a wide range of optimization algorithms. Here is an overview:

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

Refer to the [documentation][documentation] and [FAQs][docs-faq] for answers and solutions.
If you need further assistance, reach out on [Discord][discord], attend our [Office Hours][docs-office-hours], or open an issue on GitHub.

## Contributing

Contribute to the Pruna project!  Become part of the Pruna family by [contributing to the repository][docs-contributing].

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

*   **SEO-Optimized Hook:** The first sentence now directly addresses the problem and value proposition, using keywords like "AI models," "faster," "cheaper," "smaller," and "greener."
*   **Clear Headings:** Added well-defined headings (e.g., "Installation," "Quick Start") for better readability and organization.
*   **Bulleted Key Features:** Used bullet points to highlight the core benefits of Pruna, making it easy to scan and understand.
*   **Concise Language:** Streamlined the text to be more direct and impactful.
*   **Keyword Integration:** Incorporated relevant keywords throughout the text (e.g., "model optimization," "AI efficiency").
*   **Focus on Benefits:** Emphasized the positive outcomes of using Pruna (speed, cost reduction, size reduction, and environmental friendliness).
*   **Call to Action:** Encouraged users to explore the [Pruna GitHub Repo](https://github.com/PrunaAI/pruna).
*   **Revised Examples:** Updated code examples to be more succinct and focused on the core functionality of the code.
*   **Clearer Installation instructions:** The install instructions have been updated to better assist the user.
*   **Clearer Algorithm Overview** Added the performance impact of the techniques in the algorithm overview
*   **Benchmarking Results with Pruna Pro** Added a section for benchmark results for Pruna Pro
*   **Links to the core concepts**. Added links back to the concepts, if they exist in the original doc.