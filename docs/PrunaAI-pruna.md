<div align="center">

<img src="./docs/assets/images/logo.png" alt="Pruna AI Logo" width=400></img>

  <img src="./docs/assets/images/element.png" alt="Element" width=10></img>
  **Supercharge your AI models with Pruna: Faster, Cheaper, Smaller, Greener!**
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
[![Devto](https://img.shields/badge/dev-to-black?style=flat-square)][devto]
[![Reddit](https://img.shields.io/badge/Follow-r%2FPrunaAI-orange?style=social)][reddit]
[![Discord](https://img.shields.io/badge/Discord-join_us-purple?style=flat-square)][discord]
[![Huggingface](https://img.shields.io/badge/Huggingface-models-yellow?style=flat-square)][huggingface]
[![Replicate](https://img.shields.io/badge/replicate-black?style=flat-square)][replicate]

<br>

<img src="./docs/assets/images/triple_line.png" alt="Pruna AI Logo" width=600, height=30></img>

</div>

## **Pruna: Optimize Your AI Models for Peak Performance**

Pruna is a powerful model optimization framework designed to help developers build AI models that are faster, more efficient, and less resource-intensive. Leveraging a suite of cutting-edge compression techniques, Pruna empowers you to optimize your AI models with minimal effort.  **[Visit the Pruna GitHub repository](https://github.com/PrunaAI/pruna) to get started.**

### **Key Features**

*   **🚀 Faster Inference:** Accelerate model execution with advanced optimization algorithms.
*   **💾 Smaller Model Sizes:** Reduce model footprint without compromising performance.
*   **💰 Lower Costs:** Minimize computational expenses and resource requirements.
*   **🌱 Greener AI:** Decrease energy consumption and environmental impact.
*   **🛠️ Comprehensive Toolkit:** Includes caching, quantization, pruning, distillation, and compilation techniques.
*   **🌐 Broad Model Support:** Compatible with a wide range of models including LLMs, Diffusion Models, Vision Transformers, and more.
*   **💻 Easy Integration:** Optimize models with just a few lines of code.

## **Installation**

Pruna can be installed on Linux, macOS, and Windows.

**Prerequisites:**

*   Python 3.9 or higher
*   Optional: CUDA toolkit for GPU support

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

## **Quick Start**

Optimize your models in a few simple steps:

```python
from diffusers import StableDiffusionPipeline
from pruna import smash, SmashConfig

# 1. Load a pre-trained model
base_model = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")

# 2. Configure the optimization process
smash_config = SmashConfig()
smash_config["cacher"] = "deepcache"
smash_config["compiler"] = "stable_fast"

# 3. Smash your model!
smashed_model = smash(model=base_model, smash_config=smash_config)

# 4. Use the optimized model
smashed_model("An image of a cute prune.").images[0]
```

For detailed examples and more advanced configurations, see the [Pruna documentation][documentation].

## **Pruna Pro: Unleash the Full Potential**

For advanced features and even greater optimization, explore [Pruna Pro](https://docs.pruna.ai/en/stable/docs_pruna_pro/user_manual/pruna_pro.html), our enterprise solution. Pruna Pro includes:

*   Advanced Optimization Features
*   OptimizationAgent
*   Priority Support
*   ...and more!

### **Benchmark Examples with Pruna Pro**

Here's how Pruna Pro boosts performance on popular models:

**Stable Diffusion XL:**

*   Combination of Auto Caching, [DeepCache](https://github.com/horseee/DeepCache), and `torch.compile`.
*   Uses [HQQ](https://github.com/mobiusml/hqq) 8-bit quantization to reduce the size of the model from 8.8GB to 6.7GB.

<img src="./docs/assets/plots/benchmark_sdxl.svg" alt="SDXL Benchmark"/>

**FLUX [dev]:**

*   Combines Auto Caching with [TeaCache](https://github.com/ali-vilab/TeaCache).
*   Uses [Stable Fast](https://github.com/chengzeyi/stable-fast) and [HQQ](https://github.com/mobiusml/hqq) 8-bit quantization to reduce the size of FLUX from 33GB to 23GB.

<img src="./docs/assets/plots/benchmark_flux.svg" alt="FLUX [dev] Benchmark"/>

**HunyuanVideo:**

*   Combines Auto Caching with [TeaCache](https://github.com/ali-vilab/TeaCache).
*   Applying [HQQ](https://github.com/mobiusml/hqq) 8-bit quantization to the model reduced the size from 41GB to 29GB.

<img src="./docs/assets/plots/benchmark_hunyuan.svg" alt="HunyuanVideo Benchmark"/>

## **Algorithm Overview**

Pruna offers a variety of optimization algorithms. See the [documentation][documentation] for detailed descriptions.

| Technique     | Description                                                                                     | Speed | Memory | Quality |
| ------------- | ----------------------------------------------------------------------------------------------- |:-----:|:------:|:-------:|
| `batcher`     | Groups inputs together to be processed simultaneously, improving computational efficiency.          | ✅    | ❌     | ➖      |
| `cacher`      | Stores intermediate results of computations to speed up subsequent operations.                  | ✅    | ➖     | ➖      |
| `compiler`    | Optimizes the model with instructions for specific hardware.                                    | ✅    | ➖     | ➖      |
| `distiller`   | Trains a smaller model to mimic a larger model.                                                | ✅    | ✅     | ❌      |
| `quantizer`   | Reduces the precision of weights and activations, lowering memory requirements.                 | ✅    | ✅     | ❌      |
| `pruner`      | Removes less important connections and neurons.                                                 | ✅    | ✅     | ❌      |
| `recoverer`   | Restores the performance of a model after compression.                                          | ➖    | ➖     | ✅      |
| `factorizer`  | Batches several small matrix multiplications into one large fused operation.                    | ✅    | ➖     | ➖      |
| `enhancer`    | Enhances the model output by applying post-processing algorithms such as denoising or upscaling. | ❌    | ➖     | ✅      |
| `distributer` | Distributes the inference, the model or certain calculations across multiple devices.           | ✅    | ❌     | ➖      |
| `kernel`      | Specialized GPU routines that speed up parts of the computation.                                | ✅    | ➖     | ➖      |

✅ (improves), ➖ (approx. the same), ❌ (worsens)

<br><br>

<p align="center"><img src="./docs/assets/images/single_line.png" alt="Pruna AI Logo" width=600, height=30></img></p>

<br>

## **FAQ and Troubleshooting**

Find answers to common questions and troubleshooting tips in the [documentation][documentation], [FAQs][docs-faq], or existing issues.  Need help? Join the [Discord][discord], attend [Office Hours][docs-office-hours], or open an issue on GitHub.

## **Contribute**

Help build Pruna! The project welcomes contributions from the community. Become part of the Pruna family.

[![Contributors](https://contrib.rocks/image?repo=PrunaAI/pruna)](https://github.com/PrunaAI/pruna/graphs/contributors)

## **Citation**

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