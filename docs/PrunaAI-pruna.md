<div align="center">

<img src="./docs/assets/images/logo.png" alt="Pruna AI Logo" width=400>

  <img src="./docs/assets/images/element.png" alt="Element" width=10>
  **Supercharge your AI models: Make them faster, cheaper, smaller, and greener with Pruna!**
  <img src="./docs/assets/images/element.png" alt="Element" width=10>

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

<img src="./docs/assets/images/triple_line.png" alt="Pruna AI Logo" width=600, height=30>

</div>

## **Pruna: Optimize Your AI Models for Peak Performance**

Pruna is a powerful model optimization framework designed to help developers create more efficient, faster, and cost-effective AI models.  Leverage cutting-edge compression techniques to drastically improve performance with minimal code changes.  [Get started with Pruna on GitHub!](https://github.com/PrunaAI/pruna)

### **Key Features:**

*   **Faster Inference:** Accelerate model execution with advanced optimization algorithms.
*   **Smaller Model Sizes:** Reduce memory footprint without sacrificing quality.
*   **Lower Costs:** Decrease computational resource requirements and expenses.
*   **Greener AI:** Minimize energy consumption and environmental impact.

Pruna supports various model types, including LLMs, Diffusion Models, Vision Transformers, and Speech Recognition Models.

## **Installation**

Pruna is compatible with Linux, macOS, and Windows.

**Prerequisites:**

*   Python 3.9 or higher
*   Optional: [CUDA toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) for GPU support

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

## **Quick Start: Optimize Your First Model**

Quickly optimize a pre-trained model with just a few lines of code.

```python
from diffusers import StableDiffusionPipeline
from pruna import smash, SmashConfig

# Load a pre-trained model
base_model = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")

# Configure and apply Pruna's optimization
smash_config = SmashConfig()
smash_config["cacher"] = "deepcache"
smash_config["compiler"] = "stable_fast"
smashed_model = smash(model=base_model, smash_config=smash_config)

# Use the optimized model
smashed_model("An image of a cute prune.").images[0]
```

Explore detailed examples and advanced techniques in our [documentation][documentation].

## **Pruna Pro: Advanced Optimization for Enterprise Needs**

Unlock advanced features and premium support with Pruna Pro, our enterprise solution.  Pruna Pro offers our `OptimizationAgent`, and more for even greater model efficiency.

*   **Stable Diffusion XL:** Auto Caching + DeepCache + torch.compile + HQQ (8-bit quantization)
    *   9% latency reduction
    *   Model size reduced from 8.8GB to 6.7GB.
*   **FLUX \[dev]:** Auto Caching + TeaCache + Stable Fast + HQQ (8-bit quantization)
    *   13% latency reduction
    *   Model size reduced from 33GB to 23GB.
*   **HunyuanVideo:** Auto Caching + TeaCache + HQQ (8-bit quantization)
    *   Model size reduced from 41GB to 29GB.

## **Algorithm Overview**

Pruna offers a wide range of optimization algorithms.

| Technique       | Description                                                                                   | Speed | Memory | Quality |
|-----------------|-----------------------------------------------------------------------------------------------|:-----:|:------:|:-------:|
| `batcher`       | Groups multiple inputs together to be processed simultaneously, improving computational efficiency and reducing processing time. | ✅    | ❌     | ➖      |
| `cacher`        | Stores intermediate results of computations to speed up subsequent operations.               | ✅    | ➖     | ➖      |
| `compiler`      | Optimises the model with instructions for specific hardware.                                 | ✅    | ➖     | ➖      |
| `distiller`     | Trains a smaller, simpler model to mimic a larger, more complex model.                       | ✅    | ✅     | ❌      |
| `quantizer`     | Reduces the precision of weights and activations, lowering memory requirements.              | ✅    | ✅     | ❌      |
| `pruner`        | Removes less important or redundant connections and neurons, resulting in a sparser, more efficient network. | ✅    | ✅     | ❌      |
| `recoverer`     | Restores the performance of a model after compression.                                       | ➖    | ➖     | ✅      |
| `factorizer`    | Factorization batches several small matrix multiplications into one large fused operation. | ✅ | ➖ | ➖ |
| `enhancer`      | Enhances the model output by applying post-processing algorithms such as denoising or upscaling. | ❌ | ➖ | ✅ |
| `distributer`   | Distributes the inference, the model or certain calculations across multiple devices. | ✅ | ❌ | ➖ |
| `kernel`        | Kernels are specialized GPU routines that speed up parts of the computation.  | ✅ | ➖ | ➖ |

✅ (improves), ➖ (approx. the same), ❌ (worsens)

<br><br>

<p align="center"><img src="./docs/assets/images/single_line.png" alt="Pruna AI Logo" width=600, height=30>

## **FAQ and Troubleshooting**

Find answers to common questions and solutions to potential problems in our [documentation][documentation], [FAQs][docs-faq], or existing issues.  Join our [Discord][discord] or [Office Hours][docs-office-hours] for additional support.

## **Contribute**

Help improve Pruna! Become part of the Pruna community by [contributing to the repository][docs-contributing].

[![Contributors](https://contrib.rocks/image?repo=PrunaAI/pruna)](https://github.com/PrunaAI/pruna/graphs/contributors)

## **Citation**

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