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
[![Devto](https://img.shields.io/badge/dev-to-black?style=flat-square)][devto]
[![Reddit](https://img.shields.io/badge/Follow-r%2FPrunaAI-orange?style=social)][reddit]
[![Discord](https://img.shields.io/badge/Discord-join_us-purple?style=flat-square)][discord]
[![Huggingface](https://img.shields.io/badge/Huggingface-models-yellow?style=flat-square)][huggingface]
[![Replicate](https://img.shields.io/badge/replicate-black?style=flat-square)][replicate]

<br>

<img src="./docs/assets/images/triple_line.png" alt="Pruna AI Logo" width=600, height=30>
</div>

## **Pruna: AI Model Optimization Framework**

Pruna is an open-source framework designed to optimize AI models, empowering developers to achieve significant improvements in speed, efficiency, and resource utilization with minimal code changes.  **[Check out the original repo](https://github.com/PrunaAI/pruna)**

**Key Features:**

*   **Speed:** Accelerate inference times.
*   **Size Reduction:** Reduce model size while maintaining quality.
*   **Cost Savings:** Lower computational expenses and resource demands.
*   **Environmental Impact:** Decrease energy consumption and promote greener AI.

Pruna supports various model types, including LLMs, Diffusion Models, Vision Transformers, and more.  It offers a range of optimization techniques.

<img align="left" width="40" src="docs/assets/images/highlight.png" alt="Pruna Pro"/>

For advanced optimization features, our `OptimizationAgent`, and priority support consider [Pruna Pro](https://docs.pruna.ai/en/stable/docs_pruna_pro/user_manual/pruna_pro.html).
<br clear="left"/>

## Installation

Pruna is compatible with Linux, MacOS, and Windows.  Ensure you have Python 3.9+ and optionally, the [CUDA toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) for GPU support.

**Installation Options:**

1.  **Using pip:**

    ```bash
    pip install pruna
    ```

2.  **From source:**

    ```bash
    git clone https://github.com/pruna-ai/pruna.git
    cd pruna
    pip install -e .
    ```

## Quick Start

Optimize your models effortlessly with Pruna:

```python
from diffusers import StableDiffusionPipeline
base_model = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
from pruna import smash, SmashConfig

# Configure your optimization
smash_config = SmashConfig()
smash_config["cacher"] = "deepcache"
smash_config["compiler"] = "stable_fast"

# Apply Pruna's magic
smashed_model = smash(model=base_model, smash_config=smash_config)

# Use the optimized model
smashed_model("An image of a cute prune.").images[0]

# Evaluate model performance
from pruna.evaluation.task import Task
from pruna.evaluation.evaluation_agent import EvaluationAgent
from pruna.data.pruna_datamodule import PrunaDataModule

datamodule = PrunaDataModule.from_string("LAION256")
datamodule.limit_datasets(10)
task = Task("image_generation_quality", datamodule=datamodule)
eval_agent = EvaluationAgent(task)
eval_agent.evaluate(smashed_model)
```

Refer to the [documentation][documentation] for detailed algorithm explanations and tutorials.

## Pruna Pro

Pruna Pro offers advanced optimization capabilities. Here are some benchmarks:

### Stable Diffusion XL

Combining Auto Caching with DeepCache and torch.compile to reduce inference latency by 9% and using HQQ 8-bit quantization to reduce model size from 8.8GB to 6.7GB.

<img src="./docs/assets/plots/benchmark_sdxl.svg" alt="SDXL Benchmark"/>

### FLUX [dev]

Using Auto Caching with TeaCache and Stable Fast to reduce the latency by an additional 13% and using HQQ 8-bit reduced the size of FLUX from 33GB to 23GB.

<img src="./docs/assets/plots/benchmark_flux.svg" alt="FLUX [dev] Benchmark"/>

### HunyuanVideo

Applying HQQ 8-bit quantization to the model reduced the size from 41GB to 29GB.

<img src="./docs/assets/plots/benchmark_hunyuan.svg" alt="HunyuanVideo Benchmark"/>

## Algorithm Overview

Explore Pruna's optimization methods:

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
</p>

<br>

## FAQ and Troubleshooting

Find answers in the [documentation][documentation], [FAQs][docs-faq], or by opening an issue on GitHub or on [Discord][discord].

## Contributors

Pruna is developed by the Pruna AI team and community contributors. [Contribute to the repository][docs-contributing]!

[![Contributors](https://contrib.rocks/image?repo=PrunaAI/pruna)](https://github.com/PrunaAI/pruna/graphs/contributors)

## Citation

If you use Pruna in your research, cite the project:

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
</p>

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