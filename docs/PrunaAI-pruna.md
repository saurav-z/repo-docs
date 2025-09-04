<div align="center">
  <img src="./docs/assets/images/logo.png" alt="Pruna AI Logo" width=400>
  <img src="./docs/assets/images/element.png" alt="Element" width=10>
  **Supercharge your AI models with Pruna: Making them faster, cheaper, and greener!**
  <img src="./docs/assets/images/element.png" alt="Element" width=10>
  <br>
</div>

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

## About Pruna

Pruna is a cutting-edge model optimization framework designed to help developers achieve superior performance and efficiency with their AI models.  Leveraging a suite of powerful compression techniques, Pruna empowers you to dramatically improve your models' speed, reduce their size, lower computational costs, and minimize their environmental impact.

**Key Features:**

*   **Faster Inference:** Accelerate model inference times.
*   **Smaller Model Sizes:** Reduce model size without sacrificing quality.
*   **Reduced Costs:** Lower computational expenses and resource demands.
*   **Greener AI:** Decrease energy consumption and environmental footprint.
*   **User-Friendly:** Optimized for ease of use, requiring minimal code changes.
*   **Broad Compatibility:** Supports various model types, including LLMs, Diffusion Models, Vision Transformers, and Speech Recognition Models.

[Visit our GitHub repository for the source code.](https://github.com/PrunaAI/pruna)

## Installation

Pruna is available for Linux, MacOS, and Windows.

**Prerequisites:**

*   Python 3.9 or higher
*   Optional: [CUDA toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) for GPU acceleration

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

## Quick Start

Get started with Pruna in just a few lines of code.

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

# Evaluate performance (Optional)
from pruna.evaluation.task import Task
from pruna.evaluation.evaluation_agent import EvaluationAgent
from pruna.data.pruna_datamodule import PrunaDataModule

datamodule = PrunaDataModule.from_string("LAION256")
datamodule.limit_datasets(10)
task = Task("image_generation_quality", datamodule=datamodule)
eval_agent = EvaluationAgent(task)
eval_agent.evaluate(smashed_model)
```

Explore all supported [algorithms][docs-algorithms] and detailed [tutorials][documentation].

## Pruna Pro

Unlock advanced optimization features with Pruna Pro, our enterprise solution.  Benefit from enhanced features, the `OptimizationAgent`, priority support, and more.

### Benchmarks

We tested popular diffusers pipelines with Pruna Pro to see how much smaller and faster they can be made.

#### Stable Diffusion XL

| Feature            | Before  | After       | Improvement         |
|--------------------|---------|-------------|---------------------|
| Algorithm          | N/A     | Auto Caching with DeepCache  | N/A                 |
| Model Size         | 8.8GB   | 6.7GB       | Model size reduced  |
| Inference Latency | N/A     | 9%          | Additional Speed  |

<img src="./docs/assets/plots/benchmark_sdxl.svg" alt="SDXL Benchmark"/>

#### FLUX [dev]

| Feature            | Before  | After        | Improvement         |
|--------------------|---------|--------------|---------------------|
| Algorithm          | N/A     | Auto Caching with TeaCache   | N/A                 |
| Model Size         | 33GB    | 23GB         | Model size reduced  |
| Inference Latency  | N/A     | 13%          | Additional Speed  |

<img src="./docs/assets/plots/benchmark_flux.svg" alt="FLUX [dev] Benchmark"/>

#### HunyuanVideo

| Feature            | Before  | After        | Improvement         |
|--------------------|---------|--------------|---------------------|
| Algorithm          | N/A     | Auto Caching with TeaCache  | N/A                 |
| Model Size         | 41GB    | 29GB         | Model size reduced  |

<img src="./docs/assets/plots/benchmark_hunyuan.svg" alt="HunyuanVideo Benchmark"/>

## Algorithm Overview

Pruna offers a wide range of optimization algorithms.

| Technique       | Description                                                                                   | Speed | Memory | Quality |
| --------------- | ----------------------------------------------------------------------------------------------- | :---: | :----: | :-----: |
| `batcher`       | Groups inputs for simultaneous processing, improving efficiency.                              |  ✅   |   ❌   |   ➖   |
| `cacher`        | Stores intermediate computations to speed up subsequent operations.                            |  ✅   |   ➖   |   ➖   |
| `compiler`      | Optimizes the model with instructions for specific hardware.                                 |  ✅   |   ➖   |   ➖   |
| `distiller`     | Trains a smaller model to mimic a larger one.                                                |  ✅   |   ✅   |   ❌   |
| `quantizer`     | Reduces precision of weights and activations, lowering memory requirements.                  |  ✅   |   ✅   |   ❌   |
| `pruner`        | Removes redundant connections and neurons, creating a more efficient network.                |  ✅   |   ✅   |   ❌   |
| `recoverer`     | Restores the performance of a model after compression.                                       |  ➖   |   ➖   |   ✅   |
| `factorizer`    | Factorization batches several small matrix multiplications into one large fused operation. | ✅  | ➖ | ➖ |
| `enhancer`      | Enhances the model output by applying post-processing algorithms such as denoising or upscaling. |  ❌   |   ➖   |   ✅   |
| `distributer`   | Distributes the inference, the model or certain calculations across multiple devices. | ✅ | ❌ | ➖ |
| `kernel`        | Kernels are specialized GPU routines that speed up parts of the computation.  | ✅ | ➖ | ➖ |

✅ (improves), ➖ (approx. the same), ❌ (worsens)

<br><br>

<p align="center"><img src="./docs/assets/images/single_line.png" alt="Pruna AI Logo" width=600, height=30></img></p>

<br>

## FAQ and Troubleshooting

Find answers in our [documentation][documentation], [FAQs][docs-faq], or existing issues. For further assistance, contact the Pruna community on [Discord][discord], join our [Office Hours][docs-office-hours], or open an issue on GitHub.

## Contributors

Made with 💜 by the Pruna AI team and our contributors. [Contribute to the repository][docs-contributing]!

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