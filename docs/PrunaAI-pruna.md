<div align="center">

<img src="./docs/assets/images/logo.png" alt="Pruna AI Logo" width=400></img>

  <img src="./docs/assets/images/element.png" alt="Element" width=10></img>
  **Supercharge your AI models: Make them faster, cheaper, smaller, and greener with Pruna!**
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

## **Pruna: Optimize Your AI Models for Peak Performance**

Pruna is a powerful model optimization framework designed to help developers like you create faster, more efficient, and eco-friendly AI models.  [Learn more on GitHub](https://github.com/PrunaAI/pruna).

**Key Features:**

*   **Speed Up Inference:** Accelerate your models with advanced optimization techniques.
*   **Reduce Model Size:** Decrease memory footprint without sacrificing quality.
*   **Lower Costs:** Minimize computational expenses and resource requirements.
*   **Go Green:** Reduce energy consumption and environmental impact.

Pruna supports a wide range of model types, including LLMs, diffusion models, and vision transformers. It simplifies model optimization with just a few lines of code, making it accessible and easy to use.

## Installation

Pruna is available on Linux, MacOS, and Windows.

**Prerequisites:**

*   Python 3.9 or higher
*   Optional: CUDA toolkit for GPU support

**Installation Options:**

*   **Using pip:**

    ```bash
    pip install pruna
    ```

*   **From Source:**

    ```bash
    git clone https://github.com/pruna-ai/pruna.git
    cd pruna
    pip install -e .
    ```

## Quickstart

Get started with Pruna in just a few steps.

```python
from diffusers import StableDiffusionPipeline
from pruna import smash, SmashConfig

# Load your model
base_model = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")

# Configure and Smash!
smash_config = SmashConfig()
smash_config["cacher"] = "deepcache"
smash_config["compiler"] = "stable_fast"
smashed_model = smash(model=base_model, smash_config=smash_config)

# Use your optimized model
smashed_model("An image of a cute prune.").images[0]
```

Evaluate model performance:

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

Explore the comprehensive documentation for more algorithms, tutorials, and examples.

## Pruna Pro: Unleash Advanced Optimization

For advanced optimization features and support, explore Pruna Pro, our enterprise solution.

**Example Benchmarks:**

*   **Stable Diffusion XL:**  Auto Caching + DeepCache + torch.compile (9% latency reduction) + HQQ quantization (8.8GB to 6.7GB).
*   **FLUX [dev]:** Auto Caching + TeaCache + Stable Fast (13% latency reduction) + HQQ (33GB to 23GB).
*   **HunyuanVideo:** Auto Caching + TeaCache + HQQ (41GB to 29GB).

See detailed benchmarks on the original README to evaluate the benefits!

## Algorithm Overview

Pruna offers a variety of optimization algorithms.  Find a summary below:

| Technique    | Description                                                                                   | Speed | Memory | Quality |
|--------------|-----------------------------------------------------------------------------------------------|:-----:|:------:|:-------:|
| `batcher`    | Groups inputs for simultaneous processing.                                                     | ✅    | ❌     | ➖      |
| `cacher`     | Stores intermediate computations.                                                         | ✅    | ➖     | ➖      |
| `compiler`   | Optimizes for specific hardware.                                                                | ✅    | ➖     | ➖      |
| `distiller`  | Trains a smaller model to mimic a larger one.                       | ✅    | ✅     | ❌      |
| `quantizer`  | Reduces precision for lower memory.              | ✅    | ✅     | ❌      |
| `pruner`     | Removes redundant connections and neurons. | ✅    | ✅     | ❌      |
| `recoverer`  | Restores model performance after compression.                                       | ➖    | ➖     | ✅      |
| `factorizer` | Fuses matrix multiplications. | ✅ | ➖ | ➖ |
| `enhancer`   | Applies post-processing. | ❌ | ➖ | ✅ |
| `distributer`   | Distributes computation across devices. | ✅ | ❌ | ➖ |
| `kernel`   | Specializes GPU routines.  | ✅ | ➖ | ➖ |

✅ (improves), ➖ (approx. the same), ❌ (worsens)

<br><br>

<p align="center"><img src="./docs/assets/images/single_line.png" alt="Pruna AI Logo" width=600, height=30></img></p>

<br>

## FAQ and Troubleshooting

Consult the [documentation][documentation], [FAQ][docs-faq], or existing issues for solutions.  Get help on [Discord][discord], at [Office Hours][docs-office-hours], or by opening a GitHub issue.

## Contributors

Pruna is built with ❤️ by the Pruna AI team and the community. [Contribute here!][docs-contributing]

[![Contributors](https://contrib.rocks/image?repo=PrunaAI/pruna)](https://github.com/PrunaAI/pruna/graphs/contributors)

## Citation

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