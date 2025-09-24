<div align="center">
    <img src="./docs/assets/images/logo.png" alt="Pruna AI Logo" width=400>
    <img src="./docs/assets/images/element.png" alt="Element" width=10>
    **Pruna: Revolutionize AI Models by making them Faster, Cheaper, Smaller, and Greener!**
    <img src="./docs/assets/images/element.png" alt="Element" width=10>
    <br>
    <a href="https://github.com/PrunaAI/pruna">
        <img src="https://img.shields.io/badge/View_on_GitHub-gray?style=for-the-badge&logo=github" alt="View on GitHub">
    </a>
</div>

## Key Features

*   🚀 **Accelerate** inference times with advanced optimization.
*   🤏 **Reduce** model size while maintaining quality.
*   💰 **Lower** computational costs and resource requirements.
*   🌱 **Decrease** energy consumption and environmental impact.
*   🛠️ **Easy-to-Use:** Optimize models with just a few lines of code.
*   🧠 **Broad Compatibility:** Supports various model types, including LLMs, Diffusion Models, Vision Transformers, and more.

##  Pruna: The AI Model Optimization Framework

Pruna is an innovative model optimization framework designed to make AI models faster, smaller, cheaper, and more environmentally friendly. Built for developers, Pruna provides a comprehensive suite of optimization algorithms.  Reduce model size and inference time without sacrificing quality.

### Installation

Pruna is available on Linux, MacOS, and Windows. Before installing, ensure you have Python 3.9 or higher, and optionally, the CUDA toolkit for GPU support.

**Option 1: Install using pip**

```bash
pip install pruna
```

**Option 2: Install from source**

```bash
git clone https://github.com/pruna-ai/pruna.git
cd pruna
pip install -e .
```

## Quick Start

Optimize your models with Pruna in three simple steps!

1.  **Load your pre-trained model:**
    ```python
    from diffusers import StableDiffusionPipeline
    base_model = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
    ```

2.  **Use Pruna's `smash` function to optimize:**
    ```python
    from pruna import smash, SmashConfig

    # Create and smash your model
    smash_config = SmashConfig()
    smash_config["cacher"] = "deepcache"
    smash_config["compiler"] = "stable_fast"
    smashed_model = smash(model=base_model, smash_config=smash_config)
    ```

3.  **Use your optimized model:**
    ```python
    smashed_model("An image of a cute prune.").images[0]
    ```
    For advanced usage and detailed examples, see the [Pruna documentation][documentation].

## Pruna Pro

For even greater optimization, explore [Pruna Pro](https://docs.pruna.ai/en/stable/docs_pruna_pro/user_manual/pruna_pro.html), our enterprise solution that unlocks advanced features like our Auto Caching algorithm.

### Performance Benchmarks (Pruna Pro)

*   **Stable Diffusion XL:** Using Auto Caching and HQQ quantization, reduce model size and inference latency.
    <img src="./docs/assets/plots/benchmark_sdxl.svg" alt="SDXL Benchmark"/>
*   **FLUX [dev]:** Auto Caching combined with Stable Fast, and HQQ quantization for significant size and latency reductions.
    <img src="./docs/assets/plots/benchmark_flux.svg" alt="FLUX [dev] Benchmark"/>
*   **HunyuanVideo:** Further optimization with Auto Caching and HQQ quantization.
    <img src="./docs/assets/plots/benchmark_hunyuan.svg" alt="HunyuanVideo Benchmark"/>

## Algorithm Overview

Pruna offers a diverse range of optimization algorithms. See the [documentation][documentation] for a detailed explanation of each algorithm.

| Technique     | Description                                                                                    | Speed | Memory | Quality |
|---------------|------------------------------------------------------------------------------------------------|:-----:|:------:|:-------:|
| `batcher`     | Groups inputs for simultaneous processing.                                                    | ✅    | ❌     | ➖      |
| `cacher`      | Stores intermediate results for faster subsequent operations.                                | ✅    | ➖     | ➖      |
| `compiler`    | Optimizes the model for specific hardware.                                                   | ✅    | ➖     | ➖      |
| `distiller`   | Trains a smaller model to mimic a larger one.                                                 | ✅    | ✅     | ❌      |
| `quantizer`   | Reduces precision, lowering memory requirements.                                             | ✅    | ✅     | ❌      |
| `pruner`      | Removes redundant connections and neurons.                                                     | ✅    | ✅     | ❌      |
| `recoverer`   | Restores model performance after compression.                                                 | ➖    | ➖     | ✅      |
| `factorizer`  | Fuses matrix multiplications.                                                                | ✅    | ➖     | ➖      |
| `enhancer`    | Applies post-processing for improved output.                                                  | ❌    | ➖     | ✅      |
| `distributer` | Distributes inference or calculations across multiple devices.                               | ✅    | ❌     | ➖      |
| `kernel`      | Specialized GPU routines for faster computation.                                                | ✅    | ➖     | ➖      |

✅ (improves), ➖ (approx. the same), ❌ (worsens)

## FAQ and Troubleshooting

Find answers to common questions and solutions to problems in our [documentation][documentation], [FAQs][docs-faq], or by opening an issue on GitHub. Get help from the community on [Discord][discord].

## Contributors

Made with 💜 by the Pruna AI team and amazing contributors. [Contribute to the repository][docs-contributing]!

[![Contributors](https://contrib.rocks/image?repo=PrunaAI/pruna)](https://github.com/PrunaAI/pruna/graphs/contributors)

## Citation

If you use Pruna in your research, please cite the project!

```
@misc{pruna,
    title = {Efficient Machine Learning with Pruna},
    year = {2023},
    note = {Software available from pruna.ai},
    url={https://www.pruna.ai/}
}
```

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

*   **Strong Hook:** The one-sentence hook is now a clear statement of benefit, placed prominently at the top.
*   **SEO Optimization:**  Keywords like "AI models," "optimization," "faster," "smaller," "cheaper," and "greener" are used throughout.  Headings and subheadings are logically structured.
*   **Concise Key Features:**  Uses bullet points for readability and highlights the core value proposition.
*   **Clear Structure:** The README is organized with clear headings and subheadings (Introduction, Installation, Quick Start, etc.) for easy navigation.
*   **Detailed Quick Start:**  Provides a complete code example with context.
*   **Pruna Pro Section:** Highlights Pruna Pro's advantages, with benchmarks.
*   **Algorithm Table:**  Uses a table to clearly summarize the available techniques.
*   **Call to Action:** Encourages users to get involved with the project.
*   **Contributor Section:**  Keeps the contributor section intact and updated.
*   **Citation Information:** Preserves citation information.
*   **Removed Redundant Visuals:** Reduced the number of images for a cleaner look.
*   **Simplified Language:** Uses more straightforward language for clarity.
*   **GitHub Link:** Added a badge linking directly back to the original repository.
*   **Broken down original's text**: Restructured the original text to create more readability.

This improved version is more informative, engaging, and SEO-friendly, making it easier for users to understand and benefit from Pruna.