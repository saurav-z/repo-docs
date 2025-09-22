<p align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_white.svg">
        <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_black.svg">
        <img alt="Axolotl" src="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_black.svg" width="400" height="104" style="max-width: 100%;">
    </picture>
</p>

## Axolotl: Supercharge Your LLMs with Open-Source Fine-tuning

Fine-tune your large language models (LLMs) efficiently and effectively with Axolotl, a free and open-source framework.

**[View the original repository](https://github.com/axolotl-ai-cloud/axolotl)**

<p align="center">
    <img src="https://img.shields.io/github/license/axolotl-ai-cloud/axolotl.svg?color=blue" alt="GitHub License">
    <img src="https://github.com/axolotl-ai-cloud/axolotl/actions/workflows/tests.yml/badge.svg" alt="tests">
    <a href="https://codecov.io/gh/axolotl-ai-cloud/axolotl"><img src="https://codecov.io/gh/axolotl-ai-cloud/axolotl/branch/main/graph/badge.svg" alt="codecov"></a>
    <a href="https://github.com/axolotl-ai-cloud/axolotl/releases"><img src="https://img.shields.io/github/release/axolotl-ai-cloud/axolotl.svg" alt="Releases"></a>
    <br/>
    <a href="https://github.com/axolotl-ai-cloud/axolotl/graphs/contributors"><img src="https://img.shields.io/github/contributors-anon/axolotl-ai-cloud/axolotl?color=yellow&style=flat-square" alt="contributors" style="height: 20px;"></a>
    <img src="https://img.shields.io/github/stars/axolotl-ai-cloud/axolotl" alt="GitHub Repo stars">
    <br/>
    <a href="https://discord.com/invite/HhrNrHJPRb"><img src="https://img.shields.io/badge/discord-7289da.svg?style=flat-square&logo=discord" alt="discord" style="height: 20px;"></a>
    <a href="https://twitter.com/axolotl_ai"><img src="https://img.shields.io/twitter/follow/axolotl_ai?style=social" alt="twitter" style="height: 20px;"></a>
    <a href="https://colab.research.google.com/github/axolotl-ai-cloud/axolotl/blob/main/examples/colab-notebooks/colab-axolotl-example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google-colab" style="height: 20px;"></a>
    <br/>
    <img src="https://github.com/axolotl-ai-cloud/axolotl/actions/workflows/tests-nightly.yml/badge.svg" alt="tests-nightly">
    <img src="https://github.com/axolotl-ai-cloud/axolotl/actions/workflows/multi-gpu-e2e.yml/badge.svg" alt="multigpu-semi-weekly tests">
</p>

## Key Features

*   **Broad Model Support:** Train a wide variety of LLMs, including GPT-OSS, LLaMA, Mistral, Mixtral, and many more from Hugging Face Hub.
*   **Multimodal Capabilities:** Fine-tune vision-language models (VLMs) like LLaMA-Vision, Qwen2-VL, and audio models such as Voxtral.
*   **Comprehensive Training Methods:** Utilize various techniques, including full fine-tuning, LoRA, QLoRA, GPTQ, QAT, and preference/reward learning.
*   **Simplified Configuration:** Use a single YAML file for your entire fine-tuning pipeline, streamlining dataset preprocessing, training, evaluation, quantization, and inference.
*   **Performance-Optimized:** Benefit from advanced techniques like Multipacking, Flash Attention, Xformers, Flex Attention, Liger Kernel, Cut Cross Entropy, Sequence Parallelism, LoRA optimizations, and multi-GPU training (FSDP1, FSDP2, DeepSpeed) and multi-node training.
*   **Flexible Data Handling:** Load datasets from local sources, Hugging Face, and cloud storage (S3, Azure, GCP, OCI).
*   **Cloud-Ready Deployment:** Deploy easily with Docker images and PyPI packages, optimized for cloud platforms and local hardware.

## 🎉 Latest Updates

*   **[Insert latest updates here]** (Summarize the most recent updates briefly, highlighting key additions, new model support, and important feature enhancements. Keep it concise.)
    *   **[Example: ND Parallelism, New Model Support (GPT-OSS, Gemma 3n, LFM2, AFM), FP8 Finetuning, TiledMLP, QAT support, etc.]**

<details>

<summary>Expand older updates</summary>

*   **[Include older updates here]** (Summarize the previous updates briefly, highlighting key additions, new model support, and important feature enhancements. Keep it concise.)
    *   **[Example: Magistral with mistral-common tokenizer support, Llama 4 support, Beta Multimodal models support, LoRA optimisations and GRPO support etc.]**

</details>

## 🚀 Quick Start - Fine-tune Your LLM in Minutes

**(Instructions simplified to focus on key steps and clarity)**

**Requirements:**

*   NVIDIA GPU (Ampere or newer for `bf16` and Flash Attention) or AMD GPU
*   Python 3.11
*   PyTorch ≥2.6.0

### Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/axolotl-ai-cloud/axolotl/blob/main/examples/colab-notebooks/colab-axolotl-example.ipynb#scrollTo=msOCO4NRmRLa)

### Installation

#### Using pip

```bash
pip3 install -U packaging==23.2 setuptools==75.8.0 wheel ninja
pip3 install --no-build-isolation axolotl[flash-attn,deepspeed]

# Download example axolotl configs, deepspeed configs
axolotl fetch examples
axolotl fetch deepspeed_configs  # OPTIONAL
```

#### Using Docker

```bash
docker run --gpus '"all"' --rm -it axolotlai/axolotl:main-latest
```

#### Cloud Providers

<details>

-   [RunPod](https://runpod.io/gsc?template=v2ickqhz9s&ref=6i7fkpdz)
-   [Vast.ai](https://cloud.vast.ai?ref_id=62897&template_id=bdd4a49fa8bce926defc99471864cace&utm_source=github&utm_medium=developer_community&utm_campaign=template_launch_axolotl&utm_content=readme)
-   [PRIME Intellect](https://app.primeintellect.ai/dashboard/create-cluster?image=axolotl&location=Cheapest&security=Cheapest&show_spot=true)
-   [Modal](https://www.modal.com?utm_source=github&utm_medium=github&utm_campaign=axolotl)
-   [Novita](https://novita.ai/gpus-console?templateId=311)
-   [JarvisLabs.ai](https://jarvislabs.ai/templates/axolotl)
-   [Latitude.sh](https://latitude.sh/blueprint/989e0e79-3bf6-41ea-a46b-1f246e309d5c)

</details>

### Your First Fine-tune

```bash
# Fetch axolotl examples
axolotl fetch examples

# Or, specify a custom path
axolotl fetch examples --dest path/to/folder

# Train a model using LoRA
axolotl train examples/llama-3/lora-1b.yml
```

Check out the [Getting Started Guide](https://docs.axolotl.ai/docs/getting-started.html) for a detailed walkthrough.

## 📚 Documentation & Resources

*   **Installation Options:** [Installation Guide](https://docs.axolotl.ai/docs/installation.html)
*   **Configuration:** [Configuration Guide](https://docs.axolotl.ai/docs/config-reference.html)
*   **Dataset Loading:** [Dataset Loading Guide](https://docs.axolotl.ai/docs/dataset_loading.html)
*   **Dataset Formats:** [Dataset Formats](https://docs.axolotl.ai/docs/dataset-formats/)
*   **Multi-GPU & Multi-Node Training:** [Multi-GPU Training](https://docs.axolotl.ai/docs/multi-gpu.html), [Multi-Node Training](https://docs.axolotl.ai/docs/multi-node.html)
*   **Performance:** [Multipacking](https://docs.axolotl.ai/docs/multipack.html), [Sequence Parallelism](https://docs.axolotl.ai/docs/sequence_parallelism.html)
*   **API Reference:** [API Reference](https://docs.axolotl.ai/docs/api/)
*   **FAQ:** [FAQ](https://docs.axolotl.ai/docs/faq.html)

## 🤝 Get Help

*   **Discord:** [Discord Community](https://discord.gg/HhrNrHJPRb)
*   **Examples:** [Examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/)
*   **Debugging:** [Debugging Guide](https://docs.axolotl.ai/docs/debugging.html)
*   **Dedicated Support:** Contact [✉️wing@axolotl.ai](mailto:wing@axolotl.ai)

## 🌟 Contribute

Contribute to the project - [Contributing Guide](https://github.com/axolotl-ai-cloud/axolotl/blob/main/.github/CONTRIBUTING.md).

## ❤️ Sponsors

Interested in sponsoring? Contact us at [wing@axolotl.ai](mailto:wing@axolotl.ai)

## 📝 Citing Axolotl

```bibtex
@software{axolotl,
  title = {Axolotl: Open Source LLM Post-Training},
  author = {{Axolotl maintainers and contributors}},
  url = {https://github.com/axolotl-ai-cloud/axolotl},
  license = {Apache-2.0},
  year = {2023}
}
```

## 📜 License

Licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file.