<p align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_white.svg">
        <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_black.svg">
        <img alt="Axolotl" src="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_black.svg" width="400" height="104" style="max-width: 100%;">
    </picture>
</p>

## Axolotl: Supercharge Your LLMs with this Free and Open-Source Fine-tuning Framework

Fine-tune your Large Language Models (LLMs) with ease using Axolotl, a powerful and user-friendly framework designed for efficient post-training and fine-tuning.  Get started with Axolotl today and unlock the full potential of your LLMs!  For more information, visit the [original repository](https://github.com/axolotl-ai-cloud/axolotl).

[![GitHub License](https://img.shields.io/github/license/axolotl-ai-cloud/axolotl.svg?color=blue)](https://github.com/axolotl-ai-cloud/axolotl/blob/main/LICENSE)
[![Tests](https://github.com/axolotl-ai-cloud/axolotl/actions/workflows/tests.yml/badge.svg)](https://github.com/axolotl-ai-cloud/axolotl/actions/workflows/tests.yml)
[![Codecov](https://codecov.io/gh/axolotl-ai-cloud/axolotl/branch/main/graph/badge.svg)](https://codecov.io/gh/axolotl-ai-cloud/axolotl)
[![Releases](https://img.shields.io/github/release/axolotl-ai-cloud/axolotl.svg)](https://github.com/axolotl-ai-cloud/axolotl/releases)
[![Contributors](https://img.shields.io/github/contributors-anon/axolotl-ai-cloud/axolotl?color=yellow&style=flat-square)](https://github.com/axolotl-ai-cloud/axolotl/graphs/contributors)
[![GitHub Repo stars](https://img.shields.io/github/stars/axolotl-ai-cloud/axolotl)](https://github.com/axolotl-ai-cloud/axolotl)
[![Discord](https://img.shields.io/badge/discord-7289da.svg?style=flat-square&logo=discord)](https://discord.com/invite/HhrNrHJPRb)
[![Twitter](https://img.shields.io/twitter/follow/axolotl_ai?style=social)](https://twitter.com/axolotl_ai)
[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/axolotl-ai-cloud/axolotl/blob/main/examples/colab-notebooks/colab-axolotl-example.ipynb)
[![Tests Nightly](https://github.com/axolotl-ai-cloud/axolotl/actions/workflows/tests-nightly.yml/badge.svg)](https://github.com/axolotl-ai-cloud/axolotl/actions/workflows/tests-nightly.yml)
[![Multi-GPU Tests](https://github.com/axolotl-ai-cloud/axolotl/actions/workflows/multi-gpu-e2e.yml/badge.svg)](https://github.com/axolotl-ai-cloud/axolotl/actions/workflows/multi-gpu-e2e.yml)


## Key Features of Axolotl

*   **Extensive Model Support:** Train and fine-tune a wide array of LLMs, including popular models like GPT-OSS, LLaMA, Mistral, Mixtral, and many more from the Hugging Face Hub.
*   **Multimodal Capabilities:** Supports fine-tuning of vision-language models (VLMs), such as LLaMA-Vision, Qwen2-VL, and multimodal audio models.
*   **Diverse Training Methods:** Offers a comprehensive range of training techniques, including full fine-tuning, LoRA, QLoRA, GPTQ, QAT, preference tuning, RL (GRPO), and reward modeling.
*   **Simplified Configuration:** Utilize a single YAML configuration file for the entire fine-tuning pipeline, streamlining dataset preprocessing, training, evaluation, quantization, and inference.
*   **Performance Optimization:** Benefit from performance-enhancing features such as Multipacking, Flash Attention, Xformers, Flex Attention, Liger Kernel, Cut Cross Entropy, Sequence Parallelism (SP), LoRA optimizations, multi-GPU training (FSDP1, FSDP2, DeepSpeed), and multi-node training (Torchrun, Ray).
*   **Flexible Dataset Handling:** Load datasets from local storage, Hugging Face, and cloud platforms (S3, Azure, GCP, OCI).
*   **Cloud-Ready:** Provides Docker images and PyPI packages for seamless deployment on cloud platforms and local hardware.

## Latest Updates

*   **(2025/07):**
    *   ND Parallelism support added, enabling Compose Context Parallelism (CP), Tensor Parallelism (TP), and Fully Sharded Data Parallelism (FSDP) within and across nodes.
    *   Expanded model support: GPT-OSS, Gemma 3n, Liquid Foundation Model 2 (LFM2), and Arcee Foundation Models (AFM).
    *   FP8 finetuning with fp8 gather op support via `torchao`.
    *   Integration of Voxtral, Magistral 1.1, and Devstral with mistral-common tokenizer support.
    *   TiledMLP support added for Arctic Long Sequence Training (ALST).
*   **(2025/05):** Quantization Aware Training (QAT) support added.
*   **(2025/03):** Sequence Parallelism (SP) support implemented for scaling context length during fine-tuning.

<details>
<summary>Expand older updates</summary>

*   **(2025/06):** Magistral with mistral-common tokenizer support added.
*   **(2025/04):** Llama 4 support added.
*   **(2025/03):** (Beta) Fine-tuning Multimodal models is now supported.
*   **(2025/02):** LoRA optimizations added for reduced memory usage and improved training speed.  GRPO support added.
*   **(2025/01):** Reward Modelling / Process Reward Modelling fine-tuning support added.
</details>


## Quick Start Guide

### Requirements

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

### Fine-tuning Example

```bash
# Fetch axolotl examples
axolotl fetch examples

# Or, specify a custom path
axolotl fetch examples --dest path/to/folder

# Train a model using LoRA
axolotl train examples/llama-3/lora-1b.yml
```

For more detailed instructions, refer to the [Getting Started Guide](https://docs.axolotl.ai/docs/getting-started.html).

## Documentation

*   [Installation Options](https://docs.axolotl.ai/docs/installation.html) - Detailed setup instructions.
*   [Configuration Guide](https://docs.axolotl.ai/docs/config-reference.html) - Full configuration options.
*   [Dataset Loading](https://docs.axolotl.ai/docs/dataset_loading.html) - Loading datasets.
*   [Dataset Guide](https://docs.axolotl.ai/docs/dataset-formats/) - Supported dataset formats.
*   [Multi-GPU Training](https://docs.axolotl.ai/docs/multi-gpu.html)
*   [Multi-Node Training](https://docs.axolotl.ai/docs/multi-node.html)
*   [Multipacking](https://docs.axolotl.ai/docs/multipack.html)
*   [API Reference](https://docs.axolotl.ai/docs/api/) - Auto-generated documentation.
*   [FAQ](https://docs.axolotl.ai/docs/faq.html) - Frequently asked questions.

## Get Support

*   Join our [Discord community](https://discord.gg/HhrNrHJPRb).
*   Explore our [Examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/).
*   Consult the [Debugging Guide](https://docs.axolotl.ai/docs/debugging.html).
*   For dedicated support, contact [✉️wing@axolotl.ai](mailto:wing@axolotl.ai).

## Contribute

Contributions are highly encouraged!  Please see our [Contributing Guide](https://github.com/axolotl-ai-cloud/axolotl/blob/main/.github/CONTRIBUTING.md) for details.

## Sponsors

For sponsorship inquiries, please contact [wing@axolotl.ai](mailto:wing@axolotl.ai).

## Citation

```bibtex
@software{axolotl,
  title = {Axolotl: Open Source LLM Post-Training},
  author = {{Axolotl maintainers and contributors}},
  url = {https://github.com/axolotl-ai-cloud/axolotl},
  license = {Apache-2.0},
  year = {2023}
}