<p align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_white.svg">
        <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_black.svg">
        <img alt="Axolotl" src="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_black.svg" width="400" height="104" style="max-width: 100%;">
    </picture>
</p>

<!-- Badges -->
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
    <br/>
    <img src="https://github.com/axolotl-ai-cloud/axolotl/actions/workflows/tests-nightly.yml/badge.svg" alt="tests-nightly">
    <img src="https://github.com/axolotl-ai-cloud/axolotl/actions/workflows/multi-gpu-e2e.yml/badge.svg" alt="multigpu-semi-weekly tests">
</p>

## Axolotl: Fine-tune and Optimize Your AI Models with Ease

Axolotl is a powerful toolkit designed to simplify and accelerate the post-training process for various AI models, offering a streamlined workflow for fine-tuning and optimization.  [Explore the Axolotl project on GitHub](https://github.com/axolotl-ai-cloud/axolotl).

## Key Features

*   **Broad Model Support:** Train a wide range of models, including LLaMA, Mistral, Mixtral, Pythia, and other Hugging Face transformer models.
*   **Versatile Training Methods:** Utilize full fine-tuning, LoRA, QLoRA, GPTQ, QAT, Preference Tuning (DPO, IPO, KTO, ORPO), RL (GRPO), Multimodal, and Reward Modelling (RM) / Process Reward Modelling (PRM).
*   **Simplified Configuration:**  Manage your entire workflow with a single YAML configuration file, streamlining dataset preprocessing, training, evaluation, quantization, and inference.
*   **Performance Enhancements:** Leverage a suite of optimizations including Multipacking, Flash Attention, Xformers, Flex Attention, Liger Kernel, Cut Cross Entropy, Sequence Parallelism (SP), LoRA optimizations, Multi-GPU training (FSDP1, FSDP2, DeepSpeed), and Multi-node training (Torchrun, Ray), and more.
*   **Flexible Data Handling:** Load datasets from local files, Hugging Face Hub, and cloud storage providers (S3, Azure, GCP, OCI).
*   **Cloud-Ready Deployment:** Use Docker images and PyPI packages for seamless deployment on cloud platforms and local hardware.

## What's New

**Recent Updates**

*   **ND Parallelism Support**: Added support for Compose Context Parallelism (CP), Tensor Parallelism (TP), and Fully Sharded Data Parallelism (FSDP) within a single node and across multiple nodes.
*   **New Model Support**:  Includes GPT-OSS, Gemma 3n, Liquid Foundation Model 2 (LFM2), and Arcee Foundation Models (AFM).
*   **FP8 Finetuning**: Now supports FP8 finetuning with fp8 gather op via `torchao`.
*   **New Model Integrations**: Integrates Voxtral, Magistral 1.1, and Devstral with mistral-common tokenizer support.
*   **ALST Support**: Added TiledMLP support for single-GPU to multi-GPU training with DDP, DeepSpeed, and FSDP.
*   **Quantization Aware Training (QAT)**:  Support has been added.
*   **Sequence Parallelism (SP) Support**: Enhanced context length scaling.

<details>
<summary>Older Updates</summary>

*   **Magistral Integration**: Magistral with mistral-common tokenizer support.
*   **Llama 4 Support**: Support for Llama 4 models.
*   **Multimodal Support (Beta)**: Fine-tuning Multimodal models is now supported.
*   **LoRA Optimizations**: Added LoRA optimizations to reduce memory usage and improve training speed.
*   **GRPO Support**: Support for GRPO.
*   **Reward Modelling**: Fine-tuning Reward Modelling (RM) / Process Reward Modelling (PRM) fine-tuning support.

</details>

## Quick Start Guide

### Prerequisites

*   NVIDIA GPU (Ampere or newer for `bf16` and Flash Attention) or AMD GPU
*   Python 3.11
*   PyTorch ≥2.6.0

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
- [RunPod](https://runpod.io/gsc?template=v2ickqhz9s&ref=6i7fkpdz)
- [Vast.ai](https://cloud.vast.ai?ref_id=62897&template_id=bdd4a49fa8bce926defc99471864cace&utm_source=github&utm_medium=developer_community&utm_campaign=template_launch_axolotl&utm_content=readme)
- [PRIME Intellect](https://app.primeintellect.ai/dashboard/create-cluster?image=axolotl&location=Cheapest&security=Cheapest&show_spot=true)
- [Modal](https://www.modal.com?utm_source=github&utm_medium=github&utm_campaign=axolotl)
- [Novita](https://novita.ai/gpus-console?templateId=311)
- [JarvisLabs.ai](https://jarvislabs.ai/templates/axolotl)
- [Latitude.sh](https://latitude.sh/blueprint/989e0e79-3bf6-41ea-a46b-1f246e309d5c)
</details>

### Fine-tuning Your First Model

```bash
# Fetch axolotl examples
axolotl fetch examples

# Or, specify a custom path
axolotl fetch examples --dest path/to/folder

# Train a model using LoRA
axolotl train examples/llama-3/lora-1b.yml
```

For a more in-depth walkthrough, consult the [Getting Started Guide](https://docs.axolotl.ai/docs/getting-started.html).

## Documentation

*   [Installation Options](https://docs.axolotl.ai/docs/installation.html) - Detailed setup instructions for various environments.
*   [Configuration Guide](https://docs.axolotl.ai/docs/config-reference.html) - Comprehensive configuration options and examples.
*   [Dataset Loading](https://docs.axolotl.ai/docs/dataset_loading.html) - Learn how to load datasets from various sources.
*   [Dataset Guide](https://docs.axolotl.ai/docs/dataset-formats/) - Supported formats and how to use them.
*   [Multi-GPU Training](https://docs.axolotl.ai/docs/multi-gpu.html)
*   [Multi-Node Training](https://docs.axolotl.ai/docs/multi-node.html)
*   [Multipacking](https://docs.axolotl.ai/docs/multipack.html)
*   [API Reference](https://docs.axolotl.ai/docs/api/) - Auto-generated code documentation.
*   [FAQ](https://docs.axolotl.ai/docs/faq.html) - Frequently asked questions.

## Get Help

*   Join the [Discord community](https://discord.gg/HhrNrHJPRb).
*   Explore the [Examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/) directory.
*   Consult the [Debugging Guide](https://docs.axolotl.ai/docs/debugging.html).
*   For dedicated support, contact [✉️wing@axolotl.ai](mailto:wing@axolotl.ai).

## Contributing

Contributions are welcome! Please review the [Contributing Guide](https://github.com/axolotl-ai-cloud/axolotl/blob/main/.github/CONTRIBUTING.md) for guidelines.

## Sponsors

Contact [wing@axolotl.ai](mailto:wing@axolotl.ai) for sponsorship inquiries.

## Citation

```bibtex
@software{axolotl,
  title = {Axolotl: Post-Training for AI Models},
  author = {{Axolotl maintainers and contributors}},
  url = {https://github.com/axolotl-ai-cloud/axolotl},
  license = {Apache-2.0},
  year = {2023}
}
```

## License

This project is licensed under the Apache 2.0 License; see the [LICENSE](LICENSE) file.