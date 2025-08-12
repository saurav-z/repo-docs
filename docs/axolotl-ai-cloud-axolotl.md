<p align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_white.svg">
        <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_black.svg">
        <img alt="Axolotl" src="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_black.svg" width="400" height="104" style="max-width: 100%;">
    </picture>
</p>

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

## Axolotl: Supercharge Your AI Model Post-Training 🚀

Axolotl is your go-to tool for streamlining and optimizing the fine-tuning of large language models (LLMs) and other AI models.  Check out the original repo: [https://github.com/axolotl-ai-cloud/axolotl](https://github.com/axolotl-ai-cloud/axolotl).

## Key Features

*   **Model Support**: Extensive support for various LLMs, including LLaMA, Mistral, Mixtral, Pythia, and Hugging Face Transformers models.
*   **Versatile Training Methods**:
    *   Full fine-tuning
    *   LoRA, QLoRA, GPTQ, QAT
    *   Preference Tuning (DPO, IPO, KTO, ORPO)
    *   Reinforcement Learning (GRPO)
    *   Multimodal fine-tuning
    *   Reward Modeling (RM) / Process Reward Modelling (PRM).
*   **Simplified Configuration**: Use a single YAML file to manage dataset preprocessing, training, evaluation, quantization, and inference.
*   **Performance Optimization**:
    *   Multipacking
    *   Flash Attention
    *   Xformers
    *   Flex Attention
    *   Liger Kernel
    *   Cut Cross Entropy
    *   Sequence Parallelism (SP)
    *   LoRA optimizations
    *   Multi-GPU training (FSDP1, FSDP2, DeepSpeed)
    *   Multi-node training (Torchrun, Ray)
*   **Flexible Data Handling**: Load datasets from local, Hugging Face, and cloud storage (S3, Azure, GCP, OCI).
*   **Cloud-Ready**: Docker images and PyPI packages are available for easy deployment on cloud platforms and local hardware.

## 🎉 Latest Updates

*   **ND Parallelism**: Support for Compose Context Parallelism (CP), Tensor Parallelism (TP), and Fully Sharded Data Parallelism (FSDP) for multi-GPU and multi-node training.  Read the [blog post](https://huggingface.co/blog/accelerate-nd-parallel) for more information.
*   **Expanded Model Support**: Includes GPT-OSS, Gemma 3n, Liquid Foundation Model 2 (LFM2), and Arcee Foundation Models (AFM).
*   **FP8 Finetuning**:  FP8 finetuning with fp8 gather op now possible via `torchao`. Get started [here](https://docs.axolotl.ai/docs/mixed_precision.html#sec-fp8)!
*   **New Model Integrations**: Voxtral, Magistral 1.1, and Devstral with mistral-common tokenizer support.
*   **ALST Support**:  TiledMLP support for Arctic Long Sequence Training (ALST). See [examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/alst) for using ALST with Axolotl!
*   **QAT Support**:  Quantization Aware Training (QAT). Explore the [docs](https://docs.axolotl.ai/docs/qat.html) to learn more!
*   **Sequence Parallelism (SP)**:  Improved support for scaling context lengths. Learn how to scale your context length when fine-tuning: Read the [blog](https://huggingface.co/blog/axolotl-ai-co/long-context-with-sequence-parallelism-in-axolotl) and [docs](https://docs.axolotl.ai/docs/sequence_parallelism.html).

<details>
<summary>Expand Older Updates</summary>

*   **Magistral Integration**:  Magistral support with mistral-common tokenizer. See [examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/magistral) to train your models!
*   **Llama 4 Support**: Support for Llama 4 with examples at [examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/llama-4).
*   **Multimodal Fine-tuning (Beta)**:  Support for fine-tuning Multimodal models. [Docs](https://docs.axolotl.ai/docs/multimodal.html).
*   **LoRA Optimizations**:  Memory usage and speed improvements for LoRA and QLoRA. [Docs](https://docs.axolotl.ai/docs/lora_optims.html).
*   **GRPO Support**:  GRPO fine-tuning support. [Blog](https://huggingface.co/blog/axolotl-ai-co/training-llms-w-interpreter-feedback-wasm) and [GRPO example](https://github.com/axolotl-ai-cloud/grpo_code).
*   **Reward Modeling/PRM**:  Reward Modelling and Process Reward Modelling fine-tuning support.  See [docs](https://docs.axolotl.ai/docs/reward_modelling.html).

</details>

## 🚀 Quick Start

### Requirements

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

*   [RunPod](https://runpod.io/gsc?template=v2ickqhz9s&ref=6i7fkpdz)
*   [Vast.ai](https://cloud.vast.ai?ref_id=62897&template_id=bdd4a49fa8bce926defc99471864cace&utm_source=github&utm_medium=developer_community&utm_campaign=template_launch_axolotl&utm_content=readme)
*   [PRIME Intellect](https://app.primeintellect.ai/dashboard/create-cluster?image=axolotl&location=Cheapest&security=Cheapest&show_spot=true)
*   [Modal](https://www.modal.com?utm_source=github&utm_medium=github&utm_campaign=axolotl)
*   [Novita](https://novita.ai/gpus-console?templateId=311)
*   [JarvisLabs.ai](https://jarvislabs.ai/templates/axolotl)
*   [Latitude.sh](https://latitude.sh/blueprint/989e0e79-3bf6-41ea-a46b-1f246e309d5c)

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

Refer to the [Getting Started Guide](https://docs.axolotl.ai/docs/getting-started.html) for more detailed instructions.

## 📚 Documentation

*   [Installation Options](https://docs.axolotl.ai/docs/installation.html) - Detailed setup instructions
*   [Configuration Guide](https://docs.axolotl.ai/docs/config-reference.html) - Configuration options and examples
*   [Dataset Loading](https://docs.axolotl.ai/docs/dataset_loading.html) - Loading datasets from various sources
*   [Dataset Guide](https://docs.axolotl.ai/docs/dataset-formats/) - Supported formats
*   [Multi-GPU Training](https://docs.axolotl.ai/docs/multi-gpu.html)
*   [Multi-Node Training](https://docs.axolotl.ai/docs/multi-node.html)
*   [Multipacking](https://docs.axolotl.ai/docs/multipack.html)
*   [API Reference](https://docs.axolotl.ai/docs/api/) - Auto-generated code documentation
*   [FAQ](https://docs.axolotl.ai/docs/faq.html)

## 🤝 Getting Help

*   Join our [Discord community](https://discord.gg/HhrNrHJPRb)
*   Explore our [Examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/)
*   Read our [Debugging Guide](https://docs.axolotl.ai/docs/debugging.html)
*   Dedicated Support: Contact [✉️wing@axolotl.ai](mailto:wing@axolotl.ai)

## 🌟 Contributing

We welcome contributions!  See our [Contributing Guide](https://github.com/axolotl-ai-cloud/axolotl/blob/main/.github/CONTRIBUTING.md).

## ❤️ Sponsors

Contact us at [wing@axolotl.ai](mailto:wing@axolotl.ai) to become a sponsor.

## 📝 Citing Axolotl

```bibtex
@software{axolotl,
  title = {Axolotl: Post-Training for AI Models},
  author = {{Axolotl maintainers and contributors}},
  url = {https://github.com/axolotl-ai-cloud/axolotl},
  license = {Apache-2.0},
  year = {2023}
}
```

## 📜 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.