<p align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_white.svg">
        <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_black.svg">
        <img alt="Axolotl" src="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_black.svg" width="400" height="104" style="max-width: 100%;">
    </picture>
</p>

## Axolotl: Fine-tune Your AI Models with Ease and Efficiency

Axolotl is a powerful and versatile tool designed to simplify and accelerate the post-training process for large language models and other AI architectures. [Explore the Axolotl GitHub repository](https://github.com/axolotl-ai-cloud/axolotl) to dive deeper.

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

## Key Features of Axolotl:

*   **Broad Model Support:** Compatible with a wide range of models, including LLaMA, Mistral, Mixtral, Pythia, and Hugging Face Transformers causal language models.
*   **Diverse Training Methods:** Offers comprehensive training options such as full fine-tuning, LoRA, QLoRA, GPTQ, QAT, Preference Tuning (DPO, IPO, KTO, ORPO), RL (GRPO), Multimodal, and Reward Modeling (RM) / Process Reward Modelling (PRM).
*   **Simplified Configuration:** Uses a single YAML file for dataset preprocessing, training, evaluation, quantization, and inference, streamlining your workflow.
*   **Optimized Performance:** Includes cutting-edge optimizations like Multipacking, Flash Attention, Xformers, Flex Attention, Liger Kernel, Cut Cross Entropy, Sequence Parallelism (SP), LoRA optimizations, and Multi-GPU and Multi-node training capabilities.
*   **Flexible Data Handling:** Supports loading datasets from local storage, Hugging Face repositories, and various cloud platforms (S3, Azure, GCP, OCI).
*   **Cloud-Ready:** Provides Docker images and PyPI packages for easy deployment on cloud platforms and local hardware.

## Latest Updates:

### [Insert most recent updates here, focusing on the benefits of the feature.]

*   **ND Parallelism:** Axolotl now supports Compose Context Parallelism (CP), Tensor Parallelism (TP), and Fully Sharded Data Parallelism (FSDP) within a single node and across multiple nodes, enabling faster and more efficient training of large models. [Learn More](https://huggingface.co/blog/accelerate-nd-parallel)
*   **New Model Support:** Extended model support with GPT-OSS, Gemma 3n, Liquid Foundation Model 2 (LFM2), and Arcee Foundation Models (AFM) to provide more flexibility for your projects.
*   **FP8 Finetuning:** Introducing FP8 finetuning with fp8 gather op via `torchao` for efficient mixed-precision training, offering significant performance gains. [Get Started](https://docs.axolotl.ai/docs/mixed_precision.html#sec-fp8)
*   **New Models:** Added Voxtral, Magistral 1.1, and Devstral models with mistral-common tokenizer support for better model compatibility.
*   **ALST Training:** TiledMLP support has been added for single-GPU to multi-GPU training, along with DeepSpeed and FSDP support to support Arctic Long Sequence Training (ALST).

<details>

<summary>Older Updates</summary>

*   **Magistral:** Support for Magistral with mistral-common tokenizer to fine-tune with a broader set of models. [Examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/magistral)
*   **Llama 4 Support:** Full Llama 4 support with Axolotl.
*   **Multimodal Training (Beta):** Support for fine-tuning Multimodal models. [Docs](https://docs.axolotl.ai/docs/multimodal.html)
*   **LoRA Optimizations:** Memory usage and speed improvements for LoRA and QLoRA in single and multi-GPU training with LoRA optimizations. [Docs](https://docs.axolotl.ai/docs/lora_optims.html)
*   **GRPO Support:** Support for GRPO. [Blog](https://huggingface.co/blog/axolotl-ai-co/training-llms-w-interpreter-feedback-wasm)
*   **Reward Modelling Support:** Fine-tuning for Reward Modelling / Process Reward Modelling. [Docs](https://docs.axolotl.ai/docs/reward_modelling.html)
</details>

## Quick Start Guide

### Requirements

*   NVIDIA GPU (Ampere or newer for `bf16` and Flash Attention) or AMD GPU
*   Python 3.11
*   PyTorch ≥2.6.0

### Installation

**Using pip:**

```bash
pip3 install -U packaging==23.2 setuptools==75.8.0 wheel ninja
pip3 install --no-build-isolation axolotl[flash-attn,deepspeed]

# Download example axolotl configs, deepspeed configs
axolotl fetch examples
axolotl fetch deepspeed_configs  # OPTIONAL
```

**Using Docker:**

```bash
docker run --gpus '"all"' --rm -it axolotlai/axolotl:main-latest
```

[Other installation approaches are described here](https://docs.axolotl.ai/docs/installation.html).

### Your First Fine-tune

```bash
# Fetch axolotl examples
axolotl fetch examples

# Or, specify a custom path
axolotl fetch examples --dest path/to/folder

# Train a model using LoRA
axolotl train examples/llama-3/lora-1b.yml
```

For a more detailed walkthrough, consult the [Getting Started Guide](https://docs.axolotl.ai/docs/getting-started.html).

## Documentation

*   [Installation Options](https://docs.axolotl.ai/docs/installation.html)
*   [Configuration Guide](https://docs.axolotl.ai/docs/config-reference.html)
*   [Dataset Loading](https://docs.axolotl.ai/docs/dataset_loading.html)
*   [Dataset Guide](https://docs.axolotl.ai/docs/dataset-formats/)
*   [Multi-GPU Training](https://docs.axolotl.ai/docs/multi-gpu.html)
*   [Multi-Node Training](https://docs.axolotl.ai/docs/multi-node.html)
*   [Multipacking](https://docs.axolotl.ai/docs/multipack.html)
*   [API Reference](https://docs.axolotl.ai/docs/api/)
*   [FAQ](https://docs.axolotl.ai/docs/faq.html)

## Getting Help

*   Join our [Discord community](https://discord.gg/HhrNrHJPRb)
*   Explore the [Examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/) directory
*   Refer to our [Debugging Guide](https://docs.axolotl.ai/docs/debugging.html)
*   For dedicated support, contact: [✉️wing@axolotl.ai](mailto:wing@axolotl.ai)

## Contributing

We welcome contributions! Review the [Contributing Guide](https://github.com/axolotl-ai-cloud/axolotl/blob/main/.github/CONTRIBUTING.md) for details.

## Sponsors

For sponsorship opportunities, contact [wing@axolotl.ai](mailto:wing@axolotl.ai).

## Citing Axolotl

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

This project is licensed under the Apache 2.0 License (see the [LICENSE](LICENSE) file).