<p align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_white.svg">
        <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_black.svg">
        <img alt="Axolotl" src="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_black.svg" width="400" height="104" style="max-width: 100%;">
    </picture>
</p>

## Axolotl: Supercharge Your AI Model Training

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

**Axolotl simplifies and accelerates the post-training process for a wide range of AI models, empowering you to fine-tune, optimize, and deploy your models with ease.**

## Key Features

*   **Broad Model Support:** Fine-tune models like LLaMA, Mistral, Mixtral, Pythia, and other Hugging Face transformer-based causal language models.
*   **Comprehensive Training Methods:** Supports full fine-tuning, LoRA, QLoRA, GPTQ, QAT, Preference Tuning (DPO, IPO, KTO, ORPO), RL (GRPO), Multimodal, and Reward Modelling (RM) / Process Reward Modelling (PRM).
*   **Simplified Configuration:** Use a single YAML file to manage all stages, from dataset preprocessing to training, evaluation, quantization, and inference.
*   **Performance Optimizations:**
    *   Multipacking
    *   Flash Attention
    *   Xformers
    *   Flex Attention
    *   Liger Kernel
    *   Cut Cross Entropy
    *   Sequence Parallelism (SP)
    *   LoRA Optimizations
    *   Multi-GPU Training (FSDP1, FSDP2, DeepSpeed)
    *   Multi-Node Training (Torchrun, Ray)
*   **Flexible Data Handling:** Load datasets from local files, Hugging Face Hub, and cloud storage (S3, Azure, GCP, OCI).
*   **Cloud-Ready Deployment:** Ready to deploy with pre-built Docker images and PyPI packages.

## Latest Updates

*   **[2024/07]**
    *   ND Parallelism: Compose Context Parallelism (CP), Tensor Parallelism (TP), and Fully Sharded Data Parallelism (FSDP) within a single node and across multiple nodes. [Blog Post](https://huggingface.co/blog/accelerate-nd-parallel)
    *   Added Model Support: GPT-OSS, Gemma 3n, Liquid Foundation Model 2 (LFM2), and Arcee Foundation Models (AFM).
    *   FP8 Finetuning: Implemented with fp8 gather op via `torchao`. [Get Started](https://docs.axolotl.ai/docs/mixed_precision.html#sec-fp8)
    *   Integrated Models: Voxtral, Magistral 1.1, and Devstral with mistral-common tokenizer support.
    *   ALST Support: TiledMLP support for single-GPU to multi-GPU training with DDP, DeepSpeed and FSDP.  [Examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/alst)
*   **[2024/05]** Quantization Aware Training (QAT) support. [Docs](https://docs.axolotl.ai/docs/qat.html)
*   **[2024/03]** Sequence Parallelism (SP) support. [Blog](https://huggingface.co/blog/axolotl-ai-co/long-context-with-sequence-parallelism-in-axolotl) and [Docs](https://docs.axolotl.ai/docs/sequence_parallelism.html)

<details>

<summary>Expand older updates</summary>

*   **[2024/06]** Magistral with mistral-common tokenizer support.  [Examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/magistral)
*   **[2024/04]** Llama 4 support. [Examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/llama-4)
*   **[2024/03] (Beta)** Fine-tuning Multimodal models supported. [Docs](https://docs.axolotl.ai/docs/multimodal.html)
*   **[2024/02]** LoRA optimizations added. [Docs](https://docs.axolotl.ai/docs/lora_optims.html)
*   **[2024/02]** GRPO support. [Blog](https://huggingface.co/blog/axolotl-ai-co/training-llms-w-interpreter-feedback-wasm) and [GRPO example](https://github.com/axolotl-ai-cloud/grpo_code)
*   **[2024/01]** Reward Modelling / Process Reward Modelling fine-tuning support. [Docs](https://docs.axolotl.ai/docs/reward_modelling.html)

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

Other installation approaches are described [here](https://docs.axolotl.ai/docs/installation.html).

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

For a more detailed guide, consult our [Getting Started Guide](https://docs.axolotl.ai/docs/getting-started.html).

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
*   Consult our [Debugging Guide](https://docs.axolotl.ai/docs/debugging.html)
*   For dedicated support, contact [✉️wing@axolotl.ai](mailto:wing@axolotl.ai)

## Contributing

We welcome contributions!  Refer to our [Contributing Guide](https://github.com/axolotl-ai-cloud/axolotl/blob/main/.github/CONTRIBUTING.md) for details.

## Sponsors

Interested in sponsoring? Please reach out to [wing@axolotl.ai](mailto:wing@axolotl.ai).

## Citing Axolotl

If you use Axolotl in your work, please cite it using the following BibTeX entry:

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

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file.

[Back to Top](#axolotl-supercharge-your-ai-model-training)