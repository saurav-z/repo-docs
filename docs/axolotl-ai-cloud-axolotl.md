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

## Axolotl: Supercharge Your AI Model Post-Training

Axolotl is a powerful tool that simplifies and accelerates the post-training process for a wide range of AI models. [Explore the Axolotl repository](https://github.com/axolotl-ai-cloud/axolotl).

## Key Features

*   **Versatile Model Support**: Train popular models like LLaMA, Mistral, Mixtral, Pythia, and others compatible with Hugging Face transformers.
*   **Comprehensive Training Methods**: Supports full fine-tuning, LoRA, QLoRA, GPTQ, QAT, Preference Tuning (DPO, IPO, KTO, ORPO), RL (GRPO), Multimodal, and Reward Modelling (RM) / Process Reward Modelling (PRM).
*   **Simplified Configuration**:  Manage your entire workflow with a single YAML configuration file for dataset preparation, training, evaluation, quantization, and inference.
*   **Performance Optimization**:  Leverage advanced techniques like Multipacking, Flash Attention, Xformers, Flex Attention, Liger Kernel, Cut Cross Entropy, Sequence Parallelism (SP), LoRA optimizations, and multi-GPU/multi-node training for faster and more efficient training.
*   **Flexible Data Handling**: Load data from local files, Hugging Face Hub, and cloud storage (S3, Azure, GCP, OCI).
*   **Cloud-Ready Deployment**: Utilize pre-built Docker images and PyPI packages for easy deployment on cloud platforms and local hardware.

## 🎉 Latest Updates

*   **ND Parallelism**: Added support for Compose Context Parallelism (CP), Tensor Parallelism (TP), and Fully Sharded Data Parallelism (FSDP) for single and multi-node setups. ([Blog Post](https://huggingface.co/blog/accelerate-nd-parallel))
*   **Expanded Model Support**:  Now includes GPT-OSS, Gemma 3n, Liquid Foundation Model 2 (LFM2), and Arcee Foundation Models (AFM).
*   **FP8 Finetuning**: Enables FP8 fine-tuning with fp8 gather op using `torchao`. ([Docs](https://docs.axolotl.ai/docs/mixed_precision.html#sec-fp8))
*   **New Model Integrations**:  Supports Voxtral, Magistral 1.1, and Devstral with mistral-common tokenizer.
*   **ALST Support**:  Includes TiledMLP support for Arctic Long Sequence Training (ALST) across multiple GPUs. ([Examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/alst))
*   **Quantization Aware Training (QAT)**: Integrated QAT support for model optimization. ([Docs](https://docs.axolotl.ai/docs/qat.html))
*   **Sequence Parallelism (SP)**:  Implemented Sequence Parallelism for extended context length finetuning. ([Blog](https://huggingface.co/blog/axolotl-ai-co/long-context-with-sequence-parallelism-in-axolotl), [Docs](https://docs.axolotl.ai/docs/sequence_parallelism.html))
*   **Magistral with Mistral-Common Tokenizer**: added support for Magistral models. ([Examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/magistral))
*   **Llama 4 support**: Added Llama 4 support with linearized version. ([Examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/llama-4))
*   **Multimodal Fine-tuning (Beta)**: Support for fine-tuning multimodal models. ([Docs](https://docs.axolotl.ai/docs/multimodal.html))
*   **LoRA Optimizations**: Increased LoRA/QLoRA training speed and reduced memory usage. ([Docs](https://docs.axolotl.ai/docs/lora_optims.html))
*   **GRPO Support**: Added GRPO support for fine-tuning with interpreter feedback. ([Blog](https://huggingface.co/blog/axolotl-ai-co/training-llms-w-interpreter-feedback-wasm), [GRPO Example](https://github.com/axolotl-ai-cloud/grpo_code))
*   **Reward Modeling / Process Reward Modeling**: added fine-tuning support. ([Docs](https://docs.axolotl.ai/docs/reward_modelling.html)).

## 🚀 Quick Start

**Requirements**:

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

That's it! Check out our [Getting Started Guide](https://docs.axolotl.ai/docs/getting-started.html) for a more detailed walkthrough.

## 📚 Documentation

*   [Installation Options](https://docs.axolotl.ai/docs/installation.html) - Detailed setup instructions for different environments
*   [Configuration Guide](https://docs.axolotl.ai/docs/config-reference.html) - Full configuration options and examples
*   [Dataset Loading](https://docs.axolotl.ai/docs/dataset_loading.html) - Loading datasets from various sources
*   [Dataset Guide](https://docs.axolotl.ai/docs/dataset-formats/) - Supported formats and how to use them
*   [Multi-GPU Training](https://docs.axolotl.ai/docs/multi-gpu.html)
*   [Multi-Node Training](https://docs.axolotl.ai/docs/multi-node.html)
*   [Multipacking](https://docs.axolotl.ai/docs/multipack.html)
*   [API Reference](https://docs.axolotl.ai/docs/api/) - Auto-generated code documentation
*   [FAQ](https://docs.axolotl.ai/docs/faq.html) - Frequently asked questions

## 🤝 Getting Help

*   Join our [Discord community](https://discord.gg/HhrNrHJPRb) for support
*   Check out our [Examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/) directory
*   Read our [Debugging Guide](https://docs.axolotl.ai/docs/debugging.html)
*   Need dedicated support? Please contact [✉️wing@axolotl.ai](mailto:wing@axolotl.ai) for options

## 🌟 Contributing

Contributions are welcome! Please see our [Contributing Guide](https://github.com/axolotl-ai-cloud/axolotl/blob/main/.github/CONTRIBUTING.md) for details.

## ❤️ Sponsors

Interested in sponsoring? Contact us at [wing@axolotl.ai](mailto:wing@axolotl.ai)

## 📝 Citing Axolotl

If you use Axolotl in your research or projects, please cite it as follows:

```bibtex
@software{axolotl,
  title = {Axolotl: Post-Training for AI Models},
  author = {{Axolotl maintainers and contributors}},
  url = {https://github.com/axolotl-ai-cloud/axolotl},
  license = {Apache-2.0},
  year = {2023}
}
```