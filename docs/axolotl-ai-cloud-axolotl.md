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

## Axolotl: Train and Fine-tune LLMs with Ease

Axolotl is a powerful and versatile tool designed to simplify and accelerate the post-training process for various AI models.  [Explore the Axolotl repository](https://github.com/axolotl-ai-cloud/axolotl) for the latest updates and features!

### Key Features

*   **Broad Model Support**: Train a wide range of models, including LLaMA, Mistral, Mixtral, Pythia, and other HuggingFace causal language models.
*   **Diverse Training Methods**: Supports full fine-tuning, LoRA, QLoRA, GPTQ, QAT, Preference Tuning (DPO, IPO, KTO, ORPO), RL (GRPO), Multimodal, and Reward Modelling (RM) / Process Reward Modelling (PRM).
*   **Simplified Configuration**: Utilize a single YAML file for dataset preprocessing, training, evaluation, quantization, and inference.
*   **Performance Optimizations**:
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
*   **Flexible Data Handling**: Load datasets from local files, Hugging Face Hub, and cloud storage (S3, Azure, GCP, OCI).
*   **Cloud-Ready Deployment**: Ready-to-use Docker images and PyPI packages for seamless deployment on cloud platforms and local hardware.

### Latest Updates

*   **2025/06:** Magistral with mistral-common tokenizer support has been added to Axolotl. See [examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/magistral)
*   **2025/05:** Quantization Aware Training (QAT) support. Explore the [docs](https://docs.axolotl.ai/docs/qat.html) to learn more.
*   **2025/04:** Llama 4 support. See [examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/llama-4)
*   **2025/03:** Sequence Parallelism (SP) support. Read the [blog](https://huggingface.co/blog/axolotl-ai-co/long-context-with-sequence-parallelism-in-axolotl) and [docs](https://docs.axolotl.ai/docs/sequence_parallelism.html)
*   **2025/03:** (Beta) Fine-tuning Multimodal models. Check out the [docs](https://docs.axolotl.ai/docs/multimodal.html)
*   **2025/02:** LoRA optimizations. [docs](https://docs.axolotl.ai/docs/lora_optims.html)
*   **2025/02:** GRPO support. [blog](https://huggingface.co/blog/axolotl-ai-co/training-llms-w-interpreter-feedback-wasm) and [GRPO example](https://github.com/axolotl-ai-cloud/grpo_code)
*   **2025/01:** Reward Modelling / Process Reward Modelling fine-tuning support. See [docs](https://docs.axolotl.ai/docs/reward_modelling.html).

### Quick Start

**Requirements**:

*   NVIDIA GPU (Ampere or newer for `bf16` and Flash Attention) or AMD GPU
*   Python 3.11
*   PyTorch ≥2.6.0

#### Installation

##### Using pip

```bash
pip3 install -U packaging==23.2 setuptools==75.8.0 wheel ninja
pip3 install --no-build-isolation axolotl[flash-attn,deepspeed]

# Download example axolotl configs, deepspeed configs
axolotl fetch examples
axolotl fetch deepspeed_configs  # OPTIONAL
```

##### Using Docker

```bash
docker run --gpus '"all"' --rm -it axolotlai/axolotl:main-latest
```

Other installation approaches are described [here](https://docs.axolotl.ai/docs/installation.html).

#### Your First Fine-tune

```bash
# Fetch axolotl examples
axolotl fetch examples

# Or, specify a custom path
axolotl fetch examples --dest path/to/folder

# Train a model using LoRA
axolotl train examples/llama-3/lora-1b.yml
```

Check out our [Getting Started Guide](https://docs.axolotl.ai/docs/getting-started.html) for a more detailed walkthrough.

### Documentation

*   [Installation Options](https://docs.axolotl.ai/docs/installation.html) - Detailed setup instructions
*   [Configuration Guide](https://docs.axolotl.ai/docs/config-reference.html) - Full configuration options and examples
*   [Dataset Loading](https://docs.axolotl.ai/docs/dataset_loading.html) - Loading datasets from various sources
*   [Dataset Guide](https://docs.axolotl.ai/docs/dataset-formats/) - Supported formats
*   [Multi-GPU Training](https://docs.axolotl.ai/docs/multi-gpu.html)
*   [Multi-Node Training](https://docs.axolotl.ai/docs/multi-node.html)
*   [Multipacking](https://docs.axolotl.ai/docs/multipack.html)
*   [API Reference](https://docs.axolotl.ai/docs/api/) - Auto-generated code documentation
*   [FAQ](https://docs.axolotl.ai/docs/faq.html) - Frequently asked questions

### Getting Help

*   Join our [Discord community](https://discord.gg/HhrNrHJPRb)
*   Check out our [Examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/)
*   Read our [Debugging Guide](https://docs.axolotl.ai/docs/debugging.html)
*   For dedicated support, contact [✉️wing@axolotl.ai](mailto:wing@axolotl.ai)

### Contributing

Contributions are welcome! See our [Contributing Guide](https://github.com/axolotl-ai-cloud/axolotl/blob/main/.github/CONTRIBUTING.md).

### Sponsors

Thank you to our sponsors:

*   [Modal](https://www.modal.com?utm_source=github&utm_medium=github&utm_campaign=axolotl)

Interested in sponsoring? Contact [wing@axolotl.ai](mailto:wing@axolotl.ai)

### License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file.