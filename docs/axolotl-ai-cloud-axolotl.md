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

## Axolotl: The Ultimate Toolkit for Fine-Tuning AI Models

Axolotl simplifies and accelerates the process of fine-tuning a wide range of AI models, providing optimized performance and a user-friendly experience.  Explore the [Axolotl GitHub repository](https://github.com/axolotl-ai-cloud/axolotl) to get started.

**Key Features:**

*   **Comprehensive Model Support:** Train Llama, Mistral, Mixtral, Pythia, and other Hugging Face transformer models.
*   **Versatile Training Methods:** Supports full fine-tuning, LoRA, QLoRA, GPTQ, QAT, Preference Tuning (DPO, IPO, KTO, ORPO), RL (GRPO), Multimodal, and Reward Modelling (RM) / Process Reward Modelling (PRM).
*   **Simplified Configuration:** Utilize a single YAML file for dataset preprocessing, training, evaluation, quantization, and inference.
*   **Performance Enhancements:** Leverage [Multipacking](https://docs.axolotl.ai/docs/multipack.html), [Flash Attention](https://github.com/Dao-AILab/flash-attention), [Xformers](https://github.com/facebookresearch/xformers), [Flex Attention](https://pytorch.org/blog/flexattention/), [Liger Kernel](https://github.com/linkedin/Liger-Kernel), [Cut Cross Entropy](https://github.com/apple/ml-cross-entropy/tree/main), [Sequence Parallelism (SP)](https://docs.axolotl.ai/docs/sequence_parallelism.html), [LoRA optimizations](https://docs.axolotl.ai/docs/lora_optims.html), [Multi-GPU training (FSDP1, FSDP2, DeepSpeed)](https://docs.axolotl.ai/docs/multi-gpu.html), [Multi-node training (Torchrun, Ray)](https://docs.axolotl.ai/docs/multi-node.html), and more for optimal performance.
*   **Flexible Data Handling:** Load datasets from local files, Hugging Face Hub, and cloud storage (S3, Azure, GCP, OCI).
*   **Cloud-Ready Deployment:**  Utilize pre-built [Docker images](https://hub.docker.com/u/axolotlai) and [PyPI packages](https://pypi.org/project/axolotl/) for seamless cloud deployment.

## Latest Updates
-   **Magistral Support:** Added support for Magistral models with Mistral-common tokenizer (June 2024). See [examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/magistral)
-   **Quantization Aware Training (QAT):** Added QAT support (May 2024). Explore the [docs](https://docs.axolotl.ai/docs/qat.html) to learn more!
-   **Llama 4 Support:** Added Llama 4 support (April 2024). See [examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/llama-4)
-   **Sequence Parallelism:** Implemented Sequence Parallelism (SP) for scaling context length. Read the [blog](https://huggingface.co/blog/axolotl-ai-co/long-context-with-sequence-parallelism-in-axolotl) and [docs](https://docs.axolotl.ai/docs/sequence_parallelism.html)
-   **(Beta) Multimodal Support:** Fine-tune multimodal models (March 2024). Check out the [docs](https://docs.axolotl.ai/docs/multimodal.html)
-   **LoRA Optimizations:** Improved memory usage and training speed for LoRA and QLoRA (February 2024). Jump into the [docs](https://docs.axolotl.ai/docs/lora_optims.html)
-   **GRPO Support:** Added GRPO support (February 2024).  See [blog](https://huggingface.co/blog/axolotl-ai-co/training-llms-w-interpreter-feedback-wasm) and [GRPO example](https://github.com/axolotl-ai-cloud/grpo_code)
-   **Reward Modelling Support:** Added Reward Modelling / Process Reward Modelling fine-tuning (January 2024). See [docs](https://docs.axolotl.ai/docs/reward_modelling.html).

## 🚀 Quick Start

### Requirements
- NVIDIA GPU (Ampere or newer for `bf16` and Flash Attention) or AMD GPU
- Python 3.11
- PyTorch ≥2.6.0

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

### Your First Fine-tune

```bash
# Fetch axolotl examples
axolotl fetch examples

# Or, specify a custom path
axolotl fetch examples --dest path/to/folder

# Train a model using LoRA
axolotl train examples/llama-3/lora-1b.yml
```

Check out our [Getting Started Guide](https://docs.axolotl.ai/docs/getting-started.html) for a more detailed walkthrough.

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
*   Need dedicated support? Please contact [✉️wing@axolotl.ai](mailto:wing@axolotl.ai)

## 🌟 Contributing

Contributions are welcome! Please see our [Contributing Guide](https://github.com/axolotl-ai-cloud/axolotl/blob/main/.github/CONTRIBUTING.md) for details.

## ❤️ Sponsors

Thank you to our sponsors who help make Axolotl possible:

-   [Modal](https://www.modal.com?utm_source=github&utm_medium=github&utm_campaign=axolotl) - Modal lets you run
    jobs in the cloud, by just writing a few lines of Python. Customers use Modal to deploy Gen AI models at large scale,
    fine-tune large language models, run protein folding simulations, and much more.

Interested in sponsoring? Contact us at [wing@axolotl.ai](mailto:wing@axolotl.ai)

## 📜 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.