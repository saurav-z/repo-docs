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

## Axolotl: Fine-tune and optimize your AI models with ease.

Axolotl is a powerful tool designed to streamline the post-training process for various AI models, offering comprehensive support for fine-tuning, optimization, and deployment.  [Explore the Axolotl repository](https://github.com/axolotl-ai-cloud/axolotl) for more details.

**Key Features:**

*   **Broad Model Support**: Train models like LLaMA, Mistral, Mixtral, Pythia, and more, compatible with Hugging Face Transformers causal language models.
*   **Versatile Training Methods**: Utilize full fine-tuning, LoRA, QLoRA, GPTQ, QAT, Preference Tuning (DPO, IPO, KTO, ORPO), RL (GRPO), Multimodal, and Reward Modelling (RM) / Process Reward Modelling (PRM).
*   **Simplified Configuration**: Use a single YAML file for dataset preprocessing, training, evaluation, quantization, and inference, streamlining your workflow.
*   **Performance-Driven Optimizations**: Leverage advanced techniques including Multipacking, Flash Attention, Xformers, Flex Attention, Liger Kernel, Cut Cross Entropy, Sequence Parallelism (SP), LoRA optimizations, Multi-GPU training (FSDP1, FSDP2, DeepSpeed), Multi-node training (Torchrun, Ray) and many more!
*   **Flexible Data Handling**: Load datasets from local storage, Hugging Face Hub, and cloud providers (S3, Azure, GCP, OCI).
*   **Cloud-Ready Deployment**: Deploy seamlessly using provided Docker images and PyPI packages, making it easy to run on cloud platforms and local hardware.
*   **Recent Updates**:
    *   **Voxtral with mistral-common tokenizer support**: Integrated in July 2025. ([docs](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/voxtral))
    *   **TiledMLP support**: Added for single-GPU to multi-GPU training with DDP, DeepSpeed and FSDP support to support Arctic Long Sequence Training (ALST). See [examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/alst).
    *   **Magistral support**: Added with mistral-common tokenizer support (June 2025).  See [examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/magistral).
    *   **Quantization Aware Training (QAT) support**: Added (May 2025). Explore [docs](https://docs.axolotl.ai/docs/qat.html).
    *   **Llama 4 support**: Added (April 2025). See [examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/llama-4).
    *   **Sequence Parallelism (SP) support**: Implemented (March 2025). Read the [blog](https://huggingface.co/blog/axolotl-ai-co/long-context-with-sequence-parallelism-in-axolotl) and [docs](https://docs.axolotl.ai/docs/sequence_parallelism.html).
    *   **(Beta) Multimodal model fine-tuning support**: Now supported (March 2025). Check out the [docs](https://docs.axolotl.ai/docs/multimodal.html).
    *   **LoRA optimizations**: Added to reduce memory usage and improve training speed (February 2025). Jump into the [docs](https://docs.axolotl.ai/docs/lora_optims.html).
    *   **GRPO support**: Added (February 2025). Dive into our [blog](https://huggingface.co/blog/axolotl-ai-co/training-llms-w-interpreter-feedback-wasm) and [GRPO example](https://github.com/axolotl-ai-cloud/grpo_code).
    *   **Reward Modelling / Process Reward Modelling fine-tuning support**: Added (January 2025). See [docs](https://docs.axolotl.ai/docs/reward_modelling.html).

## Getting Started

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

For other installation approaches, see the [installation guide](https://docs.axolotl.ai/docs/installation.html).

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

### Fine-tuning Your First Model

```bash
# Fetch axolotl examples
axolotl fetch examples

# Or, specify a custom path
axolotl fetch examples --dest path/to/folder

# Train a model using LoRA
axolotl train examples/llama-3/lora-1b.yml
```

Refer to the [Getting Started Guide](https://docs.axolotl.ai/docs/getting-started.html) for a detailed walkthrough.

## Documentation

*   [Installation Options](https://docs.axolotl.ai/docs/installation.html) - Installation instructions for different environments
*   [Configuration Guide](https://docs.axolotl.ai/docs/config-reference.html) - Configuration options and examples
*   [Dataset Loading](https://docs.axolotl.ai/docs/dataset_loading.html) - Loading datasets from various sources
*   [Dataset Guide](https://docs.axolotl.ai/docs/dataset-formats/) - Supported formats and usage
*   [Multi-GPU Training](https://docs.axolotl.ai/docs/multi-gpu.html)
*   [Multi-Node Training](https://docs.axolotl.ai/docs/multi-node.html)
*   [Multipacking](https://docs.axolotl.ai/docs/multipack.html)
*   [API Reference](https://docs.axolotl.ai/docs/api/) - Auto-generated code documentation
*   [FAQ](https://docs.axolotl.ai/docs/faq.html) - Frequently asked questions

## Support

*   Join our [Discord community](https://discord.gg/HhrNrHJPRb)
*   Explore [Examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/)
*   Read the [Debugging Guide](https://docs.axolotl.ai/docs/debugging.html)
*   For dedicated support, contact [✉️wing@axolotl.ai](mailto:wing@axolotl.ai)

## Contributing

Contributions are welcome! See our [Contributing Guide](https://github.com/axolotl-ai-cloud/axolotl/blob/main/.github/CONTRIBUTING.md) for details.

## Sponsors

Contact [wing@axolotl.ai](mailto:wing@axolotl.ai) if you're interested in sponsoring.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.