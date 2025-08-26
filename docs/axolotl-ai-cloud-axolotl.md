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


## Axolotl: Fine-tune and Optimize Your AI Models with Ease

Axolotl is a powerful tool designed to streamline and accelerate the post-training process for a wide variety of AI models, offering flexible and efficient training solutions.  [Visit the original repository](https://github.com/axolotl-ai-cloud/axolotl) for more information.

**Key Features:**

*   **Broad Model Support**: Train models like LLaMA, Mistral, Mixtral, Pythia, and other Hugging Face causal language models.
*   **Diverse Training Methods**:  Supports full fine-tuning, LoRA, QLoRA, GPTQ, QAT, Preference Tuning (DPO, IPO, KTO, ORPO), RL (GRPO), Multimodal, and Reward Modelling (RM) / Process Reward Modelling (PRM).
*   **Simplified Configuration**:  Utilize a single YAML file to manage dataset preprocessing, training, evaluation, quantization, and inference.
*   **Performance-Driven Optimizations**: Includes features like Multipacking, Flash Attention, Xformers, Flex Attention, Liger Kernel, Cut Cross Entropy, Sequence Parallelism (SP), LoRA optimizations, and multi-GPU/node training support for speed and memory efficiency.
*   **Flexible Data Handling**: Load datasets from local files, Hugging Face Hub, and cloud storage providers (S3, Azure, GCP, OCI).
*   **Cloud-Ready**:  Deploy easily with provided Docker images and PyPI packages.

## Latest Updates

*   **[July 2024 Updates]** - Enhanced with ND Parallelism support (Compose Context Parallelism (CP), Tensor Parallelism (TP), and Fully Sharded Data Parallelism (FSDP)), added support for models like GPT-OSS, Gemma 3n, Liquid Foundation Model 2 (LFM2), and Arcee Foundation Models (AFM).  Includes FP8 finetuning via torchao and integration of models like Voxtral, Magistral 1.1, and Devstral with mistral-common tokenizer support.  Introduced TiledMLP support for Arctic Long Sequence Training (ALST).
*   **[May 2024 Update]** - Quantization Aware Training (QAT) support has been added to Axolotl.
*   **[March 2024 Update]** - Implemented Sequence Parallelism (SP) to scale context length when fine-tuning.
<details>
<summary>Expand older updates</summary>

*   **[June 2024 Update]** -  Magistral with mistral-common tokenizer support has been added.
*   **[April 2024 Update]** - Llama 4 support has been added.
*   **[March 2024 Update]** - (Beta) Fine-tuning Multimodal models is now supported.
*   **[February 2024 Update]** - LoRA optimizations were added for memory usage and improved training speed.
*   **[February 2024 Update]** - GRPO support was added.
*   **[January 2024 Update]** - Reward Modelling / Process Reward Modelling fine-tuning support was added.

</details>

## Quick Start

**Requirements:**

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

Installing with Docker can be less error prone than installing in your own environment.

```bash
docker run --gpus '"all"' --rm -it axolotlai/axolotl:main-latest
```

Other installation approaches are described [here](https://docs.axolotl.ai/docs/installation.html).

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

Check out our [Getting Started Guide](https://docs.axolotl.ai/docs/getting-started.html) for a more detailed walkthrough.

## Documentation

*   [Installation Options](https://docs.axolotl.ai/docs/installation.html) - Detailed setup instructions
*   [Configuration Guide](https://docs.axolotl.ai/docs/config-reference.html) - Full configuration options
*   [Dataset Loading](https://docs.axolotl.ai/docs/dataset_loading.html) - Loading datasets from various sources
*   [Dataset Guide](https://docs.axolotl.ai/docs/dataset-formats/) - Supported formats and how to use them
*   [Multi-GPU Training](https://docs.axolotl.ai/docs/multi-gpu.html)
*   [Multi-Node Training](https://docs.axolotl.ai/docs/multi-node.html)
*   [Multipacking](https://docs.axolotl.ai/docs/multipack.html)
*   [API Reference](https://docs.axolotl.ai/docs/api/) - Auto-generated code documentation
*   [FAQ](https://docs.axolotl.ai/docs/faq.html) - Frequently asked questions

## Getting Help

*   Join our [Discord community](https://discord.gg/HhrNrHJPRb) for support
*   Explore the [Examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/) directory
*   Consult the [Debugging Guide](https://docs.axolotl.ai/docs/debugging.html)
*   For dedicated support, contact [✉️wing@axolotl.ai](mailto:wing@axolotl.ai)

## Contributing

Contributions are welcome!  See the [Contributing Guide](https://github.com/axolotl-ai-cloud/axolotl/blob/main/.github/CONTRIBUTING.md).

## Sponsors

For sponsorship opportunities, please contact [wing@axolotl.ai](mailto:wing@axolotl.ai).

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

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file.