<p align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_white.svg">
        <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_black.svg">
        <img alt="Axolotl" src="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_black.svg" width="400" height="104" style="max-width: 100%;">
    </picture>
</p>

<!-- Badges Section -->
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

## Axolotl: The Ultimate Toolkit for Fine-Tuning and Optimizing AI Models

Axolotl empowers you to effortlessly fine-tune and optimize your AI models, providing a streamlined and efficient post-training experience.  [Get started with Axolotl](https://github.com/axolotl-ai-cloud/axolotl)!

### Key Features:

*   **Wide Model Support**: Train LLaMA, Mistral, Mixtral, Pythia, and other Hugging Face transformer models.
*   **Diverse Training Methods**: Utilize full fine-tuning, LoRA, QLoRA, GPTQ, QAT, Preference Tuning, RL (GRPO), Multimodal, and Reward Modelling (RM) / Process Reward Modelling (PRM).
*   **Simplified Configuration**: Use a single YAML file for dataset preprocessing, training, evaluation, quantization, and inference.
*   **Performance Optimization**: Leverage [Multipacking](https://docs.axolotl.ai/docs/multipack.html), [Flash Attention](https://github.com/Dao-AILab/flash-attention), [Xformers](https://github.com/facebookresearch/xformers), [Flex Attention](https://pytorch.org/blog/flexattention/), [Liger Kernel](https://github.com/linkedin/Liger-Kernel), [Cut Cross Entropy](https://github.com/apple/ml-cross-entropy/tree/main), [Sequence Parallelism (SP)](https://docs.axolotl.ai/docs/sequence_parallelism.html), [LoRA optimizations](https://docs.axolotl.ai/docs/lora_optims.html), [Multi-GPU training (FSDP1, FSDP2, DeepSpeed)](https://docs.axolotl.ai/docs/multi-gpu.html), [Multi-node training (Torchrun, Ray)](https://docs.axolotl.ai/docs/multi-node.html), and more.
*   **Flexible Data Handling**: Load datasets from local storage, Hugging Face Hub, and cloud providers (S3, Azure, GCP, OCI).
*   **Cloud-Ready Deployment**: Deploy quickly with provided [Docker images](https://hub.docker.com/u/axolotlai) and [PyPI packages](https://pypi.org/project/axolotl/) for cloud and local hardware.

##  What's New?

*   **ND Parallelism**: Axolotl now supports ND Parallelism, combining CP, TP, and FSDP within and across nodes.
*   **Expanded Model Support**:  Train GPT-OSS, Gemma 3n, Liquid Foundation Model 2 (LFM2), and Arcee Foundation Models (AFM).
*   **FP8 Finetuning**:  Utilize fp8 gather op via `torchao`.
*   **Enhanced Tokenizer Support**: Integration of Voxtral, Magistral 1.1, and Devstral with mistral-common tokenizer.
*   **ALST Support**: TiledMLP and ALST support for long sequence training.
*   **Quantization Aware Training (QAT)**: Added QAT support.
*   **Sequence Parallelism (SP)**: Added Sequence Parallelism (SP) support.

<details>
<summary>Expand older updates</summary>

*   **Magistral**:  Magistral with mistral-common tokenizer support is integrated.
*   **Llama 4**: Support for Llama 4 with the linearized version is added.
*   **Multimodal Fine-tuning (Beta)**: Fine-tuning Multimodal models is now supported.
*   **LoRA Optimizations**:  LoRA optimizations to reduce memory usage and improve training speed.
*   **GRPO Support**: Added GRPO support.
*   **Reward Modelling**: Support for Reward Modelling / Process Reward Modelling fine-tuning.
</details>

##  Quick Start Guide

###  Prerequisites:

*   NVIDIA GPU (Ampere or newer for `bf16` and Flash Attention) or AMD GPU
*   Python 3.11
*   PyTorch ≥2.6.0

###  Installation:

#### Using pip:

```bash
pip3 install -U packaging==23.2 setuptools==75.8.0 wheel ninja
pip3 install --no-build-isolation axolotl[flash-attn,deepspeed]

# Download example axolotl configs, deepspeed configs
axolotl fetch examples
axolotl fetch deepspeed_configs  # OPTIONAL
```

#### Using Docker:

```bash
docker run --gpus '"all"' --rm -it axolotlai/axolotl:main-latest
```

Explore other installation methods [here](https://docs.axolotl.ai/docs/installation.html).

#### Cloud Deployment:

<details>

*   [RunPod](https://runpod.io/gsc?template=v2ickqhz9s&ref=6i7fkpdz)
*   [Vast.ai](https://cloud.vast.ai?ref_id=62897&template_id=bdd4a49fa8bce926defc99471864cace&utm_source=github&utm_medium=developer_community&utm_campaign=template_launch_axolotl&utm_content=readme)
*   [PRIME Intellect](https://app.primeintellect.ai/dashboard/create-cluster?image=axolotl&location=Cheapest&security=Cheapest&show_spot=true)
*   [Modal](https://www.modal.com?utm_source=github&utm_medium=github&utm_campaign=axolotl)
*   [Novita](https://novita.ai/gpus-console?templateId=311)
*   [JarvisLabs.ai](https://jarvislabs.ai/templates/axolotl)
*   [Latitude.sh](https://latitude.sh/blueprint/989e0e79-3bf6-41ea-a46b-1f246e309d5c)

</details>

### Your First Fine-tune:

```bash
# Fetch axolotl examples
axolotl fetch examples

# Or, specify a custom path
axolotl fetch examples --dest path/to/folder

# Train a model using LoRA
axolotl train examples/llama-3/lora-1b.yml
```

Refer to the [Getting Started Guide](https://docs.axolotl.ai/docs/getting-started.html) for a more in-depth walkthrough.

## Resources & Documentation

*   [Installation Guide](https://docs.axolotl.ai/docs/installation.html)
*   [Configuration Reference](https://docs.axolotl.ai/docs/config-reference.html)
*   [Dataset Loading](https://docs.axolotl.ai/docs/dataset_loading.html)
*   [Dataset Formats](https://docs.axolotl.ai/docs/dataset-formats/)
*   [Multi-GPU Training](https://docs.axolotl.ai/docs/multi-gpu.html)
*   [Multi-Node Training](https://docs.axolotl.ai/docs/multi-node.html)
*   [Multipacking](https://docs.axolotl.ai/docs/multipack.html)
*   [API Reference](https://docs.axolotl.ai/docs/api/)
*   [FAQ](https://docs.axolotl.ai/docs/faq.html)

##  Get Support

*   Join our [Discord community](https://discord.gg/HhrNrHJPRb)
*   Explore our [Examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/)
*   Consult our [Debugging Guide](https://docs.axolotl.ai/docs/debugging.html)
*   For dedicated support, contact [✉️wing@axolotl.ai](mailto:wing@axolotl.ai)

##  Contribute

We welcome contributions!  Please see our [Contributing Guide](https://github.com/axolotl-ai-cloud/axolotl/blob/main/.github/CONTRIBUTING.md) for details.

##  Sponsors

For sponsorship opportunities, please reach out to [wing@axolotl.ai](mailto:wing@axolotl.ai).

##  Citing Axolotl

```bibtex
@software{axolotl,
  title = {Axolotl: Post-Training for AI Models},
  author = {{Axolotl maintainers and contributors}},
  url = {https://github.com/axolotl-ai-cloud/axolotl},
  license = {Apache-2.0},
  year = {2023}
}
```

##  License

Axolotl is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.