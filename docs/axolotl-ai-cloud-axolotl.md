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

## Axolotl: Your One-Stop Solution for AI Model Post-Training

Axolotl simplifies the fine-tuning and optimization of large language models (LLMs) and other AI models, allowing you to train and deploy your models with ease.  ([See the original repo](https://github.com/axolotl-ai-cloud/axolotl)).

### Key Features

*   **Wide Model Compatibility**: Supports fine-tuning of models like LLaMA, Mistral, Mixtral, Pythia, and more, including compatibility with Hugging Face transformers causal language models.
*   **Diverse Training Methods**: Offers various training methods, including full fine-tuning, LoRA, QLoRA, GPTQ, QAT, Preference Tuning (DPO, IPO, KTO, ORPO), RL (GRPO), Multimodal, and Reward Modelling (RM) / Process Reward Modelling (PRM).
*   **Simplified Configuration**: Uses a single YAML file for dataset preprocessing, training, evaluation, quantization, and inference, streamlining your workflow.
*   **Performance Optimization**: Integrated with performance-enhancing features like Multipacking, Flash Attention, Xformers, Flex Attention, Liger Kernel, Cut Cross Entropy, Sequence Parallelism (SP), LoRA optimizations, and Multi-GPU & Multi-Node training support (FSDP1, FSDP2, DeepSpeed, Torchrun, Ray) and many more!
*   **Flexible Data Handling**: Supports dataset loading from local, Hugging Face, and cloud storage solutions (S3, Azure, GCP, OCI).
*   **Cloud-Ready**: Provides Docker images and PyPI packages for easy deployment on cloud platforms and local hardware.

### Latest Updates

*   **(2025/07)**: ND Parallelism support, New Model Support (GPT-OSS, Gemma 3n, LFM2, AFM), FP8 finetuning, Voxtral, Magistral 1.1, Devstral, and TiledMLP Support for ALST
*   **(2025/05)**: Quantization Aware Training (QAT) support.
*   **(2025/03)**: Sequence Parallelism (SP) support.

<details>

<summary>Expand older updates</summary>

*   **(2025/06)**: Magistral with mistral-common tokenizer support.
*   **(2025/04)**: Llama 4 support.
*   **(2025/03)**: (Beta) Fine-tuning Multimodal models support.
*   **(2025/02)**: LoRA optimizations for memory usage and speed improvements, GRPO support.
*   **(2025/01)**: Reward Modelling / Process Reward Modelling fine-tuning support.

</details>

### Quick Start

#### Requirements

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

##### Cloud Providers

<details>

*   [RunPod](https://runpod.io/gsc?template=v2ickqhz9s&ref=6i7fkpdz)
*   [Vast.ai](https://cloud.vast.ai?ref_id=62897&template_id=bdd4a49fa8bce926defc99471864cace&utm_source=github&utm_medium=developer_community&utm_campaign=template_launch_axolotl&utm_content=readme)
*   [PRIME Intellect](https://app.primeintellect.ai/dashboard/create-cluster?image=axolotl&location=Cheapest&security=Cheapest&show_spot=true)
*   [Modal](https://www.modal.com?utm_source=github&utm_medium=github&utm_campaign=axolotl)
*   [Novita](https://novita.ai/gpus-console?templateId=311)
*   [JarvisLabs.ai](https://jarvislabs.ai/templates/axolotl)
*   [Latitude.sh](https://latitude.sh/blueprint/989e0e79-3bf6-41ea-a46b-1f246e309d5c)

</details>

#### Your First Fine-tune

```bash
# Fetch axolotl examples
axolotl fetch examples

# Or, specify a custom path
axolotl fetch examples --dest path/to/folder

# Train a model using LoRA
axolotl train examples/llama-3/lora-1b.yml
```

For more detailed guidance, check out the [Getting Started Guide](https://docs.axolotl.ai/docs/getting-started.html).

### Documentation

*   [Installation Options](https://docs.axolotl.ai/docs/installation.html)
*   [Configuration Guide](https://docs.axolotl.ai/docs/config-reference.html)
*   [Dataset Loading](https://docs.axolotl.ai/docs/dataset_loading.html)
*   [Dataset Guide](https://docs.axolotl.ai/docs/dataset-formats/)
*   [Multi-GPU Training](https://docs.axolotl.ai/docs/multi-gpu.html)
*   [Multi-Node Training](https://docs.axolotl.ai/docs/multi-node.html)
*   [Multipacking](https://docs.axolotl.ai/docs/multipack.html)
*   [API Reference](https://docs.axolotl.ai/docs/api/)
*   [FAQ](https://docs.axolotl.ai/docs/faq.html)

### Getting Help

*   [Discord community](https://discord.gg/HhrNrHJPRb)
*   [Examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/)
*   [Debugging Guide](https://docs.axolotl.ai/docs/debugging.html)
*   Dedicated support: [✉️wing@axolotl.ai](mailto:wing@axolotl.ai)

### Contributing

Please review the [Contributing Guide](https://github.com/axolotl-ai-cloud/axolotl/blob/main/.github/CONTRIBUTING.md).

### Sponsors

Contact [wing@axolotl.ai](mailto:wing@axolotl.ai) for sponsorship inquiries.

### Citing Axolotl

```bibtex
@software{axolotl,
  title = {Axolotl: Post-Training for AI Models},
  author = {{Axolotl maintainers and contributors}},
  url = {https://github.com/axolotl-ai-cloud/axolotl},
  license = {Apache-2.0},
  year = {2023}
}
```

### License

Apache 2.0 License - see the [LICENSE](LICENSE) file for details.