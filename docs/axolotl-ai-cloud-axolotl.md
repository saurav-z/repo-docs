<p align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_white.svg">
        <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_black.svg">
        <img alt="Axolotl" src="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_black.svg" width="400" height="104" style="max-width: 100%;">
    </picture>
</p>

[![GitHub License](https://img.shields.io/github/license/axolotl-ai-cloud/axolotl.svg?color=blue)](https://github.com/axolotl-ai-cloud/axolotl/blob/main/LICENSE)
[![Tests](https://github.com/axolotl-ai-cloud/axolotl/actions/workflows/tests.yml/badge.svg)](https://github.com/axolotl-ai-cloud/axolotl/actions/workflows/tests.yml)
[![Codecov](https://codecov.io/gh/axolotl-ai-cloud/axolotl/branch/main/graph/badge.svg)](https://codecov.io/gh/axolotl-ai-cloud/axolotl)
[![Releases](https://img.shields.io/github/release/axolotl-ai-cloud/axolotl.svg)](https://github.com/axolotl-ai-cloud/axolotl/releases)
[![Contributors](https://img.shields.io/github/contributors-anon/axolotl-ai-cloud/axolotl?color=yellow&style=flat-square)](https://github.com/axolotl-ai-cloud/axolotl/graphs/contributors)
[![GitHub Repo stars](https://img.shields.io/github/stars/axolotl-ai-cloud/axolotl)](https://github.com/axolotl-ai-cloud/axolotl)
[![Discord](https://img.shields.io/badge/discord-7289da.svg?style=flat-square&logo=discord)](https://discord.com/invite/HhrNrHJPRb)
[![Twitter](https://img.shields.io/twitter/follow/axolotl_ai?style=social)](https://twitter.com/axolotl_ai)
[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/axolotl-ai-cloud/axolotl/blob/main/examples/colab-notebooks/colab-axolotl-example.ipynb)
[![Tests Nightly](https://github.com/axolotl-ai-cloud/axolotl/actions/workflows/tests-nightly.yml/badge.svg)](https://github.com/axolotl-ai-cloud/axolotl/actions/workflows/tests-nightly.yml)
[![Multi-GPU Semi-Weekly Tests](https://github.com/axolotl-ai-cloud/axolotl/actions/workflows/multi-gpu-e2e.yml/badge.svg)](https://github.com/axolotl-ai-cloud/axolotl/actions/workflows/multi-gpu-e2e.yml)

## Axolotl: Supercharge Your AI Model Training and Fine-tuning

Axolotl is a versatile tool designed to simplify and optimize the post-training process for a wide range of AI models, enabling researchers and developers to fine-tune and deploy state-of-the-art models efficiently.

## Key Features

*   **Wide Model Support**: Compatible with Hugging Face Transformers models, including LLaMA, Mistral, Mixtral, Pythia, and more.
*   **Comprehensive Training Methods**: Supports Full Fine-tuning, LoRA, QLoRA, GPTQ, QAT, Preference Tuning (DPO, IPO, KTO, ORPO), RL (GRPO), Multimodal, and Reward Modelling (RM) / Process Reward Modelling (PRM).
*   **Simplified Configuration**: Train, evaluate, and quantize your models easily using a single, unified YAML configuration file.
*   **Performance Optimization**: Boost training speed and efficiency with features like Multipacking, Flash Attention, Xformers, Flex Attention, Liger Kernel, Cut Cross Entropy, Sequence Parallelism (SP), LoRA optimizations, and support for Multi-GPU and Multi-node training.
*   **Flexible Data Handling**: Loads datasets from local storage, Hugging Face Hub, and cloud providers (S3, Azure, GCP, OCI).
*   **Cloud-Ready**: Ready to deploy with provided Docker images and PyPI packages, making it easy to run on cloud platforms.

## Latest Updates

-   **[July 2024]**:
    -   **ND Parallelism Support**:  Achieve higher scalability with Compose Context Parallelism (CP), Tensor Parallelism (TP), and Fully Sharded Data Parallelism (FSDP) within a single node and across multiple nodes. See [blog post](https://huggingface.co/blog/accelerate-nd-parallel) for more information.
    -   **New Model Support**:  Expanded model compatibility with [GPT-OSS](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/gpt-oss), [Gemma 3n](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/gemma3n), [Liquid Foundation Model 2 (LFM2)](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/lfm2), and [Arcee Foundation Models (AFM)](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/afm).
    -   **FP8 Finetuning with `torchao`**: Now supports FP8 fine-tuning using the `torchao` library and fp8 gather op. Get started [here](https://docs.axolotl.ai/docs/mixed_precision.html#sec-fp8)!
    -   **Integration of Additional Models**:  Added support for [Voxtral](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/voxtral), [Magistral 1.1](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/magistral), and [Devstral](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/devstral) with mistral-common tokenizer support.
    -   **TiledMLP and ALST Support**: Added support for TiledMLP for single-GPU to multi-GPU training with DDP, DeepSpeed, and FSDP to support Arctic Long Sequence Training (ALST). See [examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/alst) for usage.
-   **[May 2024]**: Quantization Aware Training (QAT) support has been added. Explore the [docs](https://docs.axolotl.ai/docs/qat.html) to learn more!
-   **[March 2024]**: Implemented Sequence Parallelism (SP) support. Read the [blog](https://huggingface.co/blog/axolotl-ai-co/long-context-with-sequence-parallelism-in-axolotl) and [docs](https://docs.axolotl.ai/docs/sequence_parallelism.html) to learn how to scale your context length when fine-tuning.

<details>
<summary>Expand Older Updates</summary>

-   **[June 2024]**: Added Magistral with mistral-common tokenizer support. See [examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/magistral) to start training your own Magistral models!
-   **[April 2024]**: Llama 4 support added. See [examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/llama-4) to start training your own Llama 4 models with Axolotl's linearized version!
-   **[March 2024] (Beta)**: Fine-tuning Multimodal models is now supported.  Check out the [docs](https://docs.axolotl.ai/docs/multimodal.html).
-   **[February 2024]**: Added LoRA optimizations to reduce memory usage and improve training speed for LoRA and QLoRA in single GPU and multi-GPU training (DDP and DeepSpeed). Jump into the [docs](https://docs.axolotl.ai/docs/lora_optims.html) to give it a try.
-   **[February 2024]**: Added GRPO support. Dive into our [blog](https://huggingface.co/blog/axolotl-ai-co/training-llms-w-interpreter-feedback-wasm) and [GRPO example](https://github.com/axolotl-ai-cloud/grpo_code).
-   **[January 2024]**: Added Reward Modelling / Process Reward Modelling fine-tuning support. See [docs](https://docs.axolotl.ai/docs/reward_modelling.html).

</details>

## Quick Start

Follow these steps to get started with Axolotl:

**Requirements**:

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

Installing with Docker can be less error prone than installing in your own environment.

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

For a more detailed walkthrough, check out our [Getting Started Guide](https://docs.axolotl.ai/docs/getting-started.html).

## Documentation

*   [Installation Options](https://docs.axolotl.ai/docs/installation.html) - Detailed setup instructions
*   [Configuration Guide](https://docs.axolotl.ai/docs/config-reference.html) - Configuration options and examples
*   [Dataset Loading](https://docs.axolotl.ai/docs/dataset_loading.html) - Loading datasets from various sources
*   [Dataset Guide](https://docs.axolotl.ai/docs/dataset-formats/) - Supported formats
*   [Multi-GPU Training](https://docs.axolotl.ai/docs/multi-gpu.html)
*   [Multi-Node Training](https://docs.axolotl.ai/docs/multi-node.html)
*   [Multipacking](https://docs.axolotl.ai/docs/multipack.html)
*   [API Reference](https://docs.axolotl.ai/docs/api/)
*   [FAQ](https://docs.axolotl.ai/docs/faq.html)

## Getting Help

*   Join our [Discord community](https://discord.gg/HhrNrHJPRb)
*   Explore our [Examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/)
*   Consult our [Debugging Guide](https://docs.axolotl.ai/docs/debugging.html)
*   For dedicated support, contact [✉️wing@axolotl.ai](mailto:wing@axolotl.ai)

## Contributing

Contributions are welcome! Please refer to our [Contributing Guide](https://github.com/axolotl-ai-cloud/axolotl/blob/main/.github/CONTRIBUTING.md).

## Sponsors

For sponsorship opportunities, please contact [wing@axolotl.ai](mailto:wing@axolotl.ai)

## Citing Axolotl

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

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.