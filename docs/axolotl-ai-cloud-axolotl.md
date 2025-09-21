<p align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_white.svg">
        <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_black.svg">
        <img alt="Axolotl" src="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_black.svg" width="400" height="104" style="max-width: 100%;">
    </picture>
</p>

## Axolotl: Fine-tune LLMs Effortlessly with this Open-Source Framework

Axolotl is a powerful, open-source framework that simplifies the fine-tuning process for large language models (LLMs), making it easier than ever to train and deploy custom models. ([Original Repository](https://github.com/axolotl-ai-cloud/axolotl))

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

## Key Features

*   **Broad Model Support:** Train a wide variety of LLMs, including GPT-OSS, LLaMA, Mistral, Mixtral, Pythia, and models from the Hugging Face Hub.
*   **Multimodal Capabilities:**  Fine-tune vision-language models (VLMs) like LLaMA-Vision and LLaVA, and audio models such as Voxtral.
*   **Versatile Training Methods:**  Offers comprehensive training methods including full fine-tuning, LoRA, QLoRA, GPTQ, QAT, Preference Tuning (DPO, IPO, KTO, ORPO), RL (GRPO), and Reward Modelling (RM) / Process Reward Modelling (PRM).
*   **Simplified Configuration:**  Utilize a single YAML configuration file to manage the entire fine-tuning pipeline, encompassing dataset preprocessing, training, evaluation, quantization, and inference.
*   **Performance Optimization:** Integrates various optimization techniques such as Multipacking, Flash Attention, Xformers, Flex Attention, Liger Kernel, Cut Cross Entropy, Sequence Parallelism (SP), LoRA optimizations, Multi-GPU training (FSDP1, FSDP2, DeepSpeed), and Multi-node training (Torchrun, Ray).
*   **Flexible Data Handling:** Load datasets from local files, Hugging Face Hub, and cloud storage solutions (S3, Azure, GCP, OCI).
*   **Cloud-Ready Deployment:** Provides Docker images and PyPI packages for seamless deployment on cloud platforms and local hardware.

## Latest Updates

*   **ND Parallelism Support**:  Axolotl now supports ND Parallelism, enabling the use of Context Parallelism (CP), Tensor Parallelism (TP), and Fully Sharded Data Parallelism (FSDP) within a node and across multiple nodes. [Blog Post](https://huggingface.co/blog/accelerate-nd-parallel)
*   **Expanded Model Support**: Added support for models like GPT-OSS, Gemma 3n, Liquid Foundation Model 2 (LFM2), and Arcee Foundation Models (AFM).
*   **FP8 Fine-tuning**:  FP8 fine-tuning is now possible with fp8 gather op via `torchao`.
*   **New Model Integrations**:  Integration of models such as Voxtral, Magistral 1.1, and Devstral with mistral-common tokenizer support.
*   **ALST Support**:  Added TiledMLP support for ALST.  See [examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/alst).
*   **QAT Support**: Quantization Aware Training (QAT) support is now available.  [Docs](https://docs.axolotl.ai/docs/qat.html)

<details>

<summary>Expand older updates</summary>

*   **Magistral Support**:  Added Magistral with mistral-common tokenizer support. See [examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/magistral)
*   **Llama 4 Support**: Added support for Llama 4. See [examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/llama-4)
*   **Multimodal Fine-tuning**: Support for fine-tuning Multimodal models. [Docs](https://docs.axolotl.ai/docs/multimodal.html)
*   **LoRA Optimizations**:  LoRA optimizations to reduce memory usage and improve training speed. [Docs](https://docs.axolotl.ai/docs/lora_optims.html)
*   **GRPO Support**: Added GRPO support. [Blog](https://huggingface.co/blog/axolotl-ai-co/training-llms-w-interpreter-feedback-wasm) and [GRPO example](https://github.com/axolotl-ai-cloud/grpo_code)
*   **Reward Modelling Support**:  Added Reward Modelling / Process Reward Modelling fine-tuning support. [Docs](https://docs.axolotl.ai/docs/reward_modelling.html)

</details>

## Quick Start

Follow these steps to fine-tune your LLM in minutes:

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
-   [Vast.ai](https://cloud.vast.ai?ref_id=62897&template_id=bdd4a49fa8bce926defc99471864cace&utm_source=github&utm_medium=developer-community&utm_campaign=template_launch_axolotl&utm_content=readme)
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

Consult the [Getting Started Guide](https://docs.axolotl.ai/docs/getting-started.html) for a detailed walkthrough.

## Documentation

*   [Installation Options](https://docs.axolotl.ai/docs/installation.html) - Setup instructions.
*   [Configuration Guide](https://docs.axolotl.ai/docs/config-reference.html) - Configuration options and examples.
*   [Dataset Loading](https://docs.axolotl.ai/docs/dataset_loading.html) - Loading datasets.
*   [Dataset Guide](https://docs.axolotl.ai/docs/dataset-formats/) - Supported formats.
*   [Multi-GPU Training](https://docs.axolotl.ai/docs/multi-gpu.html)
*   [Multi-Node Training](https://docs.axolotl.ai/docs/multi-node.html)
*   [Multipacking](https://docs.axolotl.ai/docs/multipack.html)
*   [API Reference](https://docs.axolotl.ai/docs/api/) - Auto-generated code documentation.
*   [FAQ](https://docs.axolotl.ai/docs/faq.html) - Frequently asked questions.

## Getting Help

*   Join the [Discord community](https://discord.gg/HhrNrHJPRb)
*   Explore [Examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/)
*   Read the [Debugging Guide](https://docs.axolotl.ai/docs/debugging.html)
*   For dedicated support, contact [✉️wing@axolotl.ai](mailto:wing@axolotl.ai).

## Contributing

Contributions are welcome! See the [Contributing Guide](https://github.com/axolotl-ai-cloud/axolotl/blob/main/.github/CONTRIBUTING.md).

## Sponsors

Contact [wing@axolotl.ai](mailto:wing@axolotl.ai) for sponsorship opportunities.

## Citing Axolotl

```bibtex
@software{axolotl,
  title = {Axolotl: Open Source LLM Post-Training},
  author = {{Axolotl maintainers and contributors}},
  url = {https://github.com/axolotl-ai-cloud/axolotl},
  license = {Apache-2.0},
  year = {2023}
}
```

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file.