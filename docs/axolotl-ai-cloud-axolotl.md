<p align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_white.svg">
        <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_black.svg">
        <img alt="Axolotl" src="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_black.svg" width="400" height="104" style="max-width: 100%;">
    </picture>
</p>

<p align="center">
    <strong>Fine-tune Large Language Models (LLMs) Easily with Axolotl: Your Free & Open-Source Solution.</strong><br>
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
    <a href="https://colab.research.google.com/github/axolotl-ai-cloud/axolotl/blob/main/examples/colab-notebooks/colab-axolotl-example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google-colab" style="height: 20px;"></a>
    <br/>
    <img src="https://github.com/axolotl-ai-cloud/axolotl/actions/workflows/tests-nightly.yml/badge.svg" alt="tests-nightly">
    <img src="https://github.com/axolotl-ai-cloud/axolotl/actions/workflows/multi-gpu-e2e.yml/badge.svg" alt="multigpu-semi-weekly tests">
</p>

## Overview

Axolotl is a powerful, free, and open-source framework designed for streamlining the fine-tuning and post-training processes of state-of-the-art Large Language Models (LLMs). Easily customize and optimize your LLMs with Axolotl, from initial training to deployment.  Access the original repository [here](https://github.com/axolotl-ai-cloud/axolotl).

## Key Features

*   **Extensive Model Support:** Fine-tune a wide array of LLMs, including GPT-OSS, LLaMA, Mistral, Mixtral, Pythia, and models available on the Hugging Face Hub.
*   **Multimodal Training:** Support for Vision-Language Models (VLMs) such as LLaMA-Vision, Qwen2-VL, Pixtral, LLaVA, SmolVLM2, and audio models like Voxtral, with image, video, and audio support.
*   **Diverse Training Methods:** Utilize full fine-tuning, LoRA, QLoRA, GPTQ, QAT, Preference Tuning (DPO, IPO, KTO, ORPO), RL (GRPO), and Reward Modelling (RM) / Process Reward Modelling (PRM) for optimal results.
*   **Simplified Configuration:** Employ a single YAML configuration file across the entire fine-tuning pipeline, streamlining dataset preprocessing, training, evaluation, quantization, and inference.
*   **Performance Optimization:** Benefit from advanced techniques such as Multipacking, Flash Attention, Xformers, Flex Attention, Liger Kernel, Cut Cross Entropy, Sequence Parallelism (SP), LoRA optimizations, and multi-GPU training (FSDP1, FSDP2, DeepSpeed) to improve speed and efficiency.
*   **Flexible Data Handling:** Load datasets from local files, Hugging Face Hub, and cloud storage platforms (S3, Azure, GCP, OCI).
*   **Cloud-Ready Deployment:** Deploy easily with pre-built Docker images and PyPI packages, making it simple to run on cloud platforms and local hardware.

## Latest Updates

*   **2024/07:**
    *   **ND Parallelism:** Introduced support for ND Parallelism, integrating Context Parallelism (CP), Tensor Parallelism (TP), and Fully Sharded Data Parallelism (FSDP) within single and multi-node setups.  Learn more [here](https://huggingface.co/blog/accelerate-nd-parallel).
    *   **New Model Support:** Added support for GPT-OSS, Gemma 3n, Liquid Foundation Model 2 (LFM2), and Arcee Foundation Models (AFM).
    *   **FP8 Finetuning:** Implemented FP8 fine-tuning via `torchao` with fp8 gather op.  Get started [here](https://docs.axolotl.ai/docs/mixed_precision.html#sec-fp8)!
    *   **New Models:** Integrated Voxtral, Magistral 1.1, and Devstral with mistral-common tokenizer support.
    *   **ALST Support:** Added TiledMLP support with DDP, DeepSpeed and FSDP support for Arctic Long Sequence Training (ALST). See [examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/alst).
*   **2024/05:**  Quantization Aware Training (QAT) support added. Explore the [docs](https://docs.axolotl.ai/docs/qat.html).
*   **2024/03:**  Sequence Parallelism (SP) support implemented. Read the [blog](https://huggingface.co/blog/axolotl-ai-co/long-context-with-sequence-parallelism-in-axolotl) and [docs](https://docs.axolotl.ai/docs/sequence_parallelism.html) to learn how to scale your context length.

<details>
<summary>Expand Older Updates</summary>

*   **2024/06:** Added Magistral with mistral-common tokenizer support. See [examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/magistral).
*   **2024/04:** Added Llama 4 support. See [examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/llama-4).
*   **2024/03 (Beta):** Fine-tuning of Multimodal models supported.  Check out the [docs](https://docs.axolotl.ai/docs/multimodal.html).
*   **2024/02:** Added LoRA optimizations for memory reduction and faster training in single and multi-GPU setups (DDP & DeepSpeed). See [docs](https://docs.axolotl.ai/docs/lora_optims.html).
*   **2024/02:** Added GRPO support. Dive into our [blog](https://huggingface.co/blog/axolotl-ai-co/training-llms-w-interpreter-feedback-wasm) and [GRPO example](https://github.com/axolotl-ai-cloud/grpo_code).
*   **2024/01:** Added Reward Modelling / Process Reward Modelling fine-tuning support. See [docs](https://docs.axolotl.ai/docs/reward_modelling.html).
</details>

## Quick Start

### Requirements

*   NVIDIA GPU (Ampere or newer recommended for `bf16` and Flash Attention) or AMD GPU
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

Docker offers a streamlined, less error-prone installation process.

```bash
docker run --gpus '"all"' --rm -it axolotlai/axolotl:main-latest
```

Other installation methods are outlined [here](https://docs.axolotl.ai/docs/installation.html).

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

### First Fine-tune

```bash
# Fetch axolotl examples
axolotl fetch examples

# Or, specify a custom path
axolotl fetch examples --dest path/to/folder

# Train a model using LoRA
axolotl train examples/llama-3/lora-1b.yml
```

For detailed steps, refer to our [Getting Started Guide](https://docs.axolotl.ai/docs/getting-started.html).

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
*   Explore our [Examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/)
*   Consult the [Debugging Guide](https://docs.axolotl.ai/docs/debugging.html)
*   For dedicated support, contact [✉️wing@axolotl.ai](mailto:wing@axolotl.ai)

## Contributing

Contributions are highly encouraged; review our [Contributing Guide](https://github.com/axolotl-ai-cloud/axolotl/blob/main/.github/CONTRIBUTING.md) for detailed instructions.

## Sponsors

Interested in sponsoring? Please reach out at [wing@axolotl.ai](mailto:wing@axolotl.ai).

## Citing

If you use Axolotl, please cite it:

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

Licensed under the Apache 2.0 License; see the [LICENSE](LICENSE) file for details.