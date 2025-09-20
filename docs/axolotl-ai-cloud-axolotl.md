<p align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_white.svg">
        <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_black.svg">
        <img alt="Axolotl" src="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_black.svg" width="400" height="104" style="max-width: 100%;">
    </picture>
</p>

<p align="center">
    <strong>Fine-tune your Large Language Models (LLMs) efficiently and effortlessly with Axolotl, a free and open-source framework.</strong><br>
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

## About Axolotl

Axolotl is a powerful, free, and open-source framework designed to simplify and accelerate the process of fine-tuning and post-training for cutting-edge Large Language Models (LLMs).

[**Visit the Axolotl Repository on GitHub**](https://github.com/axolotl-ai-cloud/axolotl)

## Key Features

*   **Broad Model Support:** Train a wide array of models including GPT-OSS, LLaMA, Mistral, Mixtral, Pythia, and many more models available on the Hugging Face Hub.
*   **Multimodal Capabilities:** Fine-tune vision-language models (VLMs) such as LLaMA-Vision, Qwen2-VL, Pixtral, LLaVA, SmolVLM2, and audio models like Voxtral with image, video, and audio support.
*   **Flexible Training Methods:** Supports full fine-tuning, LoRA, QLoRA, GPTQ, QAT, Preference Tuning (DPO, IPO, KTO, ORPO), RL (GRPO), and Reward Modelling (RM) / Process Reward Modelling (PRM).
*   **Simplified Configuration:** Use a single YAML configuration file for the entire fine-tuning pipeline: dataset preprocessing, training, evaluation, quantization, and inference.
*   **Performance Optimizations:** Includes advanced optimizations like Multipacking, Flash Attention, Xformers, Flex Attention, Liger Kernel, Cut Cross Entropy, Sequence Parallelism (SP), LoRA optimizations, Multi-GPU training (FSDP1, FSDP2, DeepSpeed), Multi-node training (Torchrun, Ray), and many more!
*   **Versatile Dataset Handling:** Load data from local storage, Hugging Face datasets, and cloud storage solutions like S3, Azure, GCP, and OCI.
*   **Cloud-Ready Deployment:** Offers pre-built Docker images and PyPI packages for seamless integration with cloud platforms and local hardware.

## 🎉 Latest Updates

*   **ND Parallelism:** Added support for Compose Context Parallelism (CP), Tensor Parallelism (TP), and Fully Sharded Data Parallelism (FSDP) within a single node and across multiple nodes. (July 2024)
*   **New Model Support:** Expanded model support to include GPT-OSS, Gemma 3n, Liquid Foundation Model 2 (LFM2), and Arcee Foundation Models (AFM). (July 2024)
*   **FP8 Finetuning:** Introduced FP8 finetuning with fp8 gather op via `torchao`. (July 2024)
*   **Enhanced Model Integration:** Added support for Voxtral, Magistral 1.1, and Devstral, including mistral-common tokenizer integration. (July 2024)
*   **TiledMLP Support:** Implemented TiledMLP for single-GPU to multi-GPU training with DDP, DeepSpeed, and FSDP to support Arctic Long Sequence Training (ALST). (July 2024)
*   **Quantization Aware Training (QAT):** Added Quantization Aware Training (QAT) support. (May 2024)
*   **Sequence Parallelism (SP):** Introduced Sequence Parallelism (SP) for scaling context lengths during fine-tuning. (March 2024)

<details>
<summary>Expand Older Updates</summary>

*   **Magistral Integration:** Added support for Magistral with mistral-common tokenizer. (June 2024)
*   **Llama 4 Support:** Added support for Llama 4 models. (April 2024)
*   **Multimodal Fine-tuning (Beta):** Added support for fine-tuning Multimodal models. (March 2024)
*   **LoRA Optimizations:** Added LoRA optimizations for reduced memory usage and faster training. (February 2024)
*   **GRPO Support:** Integrated GRPO support. (February 2024)
*   **Reward Modelling Support:** Added Reward Modelling / Process Reward Modelling fine-tuning support. (January 2024)

</details>

## 🚀 Quick Start - Fine-tune Your LLM

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

```bash
docker run --gpus '"all"' --rm -it axolotlai/axolotl:main-latest
```

Other installation approaches are described in the [Installation Documentation](https://docs.axolotl.ai/docs/installation.html).

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

For more detailed instructions, see our [Getting Started Guide](https://docs.axolotl.ai/docs/getting-started.html).

## 📚 Documentation

*   [Installation Options](https://docs.axolotl.ai/docs/installation.html)
*   [Configuration Guide](https://docs.axolotl.ai/docs/config-reference.html)
*   [Dataset Loading](https://docs.axolotl.ai/docs/dataset_loading.html)
*   [Dataset Guide](https://docs.axolotl.ai/docs/dataset-formats/)
*   [Multi-GPU Training](https://docs.axolotl.ai/docs/multi-gpu.html)
*   [Multi-Node Training](https://docs.axolotl.ai/docs/multi-node.html)
*   [Multipacking](https://docs.axolotl.ai/docs/multipack.html)
*   [API Reference](https://docs.axolotl.ai/docs/api/)
*   [FAQ](https://docs.axolotl.ai/docs/faq.html)

## 🤝 Getting Help

*   Join our [Discord community](https://discord.gg/HhrNrHJPRb)
*   Explore our [Examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/)
*   Consult our [Debugging Guide](https://docs.axolotl.ai/docs/debugging.html)
*   For dedicated support, contact: [✉️wing@axolotl.ai](mailto:wing@axolotl.ai)

## 🌟 Contributing

Contributions are welcome! Refer to our [Contributing Guide](https://github.com/axolotl-ai-cloud/axolotl/blob/main/.github/CONTRIBUTING.md).

## ❤️ Sponsors

For sponsorship opportunities, contact [wing@axolotl.ai](mailto:wing@axolotl.ai)

## 📝 Citing Axolotl

```bibtex
@software{axolotl,
  title = {Axolotl: Open Source LLM Post-Training},
  author = {{Axolotl maintainers and contributors}},
  url = {https://github.com/axolotl-ai-cloud/axolotl},
  license = {Apache-2.0},
  year = {2023}
}
```

## 📜 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file.