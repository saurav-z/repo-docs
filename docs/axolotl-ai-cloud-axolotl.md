<p align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_white.svg">
        <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_black.svg">
        <img alt="Axolotl" src="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_black.svg" width="400" height="104" style="max-width: 100%;">
    </picture>
</p>

<p align="center">
    <strong>Fine-tune your large language models efficiently and cost-effectively with Axolotl, the free and open-source LLM fine-tuning framework.</strong><br>
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

## Key Features of Axolotl

*   **Versatile Model Support:** Train a wide range of LLMs, including popular models from Hugging Face Hub like GPT-OSS, LLaMA, Mistral, Mixtral, and Pythia.
*   **Multimodal Fine-tuning:** Supports fine-tuning of vision-language models (VLMs) such as LLaMA-Vision, Qwen2-VL, Pixtral, LLaVA, and audio models like Voxtral, with image, video, and audio support.
*   **Comprehensive Training Methods:** Offers a variety of training techniques, including full fine-tuning, LoRA, QLoRA, GPTQ, QAT, Preference Tuning (DPO, IPO, KTO, ORPO), RL (GRPO), and Reward Modelling (RM) / Process Reward Modelling (PRM).
*   **Simplified Configuration:** Utilize a single YAML configuration file to manage the entire fine-tuning pipeline, including dataset preprocessing, training, evaluation, quantization, and inference.
*   **Performance Optimization:** Leverages cutting-edge techniques such as Multipacking, Flash Attention, Xformers, Flex Attention, Liger Kernel, Cut Cross Entropy, Sequence Parallelism (SP), LoRA optimizations, and Multi-GPU training (FSDP1, FSDP2, DeepSpeed) for enhanced efficiency.
*   **Flexible Dataset Handling:** Easily load datasets from local directories, Hugging Face, and cloud storage services like S3, Azure, GCP, and OCI.
*   **Cloud-Ready Deployment:** Provides Docker images and PyPI packages, enabling seamless deployment on cloud platforms and local hardware.

**[Visit the Axolotl GitHub Repository](https://github.com/axolotl-ai-cloud/axolotl) for more details and to get started!**

## Latest Updates

*   **ND Parallelism Support:** Introduced ND Parallelism to Axolotl, enabling Compose Context Parallelism (CP), Tensor Parallelism (TP), and Fully Sharded Data Parallelism (FSDP) within a single node and across multiple nodes.  Read more in the [blog post](https://huggingface.co/blog/accelerate-nd-parallel).
*   **Expanded Model Support:** Added support for new models, including [GPT-OSS](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/gpt-oss), [Gemma 3n](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/gemma3n), [Liquid Foundation Model 2 (LFM2)](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/lfm2), and [Arcee Foundation Models (AFM)](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/afm).
*   **FP8 Finetuning:** Integrated FP8 fine-tuning capabilities with fp8 gather op via `torchao`, learn more [here](https://docs.axolotl.ai/docs/mixed_precision.html#sec-fp8).
*   **Mistral-Common Tokenizer Support:** Introduced support for [Voxtral](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/voxtral), [Magistral 1.1](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/magistral), and [Devstral](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/devstral).
*   **TiledMLP Support:** Added TiledMLP to enable single-GPU to multi-GPU training with DDP, DeepSpeed, and FSDP for Arctic Long Sequence Training (ALST), [examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/alst) for more information.
*   **Quantization Aware Training (QAT):** Added QAT support, see [docs](https://docs.axolotl.ai/docs/qat.html).
*   **Sequence Parallelism (SP) Support:** Implemented Sequence Parallelism (SP) to scale context length, read the [blog](https://huggingface.co/blog/axolotl-ai-co/long-context-with-sequence-parallelism-in-axolotl) and [docs](https://docs.axolotl.ai/docs/sequence_parallelism.html).

<details>

<summary>Expand older updates</summary>

*   2025/06: Magistral with mistral-common tokenizer support has been added to Axolotl. See [examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/magistral) to start training your own Magistral models with Axolotl!
*   2025/04: Llama 4 support has been added in Axolotl. See [examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/llama-4) to start training your own Llama 4 models with Axolotl's linearized version!
*   2025/03: (Beta) Fine-tuning Multimodal models is now supported in Axolotl. Check out the [docs](https://docs.axolotl.ai/docs/multimodal.html) to fine-tune your own!
*   2025/02: Axolotl has added LoRA optimizations to reduce memory usage and improve training speed for LoRA and QLoRA in single GPU and multi-GPU training (DDP and DeepSpeed). Jump into the [docs](https://docs.axolotl.ai/docs/lora_optims.html) to give it a try.
*   2025/02: Axolotl has added GRPO support. Dive into our [blog](https://huggingface.co/blog/axolotl-ai-co/training-llms-w-interpreter-feedback-wasm) and [GRPO example](https://github.com/axolotl-ai-cloud/grpo_code) and have some fun!
*   2025/01: Axolotl has added Reward Modelling / Process Reward Modelling fine-tuning support. See [docs](https://docs.axolotl.ai/docs/reward_modelling.html).

</details>

## 🚀 Quick Start - LLM Fine-tuning in Minutes

**Requirements**:

-   NVIDIA GPU (Ampere or newer for `bf16` and Flash Attention) or AMD GPU
-   Python 3.11
-   PyTorch ≥2.6.0

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

That's it! Check out our [Getting Started Guide](https://docs.axolotl.ai/docs/getting-started.html) for a more detailed walkthrough.

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
*   Need dedicated support? Please contact [✉️wing@axolotl.ai](mailto:wing@axolotl.ai) for options

## 🌟 Contributing

Contributions are welcome! Please see our [Contributing Guide](https://github.com/axolotl-ai-cloud/axolotl/blob/main/.github/CONTRIBUTING.md) for details.

## ❤️ Sponsors

Interested in sponsoring? Contact us at [wing@axolotl.ai](mailto:wing@axolotl.ai)

## 📝 Citing Axolotl

If you use Axolotl in your research or projects, please cite it as follows:

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

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.