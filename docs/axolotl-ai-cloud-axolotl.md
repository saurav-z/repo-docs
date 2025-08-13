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

## Axolotl: Supercharge Your AI Model Training with Ease

Axolotl is a powerful and flexible tool designed for post-training various AI models, making fine-tuning and optimization a breeze.  [Check out the original repo](https://github.com/axolotl-ai-cloud/axolotl) for more details.

### Key Features

*   **Broad Model Compatibility:** Supports a wide range of models, including LLaMA, Mistral, Mixtral, Pythia, and other HuggingFace transformers causal language models.
*   **Versatile Training Methods:** Offers a comprehensive suite of training techniques:
    *   Full fine-tuning
    *   LoRA & QLoRA
    *   GPTQ
    *   QAT
    *   Preference Tuning (DPO, IPO, KTO, ORPO)
    *   Reinforcement Learning (GRPO)
    *   Multimodal training
    *   Reward Modeling (RM) / Process Reward Modeling (PRM)
*   **Simplified Configuration:** Streamlines your workflow with a single YAML configuration file for dataset preprocessing, training, evaluation, quantization, and inference.
*   **Performance-Driven Optimizations:** Includes advanced optimizations for enhanced training speed and efficiency:
    *   Multipacking
    *   Flash Attention
    *   Xformers
    *   Flex Attention
    *   Liger Kernel
    *   Cut Cross Entropy
    *   Sequence Parallelism (SP)
    *   LoRA Optimizations
    *   Multi-GPU Training (FSDP1, FSDP2, DeepSpeed)
    *   Multi-Node Training (Torchrun, Ray)
*   **Flexible Data Handling:** Supports loading datasets from various sources:
    *   Local storage
    *   Hugging Face Hub
    *   Cloud storage (S3, Azure, GCP, OCI)
*   **Cloud-Ready Deployment:** Provides Docker images and PyPI packages for seamless deployment on cloud platforms and local hardware.

### Latest Updates

*   **2025/07:**
    *   Added ND Parallelism support (CP, TP, FSDP) within a single node and across multiple nodes.  See the [blog post](https://huggingface.co/blog/accelerate-nd-parallel).
    *   Expanded model support: GPT-OSS, Gemma 3n, Liquid Foundation Model 2 (LFM2), and Arcee Foundation Models (AFM).
    *   FP8 finetuning with fp8 gather op via `torchao`. Get started [here](https://docs.axolotl.ai/docs/mixed_precision.html#sec-fp8)!
    *   Integrated Voxtral, Magistral 1.1, and Devstral with mistral-common tokenizer support.
    *   Added TiledMLP support for Arctic Long Sequence Training (ALST). See [examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/alst) for usage.
*   **2025/05:** Introduced Quantization Aware Training (QAT) support. Explore the [docs](https://docs.axolotl.ai/docs/qat.html) for details.
*   **2025/03:** Implemented Sequence Parallelism (SP).  Read the [blog](https://huggingface.co/blog/axolotl-ai-co/long-context-with-sequence-parallelism-in-axolotl) and [docs](https://docs.axolotl.ai/docs/sequence_parallelism.html) to learn how to scale your context length.
<details>
<summary>Expand older updates</summary>

*   **2025/06:** Added Magistral with mistral-common tokenizer support. See [examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/magistral) to train your models.
*   **2025/04:** Added Llama 4 support. See [examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/llama-4).
*   **2025/03 (Beta):** Fine-tuning multimodal models support added. Check out the [docs](https://docs.axolotl.ai/docs/multimodal.html).
*   **2025/02:** Added LoRA optimizations to reduce memory usage and improve training speed for LoRA and QLoRA (single & multi-GPU). See [docs](https://docs.axolotl.ai/docs/lora_optims.html).
*   **2025/02:** Added GRPO support. See [blog](https://huggingface.co/blog/axolotl-ai-co/training-llms-w-interpreter-feedback-wasm) and [GRPO example](https://github.com/axolotl-ai-cloud/grpo_code).
*   **2025/01:** Added Reward Modelling / Process Reward Modelling fine-tuning support. See [docs](https://docs.axolotl.ai/docs/reward_modelling.html).
</details>

### 🚀 Quick Start

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

Get started with our [Getting Started Guide](https://docs.axolotl.ai/docs/getting-started.html).

### 📚 Documentation

*   [Installation Options](https://docs.axolotl.ai/docs/installation.html) - Detailed setup instructions for different environments
*   [Configuration Guide](https://docs.axolotl.ai/docs/config-reference.html) - Full configuration options and examples
*   [Dataset Loading](https://docs.axolotl.ai/docs/dataset_loading.html) - Loading datasets from various sources
*   [Dataset Guide](https://docs.axolotl.ai/docs/dataset-formats/) - Supported formats and how to use them
*   [Multi-GPU Training](https://docs.axolotl.ai/docs/multi-gpu.html)
*   [Multi-Node Training](https://docs.axolotl.ai/docs/multi-node.html)
*   [Multipacking](https://docs.axolotl.ai/docs/multipack.html)
*   [API Reference](https://docs.axolotl.ai/docs/api/) - Auto-generated code documentation
*   [FAQ](https://docs.axolotl.ai/docs/faq.html) - Frequently asked questions

### 🤝 Getting Help

*   Join our [Discord community](https://discord.gg/HhrNrHJPRb) for support
*   Check out our [Examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/) directory
*   Read our [Debugging Guide](https://docs.axolotl.ai/docs/debugging.html)
*   For dedicated support, contact [✉️wing@axolotl.ai](mailto:wing@axolotl.ai)

### 🌟 Contributing

Contributions are welcome!  See our [Contributing Guide](https://github.com/axolotl-ai-cloud/axolotl/blob/main/.github/CONTRIBUTING.md) for details.

### ❤️ Sponsors

Interested in sponsoring? Contact [wing@axolotl.ai](mailto:wing@axolotl.ai)

### 📝 Citing Axolotl

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

### 📜 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.