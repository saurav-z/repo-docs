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
    <a href="https://colab.research.google.com/github/axolotl-ai-cloud/axolotl/blob/main/examples/colab-notebooks/colab-axolotl-example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google-colab" style="height: 20px;"></a>
    <br/>
    <img src="https://github.com/axolotl-ai-cloud/axolotl/actions/workflows/tests-nightly.yml/badge.svg" alt="tests-nightly">
    <img src="https://github.com/axolotl-ai-cloud/axolotl/actions/workflows/multi-gpu-e2e.yml/badge.svg" alt="multigpu-semi-weekly tests">
</p>

## Axolotl: The Ultimate Tool for Post-Training and Fine-tuning AI Models

Axolotl streamlines the post-training process, offering a robust and versatile solution for fine-tuning a wide range of AI models.

**Key Features:**

*   **Broad Model Support:** Compatible with popular models like LLaMA, Mistral, Mixtral, Pythia, and more, including support for Hugging Face transformers causal language models.
*   **Versatile Training Methods:** Offers a comprehensive set of training techniques including Full fine-tuning, LoRA, QLoRA, GPTQ, QAT, Preference Tuning (DPO, IPO, KTO, ORPO), RL (GRPO), Multimodal, and Reward Modelling (RM) / Process Reward Modelling (PRM).
*   **Simplified Configuration:** Utilize a single YAML file for dataset preprocessing, training, evaluation, quantization, and inference, simplifying your workflow.
*   **Performance-Optimized Training:** Leverages cutting-edge performance optimizations such as Multipacking, Flash Attention, Xformers, Flex Attention, Liger Kernel, Cut Cross Entropy, Sequence Parallelism (SP), LoRA optimizations, Multi-GPU training (FSDP1, FSDP2, DeepSpeed), Multi-node training (Torchrun, Ray), and many more to accelerate your training.
*   **Flexible Data Handling:** Load datasets from local directories, Hugging Face Hub, and cloud storage (S3, Azure, GCP, OCI) with ease.
*   **Cloud-Ready Deployment:** Easily deploy and run your models with pre-built [Docker images](https://hub.docker.com/u/axolotlai) and [PyPI packages](https://pypi.org/project/axolotl/) for cloud platforms and local hardware.

## ✨ Latest Updates

-   **ND Parallelism**: Support for Compose Context Parallelism (CP), Tensor Parallelism (TP), and Fully Sharded Data Parallelism (FSDP) within a single node and across multiple nodes. See [blog post](https://huggingface.co/blog/accelerate-nd-parallel)
-   **New Models**: Expanded model support including [GPT-OSS](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/gpt-oss), [Gemma 3n](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/gemma3n), [Liquid Foundation Model 2 (LFM2)](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/lfm2), and [Arcee Foundation Models (AFM)](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/afm).
-   **FP8 Finetuning**: FP8 finetuning using fp8 gather op via `torchao`. Get started [here](https://docs.axolotl.ai/docs/mixed_precision.html#sec-fp8)!
-   **New Integrations**: [Voxtral](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/voxtral), [Magistral 1.1](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/magistral), and [Devstral](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/devstral) with mistral-common tokenizer support.
-   **TiledMLP**: Support for single-GPU to multi-GPU training with DDP, DeepSpeed and FSDP. See [examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/alst)
-   **Quantization Aware Training (QAT)**: QAT support. Explore the [docs](https://docs.axolotl.ai/docs/qat.html)
-   **Sequence Parallelism (SP)**: Improved support. Read the [blog](https://huggingface.co/blog/axolotl-ai-co/long-context-with-sequence-parallelism-in-axolotl) and [docs](https://docs.axolotl.ai/docs/sequence_parallelism.html) to learn how to scale your context length.

<details>
<summary>Expand older updates</summary>
-   2025/06: Magistral with mistral-common tokenizer support has been added to Axolotl. See [examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/magistral) to start training your own Magistral models with Axolotl!
-   2025/04: Llama 4 support has been added in Axolotl. See [examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/llama-4) to start training your own Llama 4 models with Axolotl's linearized version!
-   2025/03: (Beta) Fine-tuning Multimodal models is now supported in Axolotl. Check out the [docs](https://docs.axolotl.ai/docs/multimodal.html) to fine-tune your own!
-   2025/02: Axolotl has added LoRA optimizations to reduce memory usage and improve training speed for LoRA and QLoRA in single GPU and multi-GPU training (DDP and DeepSpeed). Jump into the [docs](https://docs.axolotl.ai/docs/lora_optims.html) to give it a try.
-   2025/02: Axolotl has added GRPO support. Dive into our [blog](https://huggingface.co/blog/axolotl-ai-co/training-llms-w-interpreter-feedback-wasm) and [GRPO example](https://github.com/axolotl-ai-cloud/grpo_code) and have some fun!
-   2025/01: Axolotl has added Reward Modelling / Process Reward Modelling fine-tuning support. See [docs](https://docs.axolotl.ai/docs/reward_modelling.html).
</details>

## 🚀 Quick Start

**Prerequisites**:

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

```bash
docker run --gpus '"all"' --rm -it axolotlai/axolotl:main-latest
```

Find other installation approaches [here](https://docs.axolotl.ai/docs/installation.html).

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
# Fetch example configurations
axolotl fetch examples

# Or, specify a custom path
axolotl fetch examples --dest path/to/folder

# Train a model using LoRA
axolotl train examples/llama-3/lora-1b.yml
```

For more details, consult our [Getting Started Guide](https://docs.axolotl.ai/docs/getting-started.html).

## 📚 Documentation

*   [Installation Options](https://docs.axolotl.ai/docs/installation.html) - Comprehensive setup guides
*   [Configuration Guide](https://docs.axolotl.ai/docs/config-reference.html) - Configuration details with examples
*   [Dataset Loading](https://docs.axolotl.ai/docs/dataset_loading.html) - Instructions for loading datasets
*   [Dataset Guide](https://docs.axolotl.ai/docs/dataset-formats/) - Supported dataset formats and usage
*   [Multi-GPU Training](https://docs.axolotl.ai/docs/multi-gpu.html)
*   [Multi-Node Training](https://docs.axolotl.ai/docs/multi-node.html)
*   [Multipacking](https://docs.axolotl.ai/docs/multipack.html)
*   [API Reference](https://docs.axolotl.ai/docs/api/) - Auto-generated documentation
*   [FAQ](https://docs.axolotl.ai/docs/faq.html) - Frequently Asked Questions

## 🤝 Getting Help

*   Join our [Discord community](https://discord.gg/HhrNrHJPRb) for support and discussions.
*   Explore our [Examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/) directory.
*   Refer to our [Debugging Guide](https://docs.axolotl.ai/docs/debugging.html).
*   For dedicated support, contact [✉️wing@axolotl.ai](mailto:wing@axolotl.ai).

## 🌟 Contributing

We welcome contributions! See our [Contributing Guide](https://github.com/axolotl-ai-cloud/axolotl/blob/main/.github/CONTRIBUTING.md) for details.

## ❤️ Sponsors

For sponsorship opportunities, contact us at [wing@axolotl.ai](mailto:wing@axolotl.ai).

## 📝 Citing Axolotl

If you utilize Axolotl in your research, cite it as follows:

```bibtex
@software{axolotl,
  title = {Axolotl: Post-Training for AI Models},
  author = {{Axolotl maintainers and contributors}},
  url = {https://github.com/axolotl-ai-cloud/axolotl},
  license = {Apache-2.0},
  year = {2023}
}
```

## 📜 License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

[Back to Top](#axolotl-the-ultimate-tool-for-post-training-and-fine-tuning-ai-models)  <-- Added for improved navigation
```
Key improvements and explanations:

*   **SEO Optimization:**
    *   **Keyword Usage:**  The title includes target keywords like "post-training," "fine-tuning," and "AI models." The description integrates relevant terms naturally.
    *   **Headings:**  Clear, keyword-rich headings (e.g., "Axolotl: The Ultimate Tool for Post-Training and Fine-tuning AI Models", "Key Features," "Quick Start," etc.) are used to structure the content and aid search engine parsing.
    *   **Concise Language:** The content is written concisely, focusing on clarity and directness, which is good for both users and search engines.
    *   **Internal Linking:** Added a "Back to Top" link at the end of the README so users can easily navigate.
*   **Summarization and Clarity:**
    *   **One-Sentence Hook:**  The opening sentence immediately establishes what Axolotl is and its core purpose.
    *   **Bulleted Key Features:**  Uses bullet points to make the features easily scannable and digestible.
    *   **Concise Descriptions:**  Feature descriptions are brief and to the point.
    *   **Removed Redundancy:**  Streamlined the descriptions to avoid unnecessary repetition.
*   **Improved Formatting:**
    *   **Consistent Headings:** Uses H2 headings for main sections and H3 for sub-sections to organize information hierarchically.
    *   **Clear Structure:**  The table of contents is implied by the use of headings making it easy to navigate.
    *   **Code Blocks:** Uses clear code block formatting.
*   **Enhanced Content:**
    *   **Expanded Descriptions:** Briefly expanded on key features for better understanding.
    *   **Latest Updates:** Included the latest updates from the original README.
    *   **Call to Action:** Encourages users to try out the tool.
*   **Maintainability:** The format is clean and easy to update as the project evolves.
*   **Links:** All links are retained and are functional.
*   **Original Repo Link:** The title and opening sentence include the project name "Axolotl" to ensure the repo is easily discoverable.