<p align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_white.svg">
        <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_black.svg">
        <img alt="Axolotl" src="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/887513285d98132142bf5db2a74eb5e0928787f1/image/axolotl_logo_digital_black.svg" width="400" height="104" style="max-width: 100%;">
    </picture>
</p>

<p align="center">
    <strong>Supercharge your LLMs with Axolotl: The open-source framework for effortless fine-tuning.</strong>
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

*   **Wide Model Support**: Train various LLMs, including GPT-OSS, LLaMA, Mistral, Mixtral, Pythia, and models from Hugging Face Hub.
*   **Multimodal Fine-tuning**: Supports vision-language models (VLMs) like LLaMA-Vision, Qwen2-VL, Pixtral, and audio models such as Voxtral.
*   **Diverse Training Methods**: Utilize full fine-tuning, LoRA, QLoRA, GPTQ, QAT, preference tuning (DPO, IPO, KTO, ORPO), RL (GRPO), and Reward Modelling (RM) / Process Reward Modelling (PRM).
*   **Simplified Configuration**: Streamline your workflow with a single YAML file for dataset preprocessing, training, evaluation, quantization, and inference.
*   **Performance Enhancements**: Leverage cutting-edge optimizations like Multipacking, Flash Attention, Xformers, Flex Attention, Liger Kernel, Cut Cross Entropy, Sequence Parallelism (SP), LoRA optimizations, multi-GPU training (FSDP1, FSDP2, DeepSpeed), and multi-node training (Torchrun, Ray).
*   **Flexible Data Handling**: Load datasets from local files, Hugging Face Hub, and cloud storage (S3, Azure, GCP, OCI).
*   **Cloud-Ready**: Deploy easily with Docker images and PyPI packages for cloud and local environments.

## What's New
* **2025/07**:
    *   **ND Parallelism:** Support for Compose Context Parallelism (CP), Tensor Parallelism (TP), and Fully Sharded Data Parallelism (FSDP) within a single node and across multiple nodes.
    *   **New Models:**  GPT-OSS, Gemma 3n, Liquid Foundation Model 2 (LFM2), and Arcee Foundation Models (AFM).
    *   **FP8 Finetuning:**  Support for FP8 finetuning with fp8 gather op via `torchao`.
    *   **Model Integrations:** Voxtral, Magistral 1.1, and Devstral with mistral-common tokenizer support.
    *   **ALST Support:**  TiledMLP support for single-GPU to multi-GPU training with DDP, DeepSpeed, and FSDP.
*   **2025/05**:
    *   **QAT Support:** Quantization Aware Training (QAT) support has been added.
*   **2025/03**:
    *   **Sequence Parallelism (SP):**  Implementation of Sequence Parallelism (SP) to scale context lengths during fine-tuning.
<details>
<summary>
    Older Updates
</summary>
*   **2025/06:** Magistral with mistral-common tokenizer support.
*   **2025/04:** Llama 4 support.
*   **2025/03:** Fine-tuning Multimodal models support (Beta).
*   **2025/02:**  LoRA optimizations for single and multi-GPU training. GRPO support.
*   **2025/01:** Reward Modelling / Process Reward Modelling fine-tuning support.
</details>

## 🚀 Quick Start - Fine-tune Your LLM

### Requirements

*   NVIDIA GPU (Ampere or newer for `bf16` and Flash Attention) or AMD GPU
*   Python 3.11
*   PyTorch ≥ 2.6.0

### Get Started Quickly

#### Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/axolotl-ai-cloud/axolotl/blob/main/examples/colab-notebooks/colab-axolotl-example.ipynb#scrollTo=msOCO4NRmRLa)

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

For alternative installation methods, refer to the [installation guide](https://docs.axolotl.ai/docs/installation.html).

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

### First Fine-tune

```bash
# Fetch axolotl examples
axolotl fetch examples

# Or, specify a custom path
axolotl fetch examples --dest path/to/folder

# Train a model using LoRA
axolotl train examples/llama-3/lora-1b.yml
```

Explore our [Getting Started Guide](https://docs.axolotl.ai/docs/getting-started.html) for a detailed walkthrough.

## 📚 Comprehensive Documentation

*   [Installation Options](https://docs.axolotl.ai/docs/installation.html)
*   [Configuration Guide](https://docs.axolotl.ai/docs/config-reference.html)
*   [Dataset Loading](https://docs.axolotl.ai/docs/dataset_loading.html)
*   [Dataset Guide](https://docs.axolotl.ai/docs/dataset-formats/)
*   [Multi-GPU Training](https://docs.axolotl.ai/docs/multi-gpu.html)
*   [Multi-Node Training](https://docs.axolotl.ai/docs/multi-node.html)
*   [Multipacking](https://docs.axolotl.ai/docs/multipack.html)
*   [API Reference](https://docs.axolotl.ai/docs/api/)
*   [FAQ](https://docs.axolotl.ai/docs/faq.html)

## 🤝 Get Support and Connect

*   Join our [Discord community](https://discord.gg/HhrNrHJPRb)
*   Browse our [Examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/)
*   Check out our [Debugging Guide](https://docs.axolotl.ai/docs/debugging.html)
*   For dedicated support, contact [✉️wing@axolotl.ai](mailto:wing@axolotl.ai)

## 🌟 Contribute

We welcome contributions! Please see our [Contributing Guide](https://github.com/axolotl-ai-cloud/axolotl/blob/main/.github/CONTRIBUTING.md).

## ❤️ Sponsors

For sponsorship opportunities, contact [wing@axolotl.ai](mailto:wing@axolotl.ai)

## 📝 Citing Axolotl

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

## 📜 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

[Back to Top](#) - [Axolotl GitHub Repository](https://github.com/axolotl-ai-cloud/axolotl)
```
Key improvements and changes:

*   **SEO Optimization:** Included relevant keywords like "LLM," "fine-tuning," "open source," etc. throughout the README.
*   **Concise Hook:**  "Supercharge your LLMs with Axolotl: The open-source framework for effortless fine-tuning." This is more engaging than a basic statement.
*   **Clear Headings and Organization:**  Used headings (e.g., "Key Features," "Quick Start") and subheadings for readability and scannability.
*   **Bulleted Lists:** Features are presented in easy-to-read bulleted lists.
*   **Highlights:** Highlighted key benefits and the most important features, including recent updates.
*   **Concise Language:**  Rephrased information for better flow and clarity.
*   **Call to Action:** Included a clear "Quick Start" section to encourage users to get started.
*   **Links Back:** Added "Back to Top" and the link to the GitHub repo at the end.
*   **Improved Formatting**: Use of bolded text to emphasize key points.
*   **Contextual Information:** Added links for deeper understanding.
*   **Cloud Providers:** Improved the cloud provider integration section.
*   **Clearer Structure**: Separated features and recent updates.