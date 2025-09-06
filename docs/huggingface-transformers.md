<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-light.svg">
    <img alt="Hugging Face Transformers Library" src="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-light.svg" width="352" height="59" style="max-width: 100%;">
  </picture>
  <br/>
  <br/>
</p>

<p align="center">
    <a href="https://huggingface.com/models"><img alt="Checkpoints on Hub" src="https://img.shields.io/endpoint?url=https://huggingface.co/api/shields/models&color=brightgreen"></a>
    <a href="https://circleci.com/gh/huggingface/transformers"><img alt="Build" src="https://img.shields.io/circleci/build/github/huggingface/transformers/main"></a>
    <a href="https://github.com/huggingface/transformers/blob/main/LICENSE"><img alt="GitHub" src="https://img.shields.io/github/license/huggingface/transformers.svg?color=blue"></a>
    <a href="https://huggingface.co/docs/transformers/index"><img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/transformers/index.svg?down_color=red&down_message=offline&up_message=online"></a>
    <a href="https://github.com/huggingface/transformers/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/transformers.svg"></a>
    <a href="https://github.com/huggingface/transformers/blob/main/CODE_OF_CONDUCT.md"><img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg"></a>
    <a href="https://zenodo.org/badge/latestdoi/155220641"><img src="https://zenodo.org/badge/155220641.svg" alt="DOI"></a>
</p>

<h4 align="center">
    <p>
        <b>English</b> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_zh-hans.md">简体中文</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_zh-hant.md">繁體中文</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ko.md">한국어</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_es.md">Español</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ja.md">日本語</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_hd.md">हिन्दी</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ru.md">Русский</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_pt-br.md">Português</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_te.md">తెలుగు</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_fr.md">Français</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_de.md">Deutsch</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_vi.md">Tiếng Việt</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ar.md">العربية</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ur.md">اردو</a> |
    </p>
</h4>

# Transformers: State-of-the-Art Models for NLP, Computer Vision, and More

**Easily access and utilize cutting-edge, pre-trained models for a variety of tasks across text, vision, audio, and multimodal applications with the Hugging Face Transformers library ([see the original repository](https://github.com/huggingface/transformers)).**

*   **Key Features:**
    *   **Extensive Model Support:** Access to a vast library of pre-trained models, including BERT, GPT-2, RoBERTa, and many more.
    *   **Unified API:** Simplified API for using and fine-tuning models across different modalities and tasks.
    *   **Ease of Use:** High-level `Pipeline` API for quick experimentation and deployment.
    *   **Framework Flexibility:** Compatible with PyTorch, TensorFlow, and Flax, allowing you to choose the best framework for your needs.
    *   **Community Driven:** Benefit from a large and active community, with numerous pre-trained models and resources available on the Hugging Face Hub.
    *   **Reduce Compute Costs:** Leverage pre-trained models to save time and resources in training and inference.

## Installation

Follow the instructions below to install the `transformers` library, along with its dependencies:

### Prerequisites

*   **Python:** 3.9+
*   **Deep Learning Frameworks:** PyTorch 2.1+, TensorFlow 2.6+, and Flax 0.4.1+

### Installation Steps

1.  **Virtual Environment (Recommended):** Create and activate a virtual environment using either [venv](https://docs.python.org/3/library/venv.html) or [uv](https://docs.astral.sh/uv/):
    ```bash
    # venv
    python -m venv .my-env
    source .my-env/bin/activate
    # uv
    uv venv .my-env
    source .my-env/bin/activate
    ```

2.  **Install Transformers:**
    ```bash
    # pip
    pip install "transformers[torch]"

    # uv
    uv pip install "transformers[torch]"
    ```

    To install from source for the latest changes:
    ```bash
    git clone https://github.com/huggingface/transformers.git
    cd transformers

    # pip
    pip install .[torch]

    # uv
    uv pip install .[torch]
    ```

## Quickstart: Getting Started with the Pipeline API

The `Pipeline` API is your entry point for utilizing pre-trained models for tasks like text generation, translation, and image classification. The pipeline handles preprocessing and postprocessing steps for you.

```python
from transformers import pipeline

# Example: Text Generation
generator = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B")
result = generator("the secret to baking a really good cake is ")
print(result[0]["generated_text"])
```

**Chat Example:**

```python
import torch
from transformers import pipeline

chat_history = [
    {"role": "system", "content": "You are a sassy, wise-cracking robot."},
    {"role": "user", "content": "Tell me about New York."},
]

text_generation_pipeline = pipeline(
    task="text-generation",
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    dtype=torch.bfloat16,
    device_map="auto",
)
response = text_generation_pipeline(chat_history, max_new_tokens=512)
print(response[0]["generated_text"][-1]["content"])
```

**More example tasks:**

*   **Automatic speech recognition**
*   **Image classification**
*   **Visual question answering**

## Why Use Transformers?

*   **Simplified Access to State-of-the-Art Models:** Easily use and fine-tune models for various tasks, from NLP to computer vision.
*   **Efficiency and Cost Savings:** Leverage pre-trained models to reduce computational costs and training time.
*   **Framework Agnostic:** Switch between PyTorch, TensorFlow, and Flax to fit your project's needs.
*   **Customization:** Adapt models and examples to suit your specific requirements.

## Why Might You Not Use Transformers?

*   **Not a Modular Toolbox:** The library focuses on model definitions rather than a general-purpose modular system.
*   **Training API Limitations:**  The training API is optimized for PyTorch models provided by Transformers.  For generic machine learning loops, you should use another library like [Accelerate](https://huggingface.co/docs/accelerate).
*   **Example Scripts:** Example scripts may require adaptation for your specific use case.

## Projects Built with Transformers

Explore and contribute to the growing ecosystem of projects built using Transformers on the [awesome-transformers](./awesome-transformers.md) page.

## Example Models

*   **Audio:** Whisper, Moonshine, Wav2Vec2, Moshi, MusicGen, Bark
*   **Computer Vision:** SAM, DepthPro, DINO v2, SuperPoint, SuperGlue, RT-DETRv2, VitPose, OneFormer, VideoMAE
*   **Multimodal:** Qwen2-Audio, LayoutLMv3, Qwen-VL, BLIP-2, GOT-OCR2, TAPAS, Emu3, Llava-OneVision, Llava, Kosmos-2
*   **NLP:** ModernBERT, Gemma, Mixtral, BART, T5, Llama, Qwen

## Citation

If you use the Transformers library, please cite the following paper:

```bibtex
@inproceedings{wolf-etal-2020-transformers,
    title = "Transformers: State-of-the-Art Natural Language Processing",
    author = "Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = oct,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-demos.6",
    pages = "38--45"
}
```