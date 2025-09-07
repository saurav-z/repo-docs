# Transformers: State-of-the-Art Models for NLP, Computer Vision, and More

**Unlock the power of cutting-edge AI with Hugging Face Transformers, your gateway to pre-trained models for diverse tasks.** Learn more and contribute at the original repository: [https://github.com/huggingface/transformers](https://github.com/huggingface/transformers).

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

### Key Features

*   **Broad Compatibility:** Seamlessly supports PyTorch, TensorFlow, and Flax.
*   **Diverse Applications:** Covers text, computer vision, audio, video, and multimodal tasks.
*   **Vast Model Library:** Access over 1M+ pre-trained models on the Hugging Face Hub.
*   **Ease of Use:** Simple API for both inference and training, with high-level `Pipeline` for quick tasks.
*   **Customization:** Easy to adapt models and examples to your specific needs.
*   **Cost-Effective:** Reduce compute costs by leveraging pre-trained models instead of training from scratch.

### Overview

Transformers is the go-to library for utilizing state-of-the-art machine learning models. It provides pre-trained models for various tasks, from natural language processing to computer vision and audio processing. This library simplifies model definition, making it compatible across training frameworks, inference engines, and adjacent modeling libraries. It promotes easy customization and efficient usage of the latest AI models.

### Installation

Install the library using pip or uv within a virtual environment.

```bash
# Create a virtual environment
python -m venv .my-env
source .my-env/bin/activate

# Install with pip
pip install "transformers[torch]"

# Install with uv
uv pip install "transformers[torch]"

# Install from source
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install .[torch]
```

### Quickstart

The `Pipeline` API is your entry point for many tasks. Here is a quick example:

```python
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B")
pipeline("the secret to baking a really good cake is ")
```

Explore the provided examples for different modalities and tasks (Automatic speech recognition, Image classification, Visual question answering).

### Why Use Transformers?

*   **Accessibility:** Easy-to-use, state-of-the-art models for various tasks.
*   **Efficiency:** Reduce costs and carbon footprint through model sharing.
*   **Flexibility:** Train, evaluate, and deploy models across different frameworks.
*   **Customization:** Adapt models and examples to your specific needs.

### Why You Might Not Use Transformers

*   Not a modular building block library.
*   The training API works with PyTorch models provided by Transformers.
*   Example scripts may require adaptation for specific use cases.

### Projects Using Transformers

Join a thriving community of projects built around Transformers on the Hugging Face Hub. See the [awesome-transformers](./awesome-transformers.md) for more.

### Example Models

(See the original document for detailed example model breakdowns.)

### Citation

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