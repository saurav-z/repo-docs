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
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_pt-br.md">Рortuguês</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_te.md">తెలుగు</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_fr.md">Français</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_de.md">Deutsch</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_vi.md">Tiếng Việt</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ar.md">العربية</a> |
        <a href="https://github.com/huggingface/transformers/blob/main/i18n/README_ur.md">اردو</a> |
    </p>
</h4>

# Hugging Face Transformers: State-of-the-Art Models for NLP, Vision, and More

**Harness the power of cutting-edge AI with Hugging Face Transformers, a library providing pre-trained models for text, vision, audio, and multimodal tasks, simplifying the development and deployment of advanced AI applications.**  Get started with the official [Hugging Face Transformers repository](https://github.com/huggingface/transformers).

<h3 align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_as_a_model_definition.png"/>
</h3>

## Key Features

*   **Extensive Model Support:** Access a vast collection of pre-trained models for a wide range of tasks across text, computer vision, audio, video, and multimodal domains.
*   **Unified API:**  A consistent and easy-to-use API for utilizing all available pre-trained models.
*   **Simplified Usage:** High-level `Pipeline` API for streamlined inference and task execution.
*   **Framework Flexibility:** Train and deploy models across PyTorch, TensorFlow, and Flax.
*   **Community and Hub Integration:** Leverage the Hugging Face Hub to discover, share, and utilize a massive library of model checkpoints (over 1M+).
*   **Easy Customization:** Adapt models to your specific needs, with exposed internals for customization and experimentation.
*   **Reduced Costs:** Minimize training costs and carbon footprint by leveraging pre-trained models and fine-tuning them.

## Installation

The Transformers library is compatible with Python 3.9+ and supports PyTorch 2.1+, TensorFlow 2.6+, and Flax 0.4.1+.

**Install with `pip` (Recommended):**

```bash
pip install "transformers[torch]"  # Or [tensorflow] or [flax] for other frameworks
```

**Install from Source (For the latest changes):**

```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install .[torch]  # Or [tensorflow] or [flax]
```

## Quickstart

Get up and running quickly with the `Pipeline` API:

```python
from transformers import pipeline

# Text Generation Example
pipeline = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B")
pipeline("the secret to baking a really good cake is ")
```

Explore the documentation for more details on using the `Pipeline` API for various modalities and tasks:
*   Automatic Speech Recognition
*   Image Classification
*   Visual Question Answering

## Why Use Transformers?

*   **State-of-the-Art Performance:** Achieve high accuracy on a wide range of AI tasks.
*   **Accessibility:** Lower the barrier to entry for researchers, engineers, and developers.
*   **Cost-Effectiveness:** Reduce compute costs and carbon footprint.
*   **Flexibility:** Choose the right framework for training, evaluation, and deployment.
*   **Customization:** Easily adapt models to your specific use cases.
*   **Community:** Leverage a large community and readily available models.

## When Not to Use Transformers

*   Not a modular toolbox for building neural networks.
*   Training API is optimized for PyTorch models.
*   Example scripts may require adaptation for specific needs.

## Projects Built with Transformers

Join the community! Check out the [awesome-transformers](./awesome-transformers.md) page for a list of incredible projects utilizing the Transformers library.

## Example Models

Explore a selection of pre-trained models across different modalities:

*   **Audio:** Audio classification, speech recognition, text-to-speech, etc.
*   **Computer Vision:** Image classification, object detection, segmentation, and more.
*   **Multimodal:** Image captioning, visual question answering, and other multimodal applications.
*   **NLP:** Text generation, text classification, summarization, translation, and question answering.

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