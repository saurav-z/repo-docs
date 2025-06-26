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

# Hugging Face Transformers: State-of-the-Art Models for NLP, Computer Vision, and More

**Harness the power of cutting-edge AI with Hugging Face Transformers, a comprehensive library offering pre-trained models for a wide range of tasks and modalities.**  ([See the original repo](https://github.com/huggingface/transformers))

<h3 align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_as_a_model_definition.png"/>
</h3>

## Key Features:

*   **Extensive Pre-trained Models:** Access a vast library of over 1M+ pre-trained models for text, images, audio, video, and multimodal tasks available on the [Hugging Face Hub](https://huggingface.com/models).
*   **Unified API:**  Use a consistent and intuitive API for interacting with all supported models, simplifying development and experimentation.
*   **Framework Agnostic:** Seamlessly integrate models with popular deep learning frameworks like PyTorch, TensorFlow, and Flax, allowing flexibility in your workflow.
*   **Ease of Use:**  Get started quickly with the `Pipeline` API, a high-level tool for easy inference and task execution.
*   **Model Customization:**  Fine-tune and adapt models to your specific needs with examples and comprehensive documentation.
*   **Community-Driven:** Benefit from a vibrant and supportive community contributing to the library's development and providing resources.
*   **Lower Compute Costs:** Leverage pre-trained models to reduce training time, resource consumption, and carbon footprint.

## Getting Started

### Installation

Transformers supports Python 3.9+ and requires PyTorch 2.1+, TensorFlow 2.6+, or Flax 0.4.1+. Choose your preferred package manager and install as shown below:

```bash
# pip
pip install "transformers[torch]" #or tensorflow, or flax

# uv
uv pip install "transformers[torch]" #or tensorflow, or flax
```

### Quickstart with Pipeline

The `Pipeline` API simplifies common tasks. Here's a quick example of text generation:

```python
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B")
pipeline("the secret to baking a really good cake is ")
# ... (output)
```

Explore the provided examples in the original README for additional usage scenarios and tasks.

## Why Use Transformers?

*   **State-of-the-Art Performance:** Achieve high accuracy in NLP, computer vision, and audio tasks.
*   **Cost-Effective:** Save on compute by using pre-trained models and transfer learning.
*   **Flexible Framework Support:**  Train, evaluate, and deploy models across different frameworks.
*   **Customization:** Tailor models to your specific use cases with comprehensive examples.

## Why *Not* Use Transformers?

*   **Not a General Toolbox:**  This library focuses on model definitions, not building blocks for custom neural networks.
*   **Training API:**  Training is optimized for PyTorch models provided by Transformers. For general machine learning loops, use [Accelerate](https://huggingface.co/docs/accelerate).
*   **Example Scripts:** Example scripts might require adaptation for specific use cases.

## Projects Using Transformers

Discover incredible projects built with Transformers and become part of the community!  See the [awesome-transformers](./awesome-transformers.md) page for a list of projects.

## Example Models

Here are a few example models that can be found on the [Hub model pages](https://huggingface.co/models).

*(See the original README for the complete model lists for each modality.)*

*   **Audio:** Whisper, Moonshine, Bark
*   **Computer Vision:** SAM, DINO v2, RT-DETRv2
*   **Multimodal:** Qwen-VL, BLIP-2, Llava
*   **NLP:** Gemma, Mixtral, Llama, Qwen

## Citation

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