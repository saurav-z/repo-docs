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

# Hugging Face Transformers: State-of-the-Art NLP, Computer Vision, Audio, and Multimodal Models

**Hugging Face Transformers empowers you with cutting-edge, pre-trained models for various AI tasks, simplifying both inference and training.** ([View on GitHub](https://github.com/huggingface/transformers))

<h3 align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_as_a_model_definition.png"/>
</h3>

## Key Features:

*   **Extensive Model Support:** Access a vast library of over 1 million+ pre-trained models for text, images, audio, video, and multimodal tasks, all available on the [Hugging Face Hub](https://huggingface.co/models).
*   **Simplified Usage:**  Utilize a unified API for easy implementation of state-of-the-art models.
*   **Efficiency and Cost Savings:** Leverage pre-trained models to reduce compute costs and accelerate development.
*   **Framework Flexibility:** Seamlessly switch between PyTorch, TensorFlow, and Flax for training and inference.
*   **Customization & Community:**  Easily adapt models to your needs with provided examples and a thriving community.

## Installation

Transformers supports Python 3.9+ with PyTorch, TensorFlow, and Flax.

```bash
# Recommended: Create a virtual environment with venv or uv (a fast Rust-based Python package manager).
# Install Transformers with pip or uv:
pip install "transformers[torch]"
# or
uv pip install "transformers[torch]"
```

## Quickstart

Get started with Transformers using the `Pipeline` API.

```python
from transformers import pipeline

# Example: Text Generation
pipeline = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B")
pipeline("the secret to baking a really good cake is ")
```

## Why Use Transformers?

*   **User-Friendly:** Intuitive and easy to learn.
*   **Resource Efficient:** Save time and compute by using pre-trained models.
*   **Framework Agnostic:** Choose your preferred framework for training and deployment.
*   **Highly Customizable:** Easily modify models and integrate them into your projects.

## Why Not Use Transformers?

*   This library is not a modular toolbox of building blocks for neural nets.
*   The training API is optimized to work with PyTorch models provided by Transformers.
*   Example scripts are only *examples* and may need to be adapted.

## Explore the Community

Explore the [awesome-transformers](./awesome-transformers.md) list to discover incredible projects built with Transformers.

## Example Models

*   **Audio:**  Whisper, Moonshine, Wav2Vec2, Moshi, MusicGen, Bark
*   **Computer Vision:** SAM, DepthPro, DINO v2, SuperGlue, RT-DETRv2, VitPose, OneFormer, VideoMAE
*   **Multimodal:** Qwen2-Audio, LayoutLMv3, Qwen-VL, BLIP-2, GOT-OCR2, TAPAS, Emu3, Llava-OneVision, Llava, Kosmos-2
*   **NLP:** ModernBERT, Gemma, Mixtral, BART, T5, Llama, Qwen

## Citation

If you use this library, please cite the following paper:
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