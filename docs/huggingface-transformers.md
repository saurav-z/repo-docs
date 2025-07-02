# Transformers: State-of-the-Art Models for Natural Language Processing and More

**Unleash the power of advanced AI with the [Hugging Face Transformers](https://github.com/huggingface/transformers) library, your gateway to cutting-edge models for text, audio, vision, and multimodal applications.**

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


## Key Features

*   **State-of-the-art models:** Access a vast collection of pre-trained models for diverse tasks, including text generation, translation, image classification, and more.
*   **Unified API:** Simplify your workflow with a consistent API for all models, regardless of their architecture or task.
*   **Easy to Use:** With few user-facing abstractions, get up and running with minimal code.
*   **Modular & Customizable:** Adapt models to your specific needs with ease.
*   **Framework Agnostic:** Seamlessly move models between PyTorch, TensorFlow, and JAX.
*   **Huge Community:** Leverage the power of a vibrant community with over 1 million model checkpoints on the Hugging Face Hub.

## Installation

Install the Transformers library to start using the state-of-the-art models! Transformers supports Python 3.9+ with PyTorch, TensorFlow, or Flax.

**Choose your preferred framework**

```bash
# Install with PyTorch
pip install "transformers[torch]"

# Install with TensorFlow
pip install "transformers[tf]"

# Install with Flax
pip install "transformers[flax]"
```

Alternatively, install from source for the latest updates:

```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install .[torch] # or [tf] or [flax]
```

## Quickstart

Get started in minutes with the `pipeline()` API:

```python
from transformers import pipeline

# Text generation example
generator = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B")
print(generator("The secret to baking a really good cake is "))

# Chat example
chat = [
    {"role": "system", "content": "You are a sassy, wise-cracking robot."},
    {"role": "user", "content": "Hey, can you tell me any fun things to do in New York?"}
]
from transformers import pipeline
import torch

pipe = pipeline(task="text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
response = pipe(chat, max_new_tokens=512)
print(response[0]["generated_text"][-1]["content"])
```

**Explore more tasks:**
*   [Automatic Speech Recognition](https://huggingface.co/docs/transformers/pipeline_tutorial#automatic-speech-recognition)
*   [Image Classification](https://huggingface.co/docs/transformers/pipeline_tutorial#image-classification)
*   [Visual Question Answering](https://huggingface.co/docs/transformers/pipeline_tutorial#visual-question-answering)

## Why Use Transformers?

*   **Efficiency:** Leverage pre-trained models to reduce compute costs and carbon footprint.
*   **Community and Accessibility:** Explore a vast hub of pre-trained models and share your own.
*   **Flexibility:** Train, evaluate, and deploy models using your preferred framework.
*   **Customization:** Tailor models to your specific needs.

## 100+ Projects Built with Transformers
*   **[Awesome-transformers](https://github.com/huggingface/transformers/blob/main/awesome-transformers.md)**: Discover exciting projects built using the Transformers library.

## Example Models

Explore a wide array of pre-trained models, categorized by modality:

*   **Audio:** Whisper, Moonshine, Wav2Vec2, Moshi, MusicGen, Bark
*   **Computer Vision:** SAM, DepthPro, DINO v2, SuperGlue, RT-DETRv2, VitPose, OneFormer, VideoMAE
*   **Multimodal:** Qwen2-Audio, LayoutLMv3, Qwen-VL, BLIP-2, GOT-OCR2, TAPAS, Emu3, Llava-OneVision, Llava, Kosmos-2
*   **NLP:** ModernBERT, Gemma, Mixtral, BART, T5, Llama, Qwen

## Citation

If you use the Transformers library in your research, please cite our paper:

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