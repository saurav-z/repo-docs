<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-light.svg">
  <img alt="Hugging Face Transformers Library" src="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-light.svg" width="352" height="59" style="max-width: 100%;">
</picture>
<br/>
<br/>

[![Checkpoints on Hub](https://img.shields.io/endpoint?url=https://huggingface.co/api/shields/models&color=brightgreen)](https://huggingface.co/models)
[![Build Status](https://img.shields.io/circleci/build/github/huggingface/transformers/main)](https://circleci.com/gh/huggingface/transformers)
[![License](https://img.shields.io/github/license/huggingface/transformers.svg?color=blue)](https://github.com/huggingface/transformers/blob/main/LICENSE)
[![Documentation](https://img.shields.io/website/http/huggingface.co/docs/transformers/index.svg?down_color=red&down_message=offline&up_message=online)](https://huggingface.co/docs/transformers/index)
[![Release](https://img.shields.io/github/release/huggingface/transformers.svg)](https://github.com/huggingface/transformers/releases)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg)](https://github.com/huggingface/transformers/blob/main/CODE_OF_CONDUCT.md)
[![DOI](https://zenodo.org/badge/latestdoi/155220641.svg)](https://zenodo.org/badge/latestdoi/155220641)

# Transformers: State-of-the-Art Models for NLP, Computer Vision, and More

**Unleash the power of cutting-edge AI with the Hugging Face Transformers library, your gateway to pre-trained models for a wide range of tasks.**

**[Explore the original repo](https://github.com/huggingface/transformers)**

<h4 align="center">
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
</h4>

## Key Features:

*   **State-of-the-Art Models:** Access a vast collection of pre-trained models for text, computer vision, audio, video, and multimodal tasks.
*   **Easy-to-Use API:** Simplified interface for both inference and training, making it accessible for everyone.
*   **Model Hub Integration:** Seamlessly integrate with the Hugging Face Hub, boasting over 1 million pre-trained model checkpoints.
*   **Framework Agnostic:** Compatible with major deep learning frameworks, including PyTorch, TensorFlow, and Flax.
*   **Customization and Flexibility:** Easily adapt models and examples to your specific needs.
*   **Community-Driven:** Benefit from a vibrant community and extensive resources.

## Installation

Install Transformers with your preferred package manager.

```bash
pip install "transformers[torch]"
```
or
```bash
uv pip install "transformers[torch]"
```

For the latest changes install from source:

```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install .[torch] # or uv pip install .[torch]
```

## Quickstart

Use the `Pipeline` API for easy access to different AI tasks.

```python
from transformers import pipeline

# Text generation
pipeline = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B")
pipeline("the secret to baking a really good cake is ")
```

Expand the examples to see how `Pipeline` works for different modalities and tasks.

<details>
<summary>Automatic speech recognition</summary>

```py
from transformers import pipeline

pipeline = pipeline(task="automatic-speech-recognition", model="openai/whisper-large-v3")
pipeline("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
{'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.'}
```

</details>

<details>
<summary>Image classification</summary>

<h3 align="center">
    <a><img src="https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"></a>
</h3>

```py
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="facebook/dinov2-small-imagenet1k-1-layer")
pipeline("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
[{'label': 'macaw', 'score': 0.997848391532898},
 {'label': 'sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita',
  'score': 0.0016551691805943847},
 {'label': 'lorikeet', 'score': 0.00018523589824326336},
 {'label': 'African grey, African gray, Psittacus erithacus',
  'score': 7.85409429227002e-05},
 {'label': 'quail', 'score': 5.502637941390276e-05}]
```

</details>

<details>
<summary>Visual question answering</summary>


<h3 align="center">
    <a><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-few-shot.jpg"></a>
</h3>

```py
from transformers import pipeline

pipeline = pipeline(task="visual-question-answering", model="Salesforce/blip-vqa-base")
pipeline(
    image="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-few-shot.jpg",
    question="What is in the image?",
)
[{'answer': 'statue of liberty'}]
```

</details>

## Why Use Transformers?

*   **Simplified AI:** Easy to use models, low barrier to entry, simple abstractions.
*   **Efficiency:** Leverage pre-trained models and reduce compute costs.
*   **Flexibility:** Train, evaluate, and deploy models across different frameworks.
*   **Customization:** Adapt existing models or examples for your needs.

## Why *Not* Use Transformers?

*   **Not a Modular Toolbox:**  This library is not a modular toolbox of building blocks for neural nets.
*   **Training API:** The training API is optimized to work with PyTorch models provided by Transformers. For generic machine learning loops, you should use another library like [Accelerate](https://huggingface.co/docs/accelerate).
*   **Example Scripts:**  Example scripts are only *examples*. They may not necessarily work out-of-the-box on your specific use case.

## Projects Built with Transformers

Explore the [awesome-transformers](./awesome-transformers.md) page to discover 100+ incredible projects built with Transformers.

## Example Models

*   **Audio:**  Whisper, Moonshine, Wav2Vec2, Moshi, MusicGen, Bark
*   **Computer Vision:** SAM, DepthPro, DINO v2, SuperPoint, SuperGlue, RT-DETRv2, VitPose, OneFormer, VideoMAE
*   **Multimodal:** Qwen2-Audio, LayoutLMv3, Qwen-VL, BLIP-2, GOT-OCR2, TAPAS, Emu3, Llava-OneVision, Llava, Kosmos-2
*   **NLP:** ModernBERT, Gemma, Mixtral, BART, T5, Llama, Qwen

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