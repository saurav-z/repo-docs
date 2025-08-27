<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-light.svg">
  <img alt="Hugging Face Transformers Library" src="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-light.svg" width="352" height="59" style="max-width: 100%;">
</picture>
<br/>
<br/>

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

# Hugging Face Transformers: State-of-the-Art Models for NLP, Computer Vision, and More

**Leverage cutting-edge pretrained models to effortlessly perform tasks in text, computer vision, audio, video, and multimodal applications with the Hugging Face Transformers library.** ([See the original repo](https://github.com/huggingface/transformers)).

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_as_a_model_definition.png" alt="Transformers Architecture Overview">

**Key Features:**

*   **Extensive Model Support:** Access a vast collection of pre-trained models for various tasks and modalities.
*   **Unified API:** Simplify your workflow with a consistent API for all models.
*   **Ease of Use:** Get started quickly with high-level APIs like `pipeline`.
*   **Cross-Framework Compatibility:** Seamlessly move models between PyTorch, TensorFlow, and JAX.
*   **Community & Resources:** Benefit from a large community, comprehensive documentation, and example scripts.
*   **Cost-Effective:** Save time and resources by utilizing pre-trained models.

## Installation

Install `transformers` using pip or uv.  Be sure to install the appropriate dependencies for your chosen framework (PyTorch, TensorFlow, or Flax).

```bash
# pip
pip install "transformers[torch]"

# uv
uv pip install "transformers[torch]"
```

## Quickstart

The `Pipeline` API offers an easy entry point to start using `transformers`.

```python
from transformers import pipeline

# Text Generation Example
generator = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B")
result = generator("The secret to baking a really good cake is ")
print(result[0]["generated_text"])
```

```python
import torch
from transformers import pipeline

chat = [
    {"role": "system", "content": "You are a sassy, wise-cracking robot as imagined by Hollywood circa 1986."},
    {"role": "user", "content": "Hey, can you tell me any fun things to do in New York?"}
]

pipeline = pipeline(task="text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", dtype=torch.bfloat16, device_map="auto")
response = pipeline(chat, max_new_tokens=512)
print(response[0]["generated_text"][-1]["content"])
```

Explore examples for other modalities:

<details>
<summary>Automatic Speech Recognition</summary>

```python
from transformers import pipeline

asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3")
result = asr("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
print(result["text"])
```
</details>

<details>
<summary>Image Classification</summary>

<img src="https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png" alt="Parrots Image">

```python
from transformers import pipeline

image_classifier = pipeline("image-classification", model="facebook/dinov2-small-imagenet1k-1-layer")
result = image_classifier("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
print(result)
```
</details>

<details>
<summary>Visual Question Answering</summary>

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-few-shot.jpg" alt="Image for VQA">

```python
from transformers import pipeline

vqa = pipeline("visual-question-answering", model="Salesforce/blip-vqa-base")
result = vqa(
    image="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-few-shot.jpg",
    question="What is in the image?",
)
print(result)
```
</details>

## Why Use Transformers?

*   **State-of-the-Art Models:** Access the latest advancements in AI.
*   **Reduced Compute Costs:** Utilize pre-trained models and avoid training from scratch.
*   **Framework Flexibility:** Train, evaluate, and deploy models in your preferred framework.
*   **Customization:** Adapt models to your specific needs with ease.

## When to Avoid Transformers

*   Not a modular toolbox for building neural networks.
*   Training API is primarily for PyTorch models.
*   Example scripts may need adaptation for specific use cases.

## Explore Models

Discover over 1 million pre-trained models on the [Hugging Face Hub](https://huggingface.co/models).

Example model categories:

*   **Audio:** Whisper, Moonshine, MusicGen, Bark
*   **Computer Vision:** SAM, DINOv2, RT-DETRv2, VitPose
*   **Multimodal:** Qwen2-Audio, LayoutLMv3, BLIP-2, Llava
*   **NLP:** Gemma, Mixtral, BART, T5, Llama, Qwen

## Citation

If you use the Transformers library, please cite our paper:

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