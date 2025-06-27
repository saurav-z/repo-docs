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

# Transformers: State-of-the-Art Models for Natural Language Processing and More

**Harness the power of cutting-edge machine learning with the Transformers library, enabling you to easily build, train, and deploy state-of-the-art models for various tasks.**

[![Checkpoints on Hub](https://img.shields.io/endpoint?url=https://huggingface.co/api/shields/models&color=brightgreen)](https://huggingface.co/models)
[![Build Status](https://img.shields.io/circleci/build/github/huggingface/transformers/main)](https://circleci.com/gh/huggingface/transformers)
[![License](https://img.shields.io/github/license/huggingface/transformers.svg?color=blue)](https://github.com/huggingface/transformers/blob/main/LICENSE)
[![Documentation](https://img.shields.io/website/http/huggingface.co/docs/transformers/index.svg?down_color=red&down_message=offline&up_message=online)](https://huggingface.co/docs/transformers/index)
[![Release](https://img.shields.io/github/release/huggingface/transformers.svg)](https://github.com/huggingface/transformers/releases)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg)](https://github.com/huggingface/transformers/blob/main/CODE_OF_CONDUCT.md)
[![DOI](https://zenodo.org/badge/latestdoi/155220641.svg)](https://zenodo.org/badge/latestdoi/155220641)

[View the original repository on GitHub](https://github.com/huggingface/transformers)

## Key Features

*   **Wide Range of Models:** Access over 1 million pre-trained model checkpoints for text, computer vision, audio, video, and multimodal tasks.
*   **Unified API:**  Use a consistent and intuitive API for both inference and training across all supported models.
*   **Framework Agnostic:** Train and deploy models with PyTorch, TensorFlow, and Flax, and easily switch between frameworks.
*   **Ease of Use:** Quickly get started with high-level APIs like `pipeline` or dive deep with customizable components.
*   **Community Driven:** Benefit from a thriving community and an extensive library of examples and resources.

## What is Transformers?

Transformers is a comprehensive library providing state-of-the-art pre-trained models that serve as the foundation for advanced machine learning applications. It centralizes model definitions, ensuring compatibility across various training frameworks, inference engines, and supporting libraries. This enables simple, customizable, and efficient model usage.

## Installation

Transformers requires Python 3.9+ and supports PyTorch 2.1+, TensorFlow 2.6+, and Flax 0.4.1+.  Choose your preferred package manager and environment setup:

**Using `venv`:**
```bash
python -m venv .my-env
source .my-env/bin/activate
pip install "transformers[torch]"  # or use tensorflow, flax as needed
```

**Using `uv` (faster):**
```bash
uv venv .my-env
source .my-env/bin/activate
uv pip install "transformers[torch]"
```

To install from source for the latest updates (may be unstable):
```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install .[torch] # or the relevant framework
```

## Quickstart with the Pipeline API

The `pipeline` API offers a straightforward way to get started with Transformers.  It handles pre-processing and post-processing for various tasks.

```python
from transformers import pipeline

# Text Generation example:
generator = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B")
print(generator("the secret to baking a really good cake is ")[0]['generated_text'])

# Chatbot example:
import torch
chat_pipeline = pipeline(task="text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
chat_history = [{"role": "system", "content": "You are a sassy robot."},
                {"role": "user", "content": "Tell me a joke."}]
response = chat_pipeline(chat_history, max_new_tokens=512)
print(response[0]["generated_text"][-1]["content"])
```
  
<details>
<summary>Expand for more examples</summary>

```python
# Automatic Speech Recognition
from transformers import pipeline
asr_pipeline = pipeline(task="automatic-speech-recognition", model="openai/whisper-large-v3")
print(asr_pipeline("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")['text'])

# Image Classification
from transformers import pipeline
img_pipeline = pipeline(task="image-classification", model="facebook/dinov2-small-imagenet1k-1-layer")
print(img_pipeline("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"))

# Visual Question Answering
from transformers import pipeline
vqa_pipeline = pipeline(task="visual-question-answering", model="Salesforce/blip-vqa-base")
print(vqa_pipeline(image="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-few-shot.jpg",
        question="What is in the image?"))
```
</details>

## Why Use Transformers?

*   **State-of-the-Art Models:** Leverage high-performing models for various tasks.
*   **Reduced Costs:** Utilize pre-trained models to save time and resources.
*   **Framework Flexibility:** Train and deploy models across multiple frameworks.
*   **Customization:** Easily tailor models to your specific needs.

## When Not to Use Transformers

*   Not a modular toolbox for creating neural networks from scratch.
*   The training API is optimized for Transformers-provided models.
*   Example scripts may require adaptation for your use case.

## Projects Powered by Transformers

Explore the [awesome-transformers](./awesome-transformers.md) page to see the many amazing projects built using this library.

## Example Models

A vast array of example models is readily available on the Hugging Face Hub.  Here are a few examples:

<details>
<summary>Audio</summary>

*   Audio classification with [Whisper](https://huggingface.co/openai/whisper-large-v3-turbo)
*   Automatic speech recognition with [Moonshine](https://huggingface.co/UsefulSensors/moonshine)
*   Keyword spotting with [Wav2Vec2](https://huggingface.co/superb/wav2vec2-base-superb-ks)
*   Speech to speech generation with [Moshi](https://huggingface.co/kyutai/moshiko-pytorch-bf16)
*   Text to audio with [MusicGen](https://huggingface.co/facebook/musicgen-large)
*   Text to speech with [Bark](https://huggingface.co/suno/bark)

</details>

<details>
<summary>Computer vision</summary>

*   Automatic mask generation with [SAM](https://huggingface.co/facebook/sam-vit-base)
*   Depth estimation with [DepthPro](https://huggingface.co/apple/DepthPro-hf)
*   Image classification with [DINO v2](https://huggingface.co/facebook/dinov2-base)
*   Keypoint detection with [SuperGlue](https://huggingface.co/magic-leap-community/superglue_outdoor)
*   Keypoint matching with [SuperGlue](https://huggingface.co/magic-leap-community/superglue)
*   Object detection with [RT-DETRv2](https://huggingface.co/PekingU/rtdetr_v2_r50vd)
*   Pose Estimation with [VitPose](https://huggingface.co/usyd-community/vitpose-base-simple)
*   Universal segmentation with [OneFormer](https://huggingface.co/shi-labs/oneformer_ade20k_swin_large)
*   Video classification with [VideoMAE](https://huggingface.co/MCG-NJU/videomae-large)

</details>

<details>
<summary>Multimodal</summary>

*   Audio or text to text with [Qwen2-Audio](https://huggingface.co/Qwen/Qwen2-Audio-7B)
*   Document question answering with [LayoutLMv3](https://huggingface.co/microsoft/layoutlmv3-base)
*   Image or text to text with [Qwen-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
*   Image captioning [BLIP-2](https://huggingface.co/Salesforce/blip2-opt-2.7b)
*   OCR-based document understanding with [GOT-OCR2](https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf)
*   Table question answering with [TAPAS](https://huggingface.co/google/tapas-base)
*   Unified multimodal understanding and generation with [Emu3](https://huggingface.co/BAAI/Emu3-Gen)
*   Vision to text with [Llava-OneVision](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf)
*   Visual question answering with [Llava](https://huggingface.co/llava-hf/llava-1.5-7b-hf)
*   Visual referring expression segmentation with [Kosmos-2](https://huggingface.co/microsoft/kosmos-2-patch14-224)

</details>

<details>
<summary>NLP</summary>

*   Masked word completion with [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base)
*   Named entity recognition with [Gemma](https://huggingface.co/google/gemma-2-2b)
*   Question answering with [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)
*   Summarization with [BART](https://huggingface.co/facebook/bart-large-cnn)
*   Translation with [T5](https://huggingface.co/google-t5/t5-base)
*   Text generation with [Llama](https://huggingface.co/meta-llama/Llama-3.2-1B)
*   Text classification with [Qwen](https://huggingface.co/Qwen/Qwen2.5-0.5B)

</details>

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