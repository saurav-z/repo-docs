# Transformers: State-of-the-Art Models for NLP, Computer Vision, and More

**Quickly and easily use cutting-edge pre-trained models for inference and training with the Hugging Face Transformers library ([original repo](https://github.com/huggingface/transformers)).**

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-light.svg">
    <img alt="Hugging Face Transformers Library" src="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-light.svg" width="352" height="59" style="max-width: 100%;">
  </picture>
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

## Key Features

*   **Simplified Model Access:** Easily access and utilize state-of-the-art pre-trained models across text, vision, audio, video, and multimodal tasks.
*   **Extensive Model Support:** Leverage over 1 million pre-trained model checkpoints available on the Hugging Face Hub.
*   **Framework Compatibility:**  Seamlessly integrate models with popular frameworks like PyTorch, TensorFlow, and Flax.
*   **Ease of Use:**  Benefit from a user-friendly API with a unified approach, including the `Pipeline` API for quick experimentation.
*   **Customization and Flexibility:**  Adapt models and examples to fit your specific needs with accessible model internals.
*   **Community-Driven Ecosystem:** Engage with a thriving community of projects built around Transformers.

### Introduction
Transformers is a powerful library by Hugging Face that provides access to state-of-the-art pre-trained models for various machine learning tasks. It acts as the go-to framework for defining modern machine learning models, ensuring compatibility across various training frameworks, inference engines, and related libraries.

<h3 align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_as_a_model_definition.png"/>
</h3>

## Installation

Ensure you have Python 3.9+ and install the necessary deep learning framework (PyTorch, TensorFlow, or Flax) and the Transformers library:

```bash
# venv
python -m venv .my-env
source .my-env/bin/activate
# uv
uv venv .my-env
source .my-env/bin/activate
# pip
pip install "transformers[torch]"
# uv
uv pip install "transformers[torch]"

```

## Quickstart

Use the `Pipeline` API for quick inference:

```python
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B")
pipeline("the secret to baking a really good cake is ")
```
[More Quickstart Examples](https://huggingface.co/docs/transformers/pipeline_tutorial)

## Why Choose Transformers?

*   **Accessibility:**  Easy to use, enabling developers of all levels.
*   **Efficiency:**  Reduce compute costs and carbon footprint with pre-trained models.
*   **Flexibility:**  Choose the best framework for training, evaluation, and production.
*   **Customization:**  Adapt models to your specific tasks with the available resources and examples.

## Why *Not* Use Transformers?

*   This library is *not* a general modular toolbox.
*   The training API is optimized for Transformers' PyTorch models.
*   Example scripts may need to be adapted for specific use cases.

## Explore the Ecosystem

Discover a vast community of projects built with Transformers in the [awesome-transformers](./awesome-transformers.md) list.

## Example Models

Explore pre-trained models for various modalities:

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
*   Keypoint detection with [SuperPoint](https://huggingface.co/magic-leap-community/superpoint)
*   Keypoint matching with [SuperGlue](https://huggingface.co/magic-leap-community/superglue_outdoor)
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