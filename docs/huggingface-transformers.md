```markdown
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

**Get started today with the leading library for cutting-edge machine learning models and unlock the power of AI across text, vision, audio, and multimodal tasks!** Explore the [Hugging Face Transformers repository](https://github.com/huggingface/transformers) to discover and utilize pre-trained models for inference and training.

## Key Features:

*   **Wide Range of Models:** Access over 1 million pre-trained transformer models for various tasks including text generation, image classification, audio processing, video analysis, and multimodal applications, available on the [Hugging Face Hub](https://huggingface.co/models).
*   **Unified API:** Utilize a consistent and easy-to-use API to interact with diverse model architectures.
*   **Simplified Customization:** Easily adapt existing models to your unique needs.
*   **Cross-Framework Compatibility:** Seamlessly move models between PyTorch, TensorFlow, and JAX.
*   **Cost and Carbon Footprint Reduction:** Leverage pre-trained models to lower compute requirements and promote sustainability.
*   **Comprehensive Community:** Benefit from a thriving ecosystem of projects and resources built around Transformers.

## Installation

To get started, install the Transformers library using pip or uv.

**Prerequisites:** Python 3.9+, PyTorch 2.1+, TensorFlow 2.6+, or Flax 0.4.1+

```bash
# Install with pip
pip install "transformers[torch]"
```

```bash
# Install with uv
uv pip install "transformers[torch]"
```

Or, install from source for the latest updates:

```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install .[torch]
```

## Quickstart

The `pipeline` API is your gateway to quick model usage. Below are some quickstart examples:
```python
from transformers import pipeline

# Text generation
generator = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B")
print(generator("the secret to baking a really good cake is ")[0]["generated_text"])
```
```python
from transformers import pipeline
import torch

# Chatbot example
chat = [
    {"role": "system", "content": "You are a sassy, wise-cracking robot as imagined by Hollywood circa 1986."},
    {"role": "user", "content": "Hey, can you tell me any fun things to do in New York?"}
]
pipeline = pipeline(task="text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
response = pipeline(chat, max_new_tokens=512)
print(response[0]["generated_text"][-1]["content"])
```

Explore examples for image classification, audio recognition, and other tasks to see the versatility of the library.

## Why Use Transformers?

*   **Simplified Usage:** Access state-of-the-art models with ease and high performance.
*   **Efficiency:** Benefit from pre-trained models to reduce computational costs and environmental impact.
*   **Flexibility:** Choose the right framework for your training, evaluation, and production needs.
*   **Customization:** Adapt models and examples to your specific requirements.
*   **Extensive Resources:** Leverage a vast community and a wealth of pre-trained models.

## When *Not* to Use Transformers:

*   When you need fine-grained control over model building blocks - this library focuses on ready-to-use models.
*   For generic machine learning loops, use libraries like [Accelerate](https://huggingface.co/docs/accelerate) instead.
*   When example scripts require extensive customization for your specific use case.

## Community & Projects

Transformers is a central hub for a rich ecosystem of projects on the [Hugging Face Hub](https://huggingface.co/models).  Explore the [awesome-transformers](./awesome-transformers.md) page to discover incredible projects built with Transformers.

## Example Models

Discover a wide range of models for various modalities on the [Hub model pages](https://huggingface.co/models).

<details>
<summary>Audio</summary>

*   [Whisper](https://huggingface.co/openai/whisper-large-v3-turbo)
*   [Moonshine](https://huggingface.co/UsefulSensors/moonshine)
*   [Wav2Vec2](https://huggingface.co/superb/wav2vec2-base-superb-ks)
*   [Moshi](https://huggingface.co/kyutai/moshiko-pytorch-bf16)
*   [MusicGen](https://huggingface.co/facebook/musicgen-large)
*   [Bark](https://huggingface.co/suno/bark)

</details>

<details>
<summary>Computer vision</summary>

*   [SAM](https://huggingface.co/facebook/sam-vit-base)
*   [DepthPro](https://huggingface.co/apple/DepthPro-hf)
*   [DINO v2](https://huggingface.co/facebook/dinov2-base)
*   [SuperGlue](https://huggingface.co/magic-leap-community/superglue_outdoor)
*   [SuperGlue](https://huggingface.co/magic-leap-community/superglue)
*   [RT-DETRv2](https://huggingface.co/PekingU/rtdetr_v2_r50vd)
*   [VitPose](https://huggingface.co/usyd-community/vitpose-base-simple)
*   [OneFormer](https://huggingface.co/shi-labs/oneformer_ade20k_swin_large)
*   [VideoMAE](https://huggingface.co/MCG-NJU/videomae-large)

</details>

<details>
<summary>Multimodal</summary>

*   [Qwen2-Audio](https://huggingface.co/Qwen/Qwen2-Audio-7B)
*   [LayoutLMv3](https://huggingface.co/microsoft/layoutlmv3-base)
*   [Qwen-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
*   [BLIP-2](https://huggingface.co/Salesforce/blip2-opt-2.7b)
*   [GOT-OCR2](https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf)
*   [TAPAS](https://huggingface.co/google/tapas-base)
*   [Emu3](https://huggingface.co/BAAI/Emu3-Gen)
*   [Llava-OneVision](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf)
*   [Llava](https://huggingface.co/llava-hf/llava-1.5-7b-hf)
*   [Kosmos-2](https://huggingface.co/microsoft/kosmos-2-patch14-224)

</details>

<details>
<summary>NLP</summary>

*   [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base)
*   [Gemma](https://huggingface.co/google/gemma-2-2b)
*   [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)
*   [BART](https://huggingface.co/facebook/bart-large-cnn)
*   [T5](https://huggingface.co/google-t5/t5-base)
*   [Llama](https://huggingface.co/meta-llama/Llama-3.2-1B)
*   [Qwen](https://huggingface.co/Qwen/Qwen2.5-0.5B)

</details>

## Citation

If you use this library, please cite the following:

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
```
