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

## Hugging Face Transformers: State-of-the-Art Models for NLP, Computer Vision, and More

**Harness the power of cutting-edge AI with Hugging Face Transformers, your go-to library for pre-trained models for text, vision, audio, and multimodal tasks.** ([See the original repo](https://github.com/huggingface/transformers))

<h3 align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_as_a_model_definition.png"/>
</h3>

This library provides a central framework for defining and using state-of-the-art machine learning models, ensuring compatibility across various training frameworks, inference engines, and related libraries.  With over 1M+ pre-trained model checkpoints available on the [Hugging Face Hub](https://huggingface.co/models), you can easily find and deploy models for various tasks.

**Key Features:**

*   **Extensive Model Support:** Access a vast collection of pre-trained models for various modalities: text, computer vision, audio, video, and multimodal applications.
*   **Simplified Usage:** Leverage a unified API for easy integration and deployment.
*   **Framework Agnostic:** Compatible with major training frameworks like PyTorch, TensorFlow, and Flax.
*   **Cost-Effective:** Utilize pre-trained models to reduce compute costs and accelerate development.
*   **Customization:** Easily fine-tune and adapt models to your specific needs.
*   **Active Community:** Benefit from a vibrant community and contribute to the development of new models.

## Installation

Choose your preferred installation method for Transformers:

*   **Requirements**:  Python 3.9+ with PyTorch 2.1+, TensorFlow 2.6+, or Flax 0.4.1+.

    1.  **Virtual Environment**:  Create and activate a virtual environment using [venv](https://docs.python.org/3/library/venv.html) or [uv](https://docs.astral.sh/uv/).

    ```bash
    # venv
    python -m venv .my-env
    source .my-env/bin/activate
    # uv
    uv venv .my-env
    source .my-env/bin/activate
    ```

    2.  **Install Transformers**:  Use pip or uv to install the library.

    ```bash
    # pip
    pip install "transformers[torch]"

    # uv
    uv pip install "transformers[torch]"
    ```
*   **Install from Source**: Install the latest changes, noting that the *latest* version may not be stable.

    ```bash
    git clone https://github.com/huggingface/transformers.git
    cd transformers

    # pip
    pip install .[torch]

    # uv
    uv pip install .[torch]
    ```

## Quickstart

Get up and running quickly with the `Pipeline` API, a high-level interface for text, audio, vision, and multimodal tasks.

```python
from transformers import pipeline

# Text Generation
pipeline = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B")
print(pipeline("the secret to baking a really good cake is "))

# Chat with a Model
import torch
chat = [
    {"role": "system", "content": "You are a sassy, wise-cracking robot as imagined by Hollywood circa 1986."},
    {"role": "user", "content": "Hey, can you tell me any fun things to do in New York?"}
]
pipeline = pipeline(task="text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
response = pipeline(chat, max_new_tokens=512)
print(response[0]["generated_text"][-1]["content"])
```

Explore how `Pipeline` works for different modalities and tasks with the examples provided in the original documentation.

## Why Use Transformers?

*   **Ease of Use:** Simple, state-of-the-art models are readily available.
*   **Efficiency:**  Reduce compute costs by leveraging pre-trained models.
*   **Flexibility:**  Choose the right framework for training, evaluation, and production.
*   **Customization:** Tailor models and examples to fit your specific requirements.

## Considerations

*   **Not a Modular Toolbox:** This library is designed for model definition, not modular building blocks.
*   **Training API:** The training API is optimized for PyTorch models provided by Transformers.
*   **Example Scripts:**  Adapt example scripts to your use case.

## Community & Projects

Transformers powers a thriving ecosystem! Explore the [awesome-transformers](./awesome-transformers.md) page which lists numerous community projects built with Transformers.  Contribute your project by opening a PR.

## Example Models

Find example models for audio, computer vision, multimodal tasks, and NLP. Explore the [Hub model pages](https://huggingface.co/models) to test models directly.

### Audio
- Audio classification with [Whisper](https://huggingface.co/openai/whisper-large-v3-turbo)
- Automatic speech recognition with [Moonshine](https://huggingface.co/UsefulSensors/moonshine)
- Keyword spotting with [Wav2Vec2](https://huggingface.co/superb/wav2vec2-base-superb-ks)
- Speech to speech generation with [Moshi](https://huggingface.co/kyutai/moshiko-pytorch-bf16)
- Text to audio with [MusicGen](https://huggingface.co/facebook/musicgen-large)
- Text to speech with [Bark](https://huggingface.co/suno/bark)

### Computer Vision
- Automatic mask generation with [SAM](https://huggingface.co/facebook/sam-vit-base)
- Depth estimation with [DepthPro](https://huggingface.co/apple/DepthPro-hf)
- Image classification with [DINO v2](https://huggingface.co/facebook/dinov2-base)
- Keypoint detection with [SuperGlue](https://huggingface.co/magic-leap-community/superglue_outdoor)
- Keypoint matching with [SuperGlue](https://huggingface.co/magic-leap-community/superglue)
- Object detection with [RT-DETRv2](https://huggingface.co/PekingU/rtdetr_v2_r50vd)
- Pose Estimation with [VitPose](https://huggingface.co/usyd-community/vitpose-base-simple)
- Universal segmentation with [OneFormer](https://huggingface.co/shi-labs/oneformer_ade20k_swin_large)
- Video classification with [VideoMAE](https://huggingface.co/MCG-NJU/videomae-large)

### Multimodal
- Audio or text to text with [Qwen2-Audio](https://huggingface.co/Qwen/Qwen2-Audio-7B)
- Document question answering with [LayoutLMv3](https://huggingface.co/microsoft/layoutlmv3-base)
- Image or text to text with [Qwen-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- Image captioning [BLIP-2](https://huggingface.co/Salesforce/blip2-opt-2.7b)
- OCR-based document understanding with [GOT-OCR2](https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf)
- Table question answering with [TAPAS](https://huggingface.co/google/tapas-base)
- Unified multimodal understanding and generation with [Emu3](https://huggingface.co/BAAI/Emu3-Gen)
- Vision to text with [Llava-OneVision](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf)
- Visual question answering with [Llava](https://huggingface.co/llava-hf/llava-1.5-7b-hf)
- Visual referring expression segmentation with [Kosmos-2](https://huggingface.co/microsoft/kosmos-2-patch14-224)

### NLP
- Masked word completion with [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base)
- Named entity recognition with [Gemma](https://huggingface.co/google/gemma-2-2b)
- Question answering with [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)
- Summarization with [BART](https://huggingface.co/facebook/bart-large-cnn)
- Translation with [T5](https://huggingface.co/google-t5/t5-base)
- Text generation with [Llama](https://huggingface.co/meta-llama/Llama-3.2-1B)
- Text classification with [Qwen](https://huggingface.co/Qwen/Qwen2.5-0.5B)

## Citation

If you use the 🤗 Transformers library, please cite the following paper:

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