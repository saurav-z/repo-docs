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

# Hugging Face Transformers: State-of-the-Art NLP for Everyone

**Unlock the power of cutting-edge natural language processing, computer vision, audio, video, and multimodal models with Hugging Face Transformers, a comprehensive library for both inference and training.**  Get started now with the [official repository](https://github.com/huggingface/transformers)!

## Key Features:

*   **Wide Range of Models:** Access a vast library of pre-trained models for various tasks, including text generation, image classification, audio processing, and more.
*   **Easy-to-Use API:**  Simplify your workflow with a unified API for all models, regardless of their architecture, using just three core classes.
*   **Reduced Costs:** Leverage pre-trained models to decrease compute time, production costs, and carbon footprint.
*   **Framework Flexibility:**  Train, evaluate, and deploy models using your preferred frameworks (PyTorch, TensorFlow, JAX/Flax), facilitating seamless transitions between training and production environments.
*   **Customization Options:** Tailor models to your specific needs with comprehensive examples and readily accessible model internals.
*   **Extensive Ecosystem:** Benefit from a vibrant community, including the [Hugging Face Hub](https://huggingface.co/models), with over 1 million model checkpoints and numerous projects built on top of Transformers.

## Installation

Get started by installing Transformers with your preferred package manager:

```bash
# pip
pip install "transformers[torch]"

# uv
uv pip install "transformers[torch]"
```

For detailed instructions, including environment setup and installing from source, refer to the [original repository](https://github.com/huggingface/transformers).

## Quickstart

The `pipeline` API offers a simple way to begin with text, audio, vision, and multimodal tasks:

```python
from transformers import pipeline

generator = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B")
print(generator("the secret to baking a really good cake is "))
```

Explore the documentation and examples within the original repository for advanced usage and task-specific implementations.

### Example tasks:

*   **Automatic Speech Recognition:**  Convert audio to text.
*   **Image Classification:** Identify the content of an image.
*   **Visual Question Answering:** Answer questions about an image.

## Why Use Transformers?

*   **Accessibility:** High-performance models with low barriers to entry.
*   **Efficiency:** Share pre-trained models to reduce computational costs.
*   **Flexibility:** Choose the right framework for each stage of the model lifecycle.
*   **Customization:** Adapt models and examples to meet your specific requirements.

## Why Not Use Transformers?

*   This library is not a modular toolbox for constructing neural nets.
*   Training loops should be handled by other libraries like [Accelerate](https://huggingface.co/docs/accelerate).
*   Example scripts require adaptation for specific use cases.

## 100+ Projects Using Transformers

Discover exciting projects built on top of Transformers and contribute your own to the [awesome-transformers](https://github.com/huggingface/transformers/blob/main/awesome-transformers.md) page!

## Model Examples

Test out various models for audio, computer vision, multimodal, and NLP tasks by browsing their respective [Hub model pages](https://huggingface.co/models).

*   **Audio:**  Whisper, Moonshine, Wav2Vec2, Moshi, MusicGen, Bark
*   **Computer Vision:** SAM, DepthPro, DINO v2, SuperPoint, SuperGlue, RT-DETRv2, VitPose, OneFormer, VideoMAE
*   **Multimodal:** Qwen2-Audio, LayoutLMv3, Qwen-VL, BLIP-2, GOT-OCR2, TAPAS, Emu3, Llava-OneVision, Llava, Kosmos-2
*   **NLP:** ModernBERT, Gemma, Mixtral, BART, T5, Llama, Qwen

## Citation

If you're using the Transformers library in your research, please cite our paper:

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