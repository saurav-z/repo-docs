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

**Hugging Face Transformers is a leading library providing cutting-edge, pre-trained models for a wide range of AI tasks, enabling developers to easily implement advanced machine learning solutions.**

### Key Features

*   **Extensive Model Library:** Access a vast collection of pre-trained models for tasks in natural language processing (NLP), computer vision, audio, and multimodal applications.
*   **Ease of Use:** Simplify model integration with an intuitive API, making it easy for both beginners and experts to utilize state-of-the-art models.
*   **Unified API:** Leverage a consistent API for all models, simplifying the development process and reducing the learning curve.
*   **Framework Flexibility:** Train and deploy models across PyTorch, TensorFlow, and Flax, offering maximum flexibility for your projects.
*   **Cost-Effective Solutions:** Utilize pre-trained models to reduce training time and computing costs, enabling faster development cycles.
*   **Customization & Community:** Customize models to suit specific needs, and benefit from a robust community with examples and resources.

### Quick Start

Get started with Hugging Face Transformers by installing it.

```bash
pip install "transformers[torch]" # For PyTorch support
```

or

```bash
pip install "transformers[tensorflow]" # For TensorFlow support
```

For the latest changes, install from source:
```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install .[torch] # or tensorflow
```

Use the `Pipeline` API for tasks like text generation, audio transcription, and image classification:

```python
from transformers import pipeline

# Text Generation
generator = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B")
print(generator("The secret to baking a really good cake is "))
```

```python
# Automatic Speech Recognition
asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3")
print(asr("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"))
```

```python
# Image Classification
classifier = pipeline("image-classification", model="facebook/dinov2-small-imagenet1k-1-layer")
print(classifier("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"))
```

*For more advanced usages, see the documentation for more complex models.*

### Why Use Transformers?

*   **Simplified AI:** Easy-to-use models for a wide range of tasks.
*   **Efficiency:** Reduce costs by using pre-trained models.
*   **Framework Agnostic:** Supports multiple frameworks.
*   **Customization:** Adapt models to your specific needs.

### Why Not Use Transformers?

*   **Modular Design:** This library is optimized for state-of-the-art models and is not designed to be a modular toolbox for building blocks.
*   **Training API Focus:** The training API is optimized for PyTorch models provided by Transformers. For generic machine learning loops, use libraries like [Accelerate](https://huggingface.co/docs/accelerate).
*   **Example Scripts:** The example scripts are for *examples* and may need to be adapted to your specific use case.

### Explore More

*   **Model Checkpoints:** Explore over 1M+ [model checkpoints](https://huggingface.co/models?library=transformers&sort=trending) on the [Hugging Face Hub](https://huggingface.com/models).
*   **Community Projects:** Discover 100+ incredible projects built with Transformers in our [awesome-transformers](./awesome-transformers.md) page.
*   **Example Models:**

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

### Citation

If you use the library, please cite the following paper:
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

[Go back to the original repository](https://github.com/huggingface/transformers)