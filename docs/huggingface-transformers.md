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

# Hugging Face Transformers: State-of-the-Art NLP & Beyond

**Hugging Face Transformers provides cutting-edge pretrained models for text, computer vision, audio, and multimodal tasks, streamlining both inference and training.**  Access the original repo [here](https://github.com/huggingface/transformers).

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_as_a_model_definition.png" alt="Transformers as a Model Definition" style="display: block; margin: 0 auto;"/>

## Key Features:

*   **Extensive Model Support:** Leverage over 1 million+ pre-trained models and checkpoints for a variety of tasks.
*   **Unified API:** A consistent, easy-to-use API for all models, simplifying your workflow.
*   **Cross-Framework Compatibility:** Seamless integration with PyTorch, TensorFlow, and Flax.
*   **Easy Customization:** Customize models to fit your specific needs and easily adapt examples.
*   **Comprehensive Task Coverage:** Support for text generation, classification, translation, image processing, audio analysis, video processing, and multimodal applications.
*   **Cost-Effective:** Utilize pre-trained models to reduce compute costs and environmental impact.

## Installation

Transformers requires Python 3.9+ and supports PyTorch 2.1+, TensorFlow 2.6+, and Flax 0.4.1+.

```bash
# venv
python -m venv .my-env
source .my-env/bin/activate

# uv
uv venv .my-env
source .my-env/bin/activate
```

Install Transformers:

```bash
pip install "transformers[torch]"
```

Install from source for the latest updates:

```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install .[torch]
```

## Quickstart

Get started quickly with the [Pipeline](https://huggingface.co/docs/transformers/pipeline_tutorial) API:

```python
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B")
pipeline("the secret to baking a really good cake is ")
```

To chat with a model, the usage pattern is the same. The only difference is you need to construct a chat history (the input to `Pipeline`) between you and the system.

> [!TIP]
> You can also chat with a model directly from the command line.
> ```shell
> transformers chat Qwen/Qwen2.5-0.5B-Instruct
> ```

```python
import torch
from transformers import pipeline

chat = [
    {"role": "system", "content": "You are a sassy, wise-cracking robot as imagined by Hollywood circa 1986."},
    {"role": "user", "content": "Hey, can you tell me any fun things to do in New York?"}
]

pipeline = pipeline(task="text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
response = pipeline(chat, max_new_tokens=512)
print(response[0]["generated_text"][-1]["content"])
```

## Why Use Transformers?

*   **Ease of Use:** Simple API and pre-trained models make it easy to get started.
*   **Performance:** State-of-the-art results across various modalities.
*   **Efficiency:** Reduce compute costs and carbon footprint by leveraging pre-trained models.
*   **Flexibility:** Choose the right framework for training, evaluation, and deployment.
*   **Customization:** Adapt models and examples to your specific needs.

## Why Not Use Transformers?

*   Not a modular toolbox for building neural networks.
*   Training API optimized for Transformers' PyTorch models. Use [Accelerate](https://huggingface.co/docs/accelerate) for general machine learning loops.
*   Example scripts may require adaptation for specific use cases.

## Projects Built with Transformers

Explore the [awesome-transformers](./awesome-transformers.md) page for a list of amazing projects using Transformers.

## Example Models

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
```