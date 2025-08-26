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

## Transformers: Your Gateway to State-of-the-Art AI Models

**Harness the power of cutting-edge AI with the Hugging Face Transformers library, providing easy access to pre-trained models for a wide range of tasks.**  

  [View the original repo](https://github.com/huggingface/transformers)

<h3 align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_as_a_model_definition.png"/>
</h3>

Hugging Face Transformers is the go-to library for utilizing state-of-the-art machine learning models in text, computer vision, audio, video, and multimodal applications, simplifying both inference and training.

**Key Features:**

*   **Extensive Model Support:** Access over 1 million+ pre-trained [model checkpoints](https://huggingface.co/models?library=transformers&sort=trending) on the [Hugging Face Hub](https://huggingface.com/models), covering diverse modalities and tasks.
*   **Unified API:** Enjoy a consistent and easy-to-use API for all supported models, simplifying your workflow.
*   **Cross-Framework Compatibility:** Seamlessly integrate with major training frameworks (Axolotl, Unsloth, DeepSpeed, etc.) and inference engines (vLLM, SGLang, TGI, etc.) for maximum flexibility.
*   **Simplified Customization:** Easily adapt models to your specific needs with well-documented examples and customizable model internals.
*   **Reduced Costs and Carbon Footprint:** Leverage pre-trained models to significantly lower compute time and costs, promoting sustainable AI practices.

## Installation

Transformers supports Python 3.9+ with PyTorch 2.1+, TensorFlow 2.6+, and Flax 0.4.1+.

**Setup a Virtual Environment (Recommended)**

Create and activate a virtual environment using either [venv](https://docs.python.org/3/library/venv.html) or [uv](https://docs.astral.sh/uv):

```bash
# venv
python -m venv .my-env
source .my-env/bin/activate

# uv
uv venv .my-env
source .my-env/bin/activate
```

**Install Transformers:**

```bash
# pip
pip install "transformers[torch]"

# uv
uv pip install "transformers[torch]"
```

**Install from Source (For Latest Changes - may be unstable):**

```bash
git clone https://github.com/huggingface/transformers.git
cd transformers

# pip
pip install .[torch]

# uv
uv pip install .[torch]
```

## Quickstart

Get started quickly using the [Pipeline API](https://huggingface.co/docs/transformers/pipeline_tutorial), a high-level tool for various tasks.

```python
from transformers import pipeline

# Text Generation Example
pipeline = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B")
print(pipeline("the secret to baking a really good cake is "))

# Chat Example
import torch
chat = [
    {"role": "system", "content": "You are a sassy, wise-cracking robot as imagined by Hollywood circa 1986."},
    {"role": "user", "content": "Hey, can you tell me any fun things to do in New York?"}
]
pipeline = pipeline(task="text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", dtype=torch.bfloat16, device_map="auto")
response = pipeline(chat, max_new_tokens=512)
print(response[0]["generated_text"][-1]["content"])
```

Explore the examples below to see how the `Pipeline` works for different modalities and tasks.

<details>
<summary>Automatic Speech Recognition</summary>

```python
from transformers import pipeline
pipeline = pipeline(task="automatic-speech-recognition", model="openai/whisper-large-v3")
print(pipeline("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")['text'])
```

</details>

<details>
<summary>Image Classification</summary>

<h3 align="center">
    <a><img src="https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"></a>
</h3>

```python
from transformers import pipeline
pipeline = pipeline(task="image-classification", model="facebook/dinov2-small-imagenet1k-1-layer")
print(pipeline("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"))
```

</details>

<details>
<summary>Visual Question Answering</summary>

<h3 align="center">
    <a><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-few-shot.jpg"></a>
</h3>

```python
from transformers import pipeline
pipeline = pipeline(task="visual-question-answering", model="Salesforce/blip-vqa-base")
print(pipeline(
    image="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-few-shot.jpg",
    question="What is in the image?",
))
```

</details>

## Why Use Transformers?

*   **Ease of Use:** Leverage state-of-the-art models with a simple, unified API.
*   **Efficiency:** Reduce compute costs by using pre-trained models and fine-tuning them for your needs.
*   **Flexibility:** Train, evaluate, and deploy models across various frameworks (PyTorch, JAX, TensorFlow).
*   **Customization:** Adapt models easily with provided examples and access to model internals.

## Limitations of Transformers:

*   Not a modular building block library, the focus is on supporting research and quick iteration.
*   The training API is optimized for the Transformers library, consider [Accelerate](https://huggingface.co/docs/accelerate) for general machine learning loops.
*   Example scripts are meant to be adapted for specific use cases.

## Explore the Community

Discover over 100 incredible projects built with Transformers in the [awesome-transformers](./awesome-transformers.md) page.

## Example Models

Browse popular models across various modalities on their respective [Hub model pages](https://huggingface.co/models).

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

If you use the 🤗 Transformers library, please cite our paper:

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