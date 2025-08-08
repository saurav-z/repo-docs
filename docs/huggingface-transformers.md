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

## Hugging Face Transformers: State-of-the-Art NLP Made Easy

**Harness the power of cutting-edge AI models with Hugging Face Transformers, a Python library providing pre-trained models for text, vision, audio, and multimodal tasks, built for both inference and training.**  Find the source code [here](https://github.com/huggingface/transformers).

**Key Features:**

*   **Wide Variety of Models:** Access a vast library of over 1 million pre-trained models for diverse tasks like text generation, translation, image classification, and more.
*   **Ease of Use:** Simplified API with the `Pipeline` class for quick and easy implementation, handling preprocessing and output.
*   **Cross-Framework Compatibility:** Seamlessly integrates with popular frameworks like PyTorch, TensorFlow, and Flax.
*   **Customization:** Easily fine-tune and adapt models to your specific needs with comprehensive examples and exposed model internals.
*   **Cost-Effective:** Leverage shared, pre-trained models to reduce compute costs and carbon footprint.
*   **Multimodal Support:** Explore models that handle text, vision, audio, and their combinations.
*   **Community Driven:** Benefit from a vibrant community with many projects built on Transformers.

### Installation

*   **Prerequisites:** Python 3.9+ and PyTorch 2.1+, TensorFlow 2.6+, or Flax 0.4.1+.
*   **Virtual Environment (recommended):** Use `venv` or `uv` to create and activate a virtual environment.
*   **Install:**
    ```bash
    # pip
    pip install "transformers[torch]"

    # uv
    uv pip install "transformers[torch]"
    ```
*   **Install from Source:** For the latest updates, install from source:

    ```bash
    git clone https://github.com/huggingface/transformers.git
    cd transformers

    # pip
    pip install .[torch]

    # uv
    uv pip install .[torch]
    ```

### Quickstart

Get started with Transformers using the `Pipeline` API.

```python
from transformers import pipeline

# Example: Text generation using Qwen
pipeline = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B")
print(pipeline("the secret to baking a really good cake is ")[0]['generated_text'])
```

```python
import torch
from transformers import pipeline

# Example: Chat with a model using Llama 3
chat = [
    {"role": "system", "content": "You are a sassy, wise-cracking robot as imagined by Hollywood circa 1986."},
    {"role": "user", "content": "Hey, can you tell me any fun things to do in New York?"}
]

pipeline = pipeline(task="text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
response = pipeline(chat, max_new_tokens=512)
print(response[0]["generated_text"][-1]["content"])
```

**(Expand the below sections to see more pipeline examples for diverse tasks.)**

<details>
<summary>Automatic speech recognition</summary>

```py
from transformers import pipeline

pipeline = pipeline(task="automatic-speech-recognition", model="openai/whisper-large-v3")
print(pipeline("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")['text'])
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
print(pipeline("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"))
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
print(pipeline(
    image="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-few-shot.jpg",
    question="What is in the image?",
)[0]['answer'])
```

</details>

### Why Use Transformers?

*   **State-of-the-Art Models:** Access high-performing models across various modalities.
*   **Simplified Development:** Easy-to-use API and unified interface.
*   **Cost & Resource Efficiency:** Share pretrained models, saving on compute.
*   **Flexibility:** Easily customize models for specific needs.

### When Not to Use Transformers

*   Not ideal for general-purpose modular neural network building blocks.
*   The training API is optimized for PyTorch models.
*   Example scripts may require adaptation to your specific use case.

### Projects Using Transformers

Explore the [awesome-transformers](./awesome-transformers.md) page for 100+ incredible projects built with Transformers.

### Example Models

**(Browse the examples below to see a few of the models offered by this library.)**

<details>
<summary>Audio</summary>

-   Audio classification with [Whisper](https://huggingface.co/openai/whisper-large-v3-turbo)
-   Automatic speech recognition with [Moonshine](https://huggingface.co/UsefulSensors/moonshine)
-   Keyword spotting with [Wav2Vec2](https://huggingface.co/superb/wav2vec2-base-superb-ks)
-   Speech to speech generation with [Moshi](https://huggingface.co/kyutai/moshiko-pytorch-bf16)
-   Text to audio with [MusicGen](https://huggingface.co/facebook/musicgen-large)
-   Text to speech with [Bark](https://huggingface.co/suno/bark)

</details>

<details>
<summary>Computer vision</summary>

-   Automatic mask generation with [SAM](https://huggingface.co/facebook/sam-vit-base)
-   Depth estimation with [DepthPro](https://huggingface.co/apple/DepthPro-hf)
-   Image classification with [DINO v2](https://huggingface.co/facebook/dinov2-base)
-   Keypoint detection with [SuperPoint](https://huggingface.co/magic-leap-community/superpoint)
-   Keypoint matching with [SuperGlue](https://huggingface.co/magic-leap-community/superglue_outdoor)
-   Object detection with [RT-DETRv2](https://huggingface.co/PekingU/rtdetr_v2_r50vd)
-   Pose Estimation with [VitPose](https://huggingface.co/usyd-community/vitpose-base-simple)
-   Universal segmentation with [OneFormer](https://huggingface.co/shi-labs/oneformer_ade20k_swin_large)
-   Video classification with [VideoMAE](https://huggingface.co/MCG-NJU/videomae-large)

</details>

<details>
<summary>Multimodal</summary>

-   Audio or text to text with [Qwen2-Audio](https://huggingface.co/Qwen/Qwen2-Audio-7B)
-   Document question answering with [LayoutLMv3](https://huggingface.co/microsoft/layoutlmv3-base)
-   Image or text to text with [Qwen-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
-   Image captioning [BLIP-2](https://huggingface.co/Salesforce/blip2-opt-2.7b)
-   OCR-based document understanding with [GOT-OCR2](https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf)
-   Table question answering with [TAPAS](https://huggingface.co/google/tapas-base)
-   Unified multimodal understanding and generation with [Emu3](https://huggingface.co/BAAI/Emu3-Gen)
-   Vision to text with [Llava-OneVision](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf)
-   Visual question answering with [Llava](https://huggingface.co/llava-hf/llava-1.5-7b-hf)
-   Visual referring expression segmentation with [Kosmos-2](https://huggingface.co/microsoft/kosmos-2-patch14-224)

</details>

<details>
<summary>NLP</summary>

-   Masked word completion with [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base)
-   Named entity recognition with [Gemma](https://huggingface.co/google/gemma-2-2b)
-   Question answering with [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)
-   Summarization with [BART](https://huggingface.co/facebook/bart-large-cnn)
-   Translation with [T5](https://huggingface.co/google-t5/t5-base)
-   Text generation with [Llama](https://huggingface.co/meta-llama/Llama-3.2-1B)
-   Text classification with [Qwen](https://huggingface.co/Qwen/Qwen2.5-0.5B)

</details>

### Citation

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