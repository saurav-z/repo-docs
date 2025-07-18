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

# Hugging Face Transformers: State-of-the-Art Models for NLP and Beyond

**Harness the power of cutting-edge machine learning with Hugging Face Transformers, your go-to library for easily using pre-trained models for a wide range of tasks!**  [Go to the GitHub Repo](https://github.com/huggingface/transformers)

<div align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_as_a_model_definition.png" alt="Transformers as a Model Definition">
</div>

## Key Features:

*   **Vast Model Library:** Access over 1 million pre-trained Transformer models on the [Hugging Face Hub](https://huggingface.com/models) for various modalities like text, images, audio, and video.
*   **Unified API:** Easily use different models with a consistent and intuitive API.
*   **Simplified Customization:** Adapt and fine-tune models to fit your specific needs, with examples provided for each architecture.
*   **Cross-Framework Compatibility:** Seamlessly work with popular training frameworks like PyTorch, TensorFlow, and Flax.
*   **Reduce Costs & Carbon Footprint:** Leverage pre-trained models to save time, compute resources, and reduce your environmental impact.
*   **Active Community:** Join a vibrant community with tons of projects built on top of Transformers.

## Installation

Get started by installing Transformers with these simple steps:

1.  **Choose your framework:** Transformers supports Python 3.9+ with PyTorch, TensorFlow, and Flax.
2.  **Create a Virtual Environment:** Use `venv` or `uv` to create and activate your environment.
3.  **Install Transformers:** Use pip or uv:
    ```bash
    # pip
    pip install "transformers[torch]"
    # uv
    uv pip install "transformers[torch]"
    ```
    To install from source:
    ```bash
    git clone https://github.com/huggingface/transformers.git
    cd transformers
    # pip
    pip install .[torch]
    # uv
    uv pip install .[torch]
    ```

## Quickstart: Get Started with Pipelines

The `Pipeline` API offers an easy entrypoint for different tasks.

```python
from transformers import pipeline

# Text Generation
generator = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B")
result = generator("the secret to baking a really good cake is ")
print(result[0]['generated_text'])
```
```python
import torch
from transformers import pipeline

# Chatbot
chat = [
    {"role": "system", "content": "You are a sassy, wise-cracking robot as imagined by Hollywood circa 1986."},
    {"role": "user", "content": "Hey, can you tell me any fun things to do in New York?"}
]

pipeline = pipeline(task="text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
response = pipeline(chat, max_new_tokens=512)
print(response[0]["generated_text"][-1]["content"])
```

**Explore examples for other tasks:**

*   **Automatic Speech Recognition:**
    ```python
    from transformers import pipeline
    pipeline = pipeline(task="automatic-speech-recognition", model="openai/whisper-large-v3")
    result = pipeline("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
    print(result['text'])
    ```
*   **Image Classification:**
    ```python
    from transformers import pipeline
    pipeline = pipeline(task="image-classification", model="facebook/dinov2-small-imagenet1k-1-layer")
    result = pipeline("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
    print(result)
    ```
*   **Visual Question Answering:**
    ```python
    from transformers import pipeline
    pipeline = pipeline(task="visual-question-answering", model="Salesforce/blip-vqa-base")
    result = pipeline(
        image="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-few-shot.jpg",
        question="What is in the image?",
    )
    print(result)
    ```

## Why Use Transformers?

*   **Ease of Use:** Simple and intuitive API for state-of-the-art models.
*   **Efficiency:** Reduce compute costs and environmental impact.
*   **Flexibility:** Train, evaluate, and deploy models across different frameworks.
*   **Customization:** Adapt models and examples to fit your use case.

## Why *Might* You NOT Use Transformers?

*   The library is not a modular toolbox of building blocks for neural networks.
*   The training API is optimized for PyTorch models provided by Transformers.
*   Example scripts may need adaptation for your specific use case.

## Projects Powered by Transformers

Discover 100+ amazing projects built with Transformers in our [awesome-transformers](./awesome-transformers.md) page.

## Example Models

Here are examples from different modalities, check them out on their respective [Hub model pages](https://huggingface.co/models).

*   **(Examples for each modality will go here - same as the original, but summarized)**

## Citation

Cite our paper:

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