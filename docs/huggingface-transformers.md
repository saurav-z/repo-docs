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

**Harness the power of cutting-edge AI with Hugging Face Transformers, a library providing pre-trained models for a wide range of tasks and modalities, from text to video!** Access the [original repo here](https://github.com/huggingface/transformers).

<h3 align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_as_a_model_definition.png"/>
</h3>

## Key Features

*   **Extensive Model Library:** Access over 1 million pre-trained [model checkpoints](https://huggingface.co/models?library=transformers&sort=trending) on the Hugging Face Hub, covering text, images, audio, video, and multimodal tasks.
*   **Unified API:** Simplify your workflow with a consistent API for using all pre-trained models.
*   **Ease of Use:** Get started quickly with the `Pipeline` API, designed for text generation, audio processing, image classification, and more.
*   **Framework Flexibility:** Train and deploy models with PyTorch, TensorFlow, and Flax.
*   **Customization:** Easily adapt models and examples to fit your specific needs.
*   **Cost-Effective:** Leverage pre-trained models to reduce compute costs and accelerate development.
*   **Community-Driven:** Benefit from a vibrant community and numerous projects built on Transformers.

## Installation

Transformers requires Python 3.9+ and supports PyTorch 2.1+, TensorFlow 2.6+, and Flax 0.4.1+.

Choose your preferred installation method:

*   **Using `venv`:**
    ```bash
    python -m venv .my-env
    source .my-env/bin/activate
    pip install "transformers[torch]"
    ```

*   **Using `uv`:**
    ```bash
    uv venv .my-env
    source .my-env/bin/activate
    uv pip install "transformers[torch]"
    ```
*   **From Source (for latest changes):**
    ```bash
    git clone https://github.com/huggingface/transformers.git
    cd transformers
    pip install .[torch]
    ```

## Quickstart with the Pipeline API

Get started with Transformers effortlessly using the [Pipeline](https://huggingface.co/docs/transformers/pipeline_tutorial) API:

```python
from transformers import pipeline

# Example: Text generation
pipeline = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B")
pipeline("the secret to baking a really good cake is ")
```

```python
import torch
from transformers import pipeline

# Example: Chat with a model
chat = [
    {"role": "system", "content": "You are a sassy, wise-cracking robot as imagined by Hollywood circa 1986."},
    {"role": "user", "content": "Hey, can you tell me any fun things to do in New York?"}
]

pipeline = pipeline(task="text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
response = pipeline(chat, max_new_tokens=512)
print(response[0]["generated_text"][-1]["content"])
```

## Why Use Transformers?

*   **State-of-the-Art Models:** Achieve high performance across various AI tasks.
*   **Democratization:** Make advanced models accessible to everyone.
*   **Cost Savings:** Reduce training time and infrastructure costs.
*   **Flexibility:** Train, evaluate, and deploy models with your preferred framework.
*   **Customization:** Easily tailor models to your specific needs.

## Limitations

*   This library is not a modular toolbox for building neural nets.
*   The training API is optimized for Transformers models.
*   Example scripts are templates; adapt them for your use case.

## Explore and Contribute

Explore the [Hugging Face Hub](https://huggingface.co/models) to find models. Contribute to the community by [opening an issue](https://github.com/huggingface/transformers/issues) or adding to the [awesome-transformers](./awesome-transformers.md) list of projects!

## Example Models (by Modality)

*(See original README for details)*

## Citation

*(See original README for citation information)*