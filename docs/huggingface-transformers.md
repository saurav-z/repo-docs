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

<h1 align="center">Hugging Face Transformers: State-of-the-Art Models for NLP, Computer Vision, and More!</h1>

<p align="center">
  Unlock the power of cutting-edge AI with Hugging Face Transformers, the leading library for pre-trained models.
</p>

<div align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_as_a_model_definition.png"/>
</div>

[Explore the Hugging Face Transformers Library on GitHub](https://github.com/huggingface/transformers)

## Key Features:

*   **Wide Variety of Models:** Access a vast library of pre-trained models for various tasks, including natural language processing (NLP), computer vision, audio, video, and multimodal applications.
*   **Ease of Use:** Simplify your workflow with a user-friendly API and high-level abstractions, making it easy for researchers, engineers, and developers to get started.
*   **Flexibility:** Train state-of-the-art models with just a few lines of code, and seamlessly switch between PyTorch, TensorFlow, and Flax.
*   **Cost-Effective:** Leverage pre-trained models to reduce compute costs and accelerate development cycles.
*   **Customization:** Tailor models to your specific needs with comprehensive examples and exposed model internals.
*   **Unified API:** Leverage a unified API for all of our pretrained models to provide consistency and efficiency.

## What is Hugging Face Transformers?

The Hugging Face Transformers library provides state-of-the-art, pre-trained models for a wide range of AI tasks. It serves as a central framework for model definitions, ensuring compatibility across various training frameworks (Axolotl, Unsloth, DeepSpeed, etc.) and inference engines (vLLM, SGLang, TGI, etc.) and modeling libraries.  With millions of model checkpoints available on the [Hugging Face Hub](https://huggingface.com/models), you can quickly integrate powerful AI capabilities into your projects.

## Installation

Get started quickly by installing Transformers:

```bash
pip install "transformers[torch]"
```

For the latest updates or to contribute:

```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install .[torch]
```

## Quickstart

Use the `Pipeline` API for high-level inference across various tasks:

```python
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B")
pipeline("the secret to baking a really good cake is ")
```

```python
# Chat Example
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

## Why Choose Hugging Face Transformers?

*   **Simplified AI:** Easy-to-use, state-of-the-art models for various AI tasks.
*   **Efficiency:** Reduce compute costs by using pre-trained models.
*   **Framework Agnostic:** Choose the best framework for training and production.
*   **Customization:** Easily adapt models to your specific needs.

## Example Models

Explore these models across different modalities:

*   **Audio:** [Whisper](https://huggingface.co/openai/whisper-large-v3-turbo), [Moonshine](https://huggingface.co/UsefulSensors/moonshine)
*   **Computer Vision:** [SAM](https://huggingface.co/facebook/sam-vit-base), [DINO v2](https://huggingface.co/facebook/dinov2-base)
*   **Multimodal:** [Qwen2-Audio](https://huggingface.co/Qwen/Qwen2-Audio-7B), [BLIP-2](https://huggingface.co/Salesforce/blip2-opt-2.7b)
*   **NLP:** [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base), [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)

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