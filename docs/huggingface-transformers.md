# Hugging Face Transformers: State-of-the-Art Models for AI Tasks

**Unlock the power of cutting-edge AI with Hugging Face Transformers, a versatile library providing pre-trained models for text, computer vision, audio, video, and multimodal tasks.**  Explore the [original repo](https://github.com/huggingface/transformers) for more information.

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

## Key Features

*   **Wide Range of Models:** Access to over 1 million pretrained model checkpoints on the [Hugging Face Hub](https://huggingface.co/models?library=transformers&sort=trending) for various tasks across text, vision, audio, video, and multimodal applications.
*   **Simplified Usage:** Utilize the high-level [Pipeline](https://huggingface.co/docs/transformers/pipeline_tutorial) API for straightforward inference and training.
*   **Framework Flexibility:**  Seamlessly move models between PyTorch, TensorFlow, and Flax frameworks.
*   **Customization:** Easily adapt models and examples to fit your specific project needs.
*   **Cost-Effective:** Reduce compute costs and carbon footprint by leveraging shared, pre-trained models.

## Installation

Transformers requires Python 3.9+ and supports PyTorch 2.1+, TensorFlow 2.6+, and Flax 0.4.1+.

**Install using `pip`:**

```bash
pip install "transformers[torch]"
```

**Install from source:**

```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install .[torch]
```

## Quickstart

Get started with Transformers using the `Pipeline` API:

```python
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B")
pipeline("the secret to baking a really good cake is ")
```

You can also chat with a model using a chat history:

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

*   **State-of-the-Art Models:** Achieve high performance on various tasks.
*   **Easy to Use:** Simplified API with a unified interface.
*   **Cost-Efficient:** Leverage pre-trained models and reduce compute needs.
*   **Customizable:** Adapt models and examples to fit your specific project needs.

## Why Might You Not Use Transformers?

*   **Not a Modular Toolbox:**  Focused on enabling quick iteration on specific models.
*   **Training API Focus:**  Optimized for PyTorch models and may not be ideal for all generic machine learning loops.
*   **Example Scripts:**  Requires adaptation for specific use cases.

## Example Models (Quick Links)

*   [Audio Examples](#example-models)
*   [Computer Vision Examples](#example-models)
*   [Multimodal Examples](#example-models)
*   [NLP Examples](#example-models)

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