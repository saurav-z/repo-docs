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

<h1 align="center">Hugging Face Transformers: State-of-the-Art NLP & Beyond</h1>

<p align="center">
    <a href="https://github.com/huggingface/transformers">
        <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_as_a_model_definition.png" alt="Transformers Logo" width="600">
    </a>
</p>

Hugging Face Transformers provides cutting-edge, pre-trained models for NLP, computer vision, audio, and more, making advanced AI accessible to everyone.

## Key Features

*   **Wide Variety of Models:** Access over 1 million pre-trained models for various tasks, including text generation, image classification, speech recognition, and multimodal applications.
*   **Easy-to-Use API:** Simplify your workflow with the high-level `Pipeline` API for quick and easy inference across various modalities.
*   **Framework Agnostic:** Seamlessly train, evaluate, and deploy models across PyTorch, TensorFlow, and Flax, leveraging the best tools for each stage.
*   **Cost-Effective:** Save on compute costs by using pre-trained models and fine-tuning them for specific tasks.
*   **Community Driven:** Benefit from a thriving community with an extensive [Hugging Face Hub](https://huggingface.co/models) of models and examples.

## Installation

Get started by installing the Transformers library using pip or uv:

```bash
# venv
python -m venv .my-env
source .my-env/bin/activate

# pip
pip install "transformers[torch]"

# uv
uv venv .my-env
source .my-env/bin/activate
uv pip install "transformers[torch]"
```

For the latest features, install from source:

```bash
git clone https://github.com/huggingface/transformers.git
cd transformers

# pip
pip install .[torch]

# uv
uv pip install .[torch]
```

## Quickstart with the Pipeline API

The `Pipeline` API provides a simple way to get started with Transformers. Here's a basic example for text generation:

```python
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B")
pipeline("the secret to baking a really good cake is ")
```

Explore further examples for:

*   [Automatic speech recognition](#automatic-speech-recognition)
*   [Image classification](#image-classification)
*   [Visual question answering](#visual-question-answering)

## Why Use Transformers?

*   **Simplified AI:** Easily utilize state-of-the-art models for complex tasks.
*   **Reduced Costs:** Leverage pre-trained models to save on computational resources.
*   **Framework Flexibility:** Choose the best framework for your training, evaluation, and production needs.
*   **Customization:** Adapt models to your specific needs with extensive examples and model internals.

## Explore the Community

Discover over 100 innovative projects built with Transformers on the [awesome-transformers page](./awesome-transformers.md).

## Example Models

Explore a diverse range of models across different modalities, including:

*   [Audio](#audio)
*   [Computer Vision](#computer-vision)
*   [Multimodal](#multimodal)
*   [NLP](#nlp)

## Citation

If you use the Transformers library in your research, please cite our paper:

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