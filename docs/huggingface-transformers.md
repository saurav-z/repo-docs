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


<h1 align="center">Hugging Face Transformers: State-of-the-Art Models for AI</h1>

<p align="center">
  <b>Unlock the power of cutting-edge AI with the Hugging Face Transformers library, providing access to thousands of pre-trained models for various tasks.</b>
</p>

<h3 align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_as_a_model_definition.png"/>
</h3>

**[Hugging Face Transformers](https://github.com/huggingface/transformers) is your gateway to the world of advanced AI models, offering easy-to-use tools for both inference and training.** Centralizing model definitions across the ecosystem, this library ensures compatibility across various frameworks and is essential for any project.  
  
**Key Features:**

*   **Vast Model Library:** Access over 1 million pre-trained transformer models on the [Hugging Face Hub](https://huggingface.co/models).
*   **Multi-Framework Compatibility:** Seamlessly integrate with PyTorch, TensorFlow, Flax, and more.
*   **Unified API:**  Easy-to-use API for a wide range of tasks, including text, vision, audio, video, and multimodal models.
*   **Simplified Training:**  Train state-of-the-art models with just a few lines of code, reducing complexity and time.
*   **Customization:** Easily adapt models or examples to fit your unique needs, supporting both researchers and developers.
*   **Cost-Effective:** Leverage pre-trained models to reduce computational costs, lower your carbon footprint and accelerate your projects.

## Installation

Get started quickly with `transformers` by installing it with your preferred package manager. 

```bash
# pip (using venv)
python -m venv .my-env
source .my-env/bin/activate
pip install "transformers[torch]"

# uv (using uv)
uv venv .my-env
source .my-env/bin/activate
uv pip install "transformers[torch]"
```
***

```bash
# Install from source (for the latest changes)
git clone https://github.com/huggingface/transformers.git
cd transformers

# pip
pip install .[torch]

# uv
uv pip install .[torch]
```

## Quickstart: Get Started with the Pipeline API

The `pipeline` API is a user-friendly, high-level interface for quickly implementing various AI tasks, from text generation to image classification.

```python
from transformers import pipeline

# Text Generation Example
generator = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B")
result = generator("the secret to baking a really good cake is ")
print(result[0]['generated_text'])
```

## Model Examples by Modality

*   **Audio:** Speech recognition, audio classification, and text-to-speech tasks.
*   **Computer Vision:** Image classification, object detection, and image segmentation.
*   **Multimodal:** Document understanding and image captioning.
*   **NLP:** Text classification, question answering, and text summarization.

For detailed examples and specific model usages, expand the sections in the original documentation.

## Why Use Transformers?

*   **Accessibility:** Easy to use for both beginners and experts.
*   **Efficiency:** Saves time and resources by using pre-trained models.
*   **Flexibility:** Train, evaluate, and deploy models using your preferred framework.

## Why Not Use Transformers?

*   This library may not be suitable for modular building blocks, and the training API is optimized for PyTorch models.
*   The example scripts may need to be adapted for your specific use case.

## Join the Community!

Explore the growing ecosystem of projects powered by Transformers: [awesome-transformers](./awesome-transformers.md)

## Citation

If you use the 🤗 Transformers library in your research, please cite our paper:
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