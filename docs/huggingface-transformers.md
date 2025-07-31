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

<h1 align="center">Hugging Face Transformers: State-of-the-Art Models for NLP, Computer Vision, and More</h1>

<p align="center"><b>Power your AI projects with cutting-edge pretrained models for various modalities, including text, images, audio, and video, all in one versatile library.</b></p>

<h3 align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_as_a_model_definition.png"/>
</h3>

<p>The Hugging Face <a href="https://github.com/huggingface/transformers">Transformers</a> library provides access to a vast collection of pre-trained models for a wide range of tasks across different modalities.  It simplifies the process of using state-of-the-art machine learning models for both inference and training, allowing you to build powerful AI applications quickly and efficiently.</p>

## Key Features

*   **Comprehensive Model Support:** Access to over 1 million [pre-trained model checkpoints](https://huggingface.co/models?library=transformers&sort=trending) on the [Hugging Face Hub](https://huggingface.com/models), covering NLP, computer vision, audio, video, and multimodal tasks.
*   **Unified API:** Consistent and easy-to-use API for all pre-trained models, simplifying model selection and usage.
*   **Framework Compatibility:** Works seamlessly with popular frameworks such as PyTorch, TensorFlow, and Flax, providing flexibility in your development workflow.
*   **Ease of Use:**  Utilize high-level APIs like the `Pipeline` for quick experimentation and deployment, streamlining your development process.
*   **Customization and Extensibility:**  Fine-tune and adapt models to your specific needs with examples for each architecture.
*   **Community-Driven:** Benefit from a vibrant community of developers and researchers who contribute to the library's growth and provide support.
*   **Efficiency:** Leverages shared models, reducing compute costs, and promotes a smaller carbon footprint.

## Installation

Transformers requires Python 3.9+ and supports PyTorch 2.1+, TensorFlow 2.6+, and Flax 0.4.1+.

1.  **Create a Virtual Environment:**  Use `venv` or `uv`.

```bash
# venv
python -m venv .my-env
source .my-env/bin/activate
# uv
uv venv .my-env
source .my-env/bin/activate
```

2.  **Install Transformers:**

```bash
# pip
pip install "transformers[torch]"

# uv
uv pip install "transformers[torch]"
```

3.  **(Optional) Install from Source:** For the latest changes or to contribute, clone the repository and install.

```bash
git clone https://github.com/huggingface/transformers.git
cd transformers

# pip
pip install .[torch]

# uv
uv pip install .[torch]
```

## Quickstart

The `Pipeline` API provides a high-level interface for inference across various tasks.

```python
from transformers import pipeline

# Text Generation
pipeline = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B")
print(pipeline("the secret to baking a really good cake is "))
```

```python
# Chat with a model
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

*(See the original README for example details on other tasks.)*

## Example Models

*(See the original README for example models.)*

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