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

## Hugging Face Transformers: State-of-the-Art Models for NLP, Computer Vision, and More

**Hugging Face Transformers is a powerful library that provides easy-to-use tools and pre-trained models for a wide range of AI tasks, from text generation to image classification.**  Explore the [Hugging Face Transformers repository](https://github.com/huggingface/transformers) to get started with cutting-edge machine learning models.

### Key Features

*   **Extensive Model Support:** Access a vast library of pre-trained models for text, computer vision, audio, video, and multimodal tasks.
*   **Simplified Usage:**  Utilize a unified API for easy inference and training, with high-level abstractions like the `Pipeline` API for quick experimentation.
*   **Framework Flexibility:**  Seamlessly integrate with PyTorch, TensorFlow, and Flax, and easily move models between frameworks.
*   **Pre-trained Models:** Access over 1 million pre-trained model checkpoints on the [Hugging Face Hub](https://huggingface.co/models).
*   **Customization:** Easily adapt models to your specific needs, with clear examples and exposed internals for fine-tuning.
*   **Community-Driven:** Benefit from a vibrant community with numerous projects and resources built around Transformers.

### Installation

Transformers requires Python 3.9+ and supports PyTorch, TensorFlow, and Flax.  Follow these steps to install:

1.  **Create a virtual environment:** Use `venv` or `uv`.
2.  **Activate the environment.**
3.  **Install Transformers:**

    ```bash
    # pip
    pip install "transformers[torch]"
    # uv
    uv pip install "transformers[torch]"
    ```
    Or, install from source for the latest changes:

    ```bash
    git clone https://github.com/huggingface/transformers.git
    cd transformers
    pip install .[torch]  # Or use uv
    ```

### Quickstart with the Pipeline API

Get started quickly with the `Pipeline` API, a high-level tool for inference across different modalities:

```python
from transformers import pipeline

# Text Generation
pipeline = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B")
print(pipeline("the secret to baking a really good cake is ")[0]['generated_text'])
```

*   **Chatbot Example:**

    ```python
    import torch
    from transformers import pipeline

    chat = [
        {"role": "system", "content": "You are a sassy, wise-cracking robot."},
        {"role": "user", "content": "Hey, can you tell me any fun things to do in New York?"}
    ]

    pipeline = pipeline(task="text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
    response = pipeline(chat, max_new_tokens=512)
    print(response[0]["generated_text"][-1]["content"])
    ```

*   **Other tasks include:**
    *   Automatic Speech Recognition
    *   Image Classification
    *   Visual Question Answering

### Why Choose Transformers?

*   **Ease of Use:** Quickly deploy state-of-the-art models.
*   **Cost-Effectiveness:** Reduce compute costs by leveraging pre-trained models.
*   **Flexibility:**  Easily switch between training and inference frameworks.
*   **Customization:** Adapt models to fit your project.

### 100+ Community Projects

Transformers is more than just a library, it's a community.  Explore the [awesome-transformers](./awesome-transformers.md) page to discover amazing projects built using the library.

### Example Models

[Explore example models](https://huggingface.co/models?library=transformers&sort=trending) for various tasks and modalities, including:

*   Audio (e.g., Speech Recognition, Text to Speech)
*   Computer Vision (e.g., Image Classification, Object Detection)
*   Multimodal (e.g., Image Captioning, Visual Question Answering)
*   NLP (e.g., Text Generation, Summarization)

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