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

**Harness the power of cutting-edge AI with Hugging Face Transformers, the go-to library for easily using and adapting pre-trained models.**  Find out more on the [original repo](https://github.com/huggingface/transformers).

## Key Features

*   **Wide Variety of Models:** Access a vast collection of pre-trained models for tasks like text generation, image classification, audio processing, and more.
*   **Easy-to-Use API:**  Quickly get started with simple, high-level APIs like `pipeline` for common tasks.
*   **Framework Flexibility:**  Seamlessly integrate with PyTorch, TensorFlow, and Flax, and easily move models between them.
*   **Customization & Fine-tuning:** Adapt models to your specific needs with extensive examples and model customization options.
*   **Cost-Effective:** Leverage pre-trained models to reduce compute costs and accelerate your projects.
*   **Community-Driven:** Join a thriving community with over 1 million model checkpoints available on the Hugging Face Hub.

### **Installation**

*   **Requirements:**  Python 3.9+, PyTorch 2.1+, TensorFlow 2.6+, and Flax 0.4.1+.
*   **Virtual Environment:** Create and activate a virtual environment (using `venv` or `uv`) to manage dependencies.
*   **Installation Options:**
    *   **pip:** `pip install "transformers[torch]"`
    *   **uv:** `uv pip install "transformers[torch]"`
    *   **From Source:**  For the latest features (potentially unstable), clone the repository and install with `pip install .[torch]` or `uv pip install .[torch]`.

### **Quickstart with the `pipeline` API**

The `pipeline` API simplifies using transformers for various tasks. Here's a quick example for text generation:

```python
from transformers import pipeline

generator = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B")
result = generator("the secret to baking a really good cake is ")
print(result[0]["generated_text"])
```

To chat with a model, use the same pattern, but construct a chat history.

### **Why Choose Transformers?**

*   **Simplicity and Ease of Use:** Designed for researchers, engineers, and developers.
*   **Efficiency:**  Reduce training time and costs by leveraging pre-trained models.
*   **Framework Agnostic:** Train, evaluate, and deploy models with your preferred framework.
*   **Customization:** Tailor models to your specific use cases with ease.

### **Why Not Use Transformers?**

*   **Not a Modular Toolbox:**  The library prioritizes model-specific code for easy iteration.
*   **Training API:** The training API is optimized for PyTorch models provided by Transformers. For generic machine learning loops, use another library.
*   **Example Scripts:** Adapt the example scripts to your specific use case.

### **Explore Further**

*   **Model Checkpoints:** Explore over 1 million models on the [Hugging Face Hub](https://huggingface.co/models).
*   **[100+ Projects Built with Transformers](awesome-transformers.md)**
*   **[Documentation](https://huggingface.co/docs/transformers/index)**
*   **[Hugging Face Enterprise Hub](https://huggingface.co/enterprise)**

### **Citation**

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