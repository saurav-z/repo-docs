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

<h2 align="center">
  Revolutionize Your AI Projects with Hugging Face Transformers
</h2>

<p align="center">
    <a href="https://github.com/huggingface/transformers">
        <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_as_a_model_definition.png" alt="Transformers Logo" width="600"/>
    </a>
</p>

Hugging Face Transformers provides state-of-the-art pre-trained models and tools for developers and researchers to easily build and deploy cutting-edge AI applications.

**Key Features:**

*   **Wide Variety of Models:** Access a comprehensive collection of pre-trained models for text, computer vision, audio, video, and multimodal tasks.
*   **Simplified Usage:** Easily integrate these models into your projects with a user-friendly API and high-level abstractions like the Pipeline.
*   **Unified Ecosystem:** Benefit from a centralized model definition that ensures compatibility across various training frameworks, inference engines, and related libraries.
*   **Cost-Effective Solutions:** Reduce computational costs and your carbon footprint by leveraging pre-trained models instead of training from scratch.
*   **Customization and Flexibility:** Easily tailor models and examples to meet your specific requirements and explore model internals.
*   **Extensive Community Support:** Join a vibrant community of developers, researchers, and enthusiasts, with access to a vast library of community-contributed projects.

**Get Started:**

### Installation

Install Transformers using `pip` or `uv`:

```bash
# pip
pip install "transformers[torch]"

# uv
uv pip install "transformers[torch]"
```

Or from source:

```bash
git clone https://github.com/huggingface/transformers.git
cd transformers

# pip
pip install .[torch]

# uv
uv pip install .[torch]
```

### Quickstart

Use the `Pipeline` API for easy inference:

```python
from transformers import pipeline

# Example: Text Generation
pipeline = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B")
print(pipeline("the secret to baking a really good cake is "))
```

### Explore further

*   [Hugging Face Hub](https://huggingface.co/models): Find over 1 million pre-trained Transformers models.
*   [Documentation](https://huggingface.co/docs/transformers/index):  Dive deeper into the library's capabilities.

**Why Use Transformers?**

*   **Ease of Use:** State-of-the-art models are simple to integrate.
*   **Efficiency:** Reduce costs and environmental impact.
*   **Framework Agnostic:** Train, evaluate, and deploy on the framework of your choice.
*   **Customization:** Adapt models and examples to your specific needs.

**Why Not Use Transformers?**

*   **Not a Modular Toolbox:**  Focus is on easy iteration, not complex abstractions.
*   **Training API:** Best suited for PyTorch models.
*   **Example Scripts:**  Adapt code for your specific use case.

**[See 100 Projects Using Transformers](awesome-transformers.md)**

### Example Models

*   **Audio:** [Whisper](https://huggingface.co/openai/whisper-large-v3-turbo), [Moonshine](https://huggingface.co/UsefulSensors/moonshine), [MusicGen](https://huggingface.co/facebook/musicgen-large)
*   **Computer Vision:** [SAM](https://huggingface.co/facebook/sam-vit-base), [DINO v2](https://huggingface.co/facebook/dinov2-base), [VideoMAE](https://huggingface.co/MCG-NJU/videomae-large)
*   **Multimodal:** [Qwen-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct), [BLIP-2](https://huggingface.co/Salesforce/blip2-opt-2.7b), [LLaVA](https://huggingface.co/llava-hf/llava-1.5-7b-hf)
*   **NLP:** [Gemma](https://huggingface.co/google/gemma-2-2b), [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1), [BART](https://huggingface.co/facebook/bart-large-cnn)

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

**Explore the power of Transformers: [Hugging Face Transformers GitHub](https://github.com/huggingface/transformers)**