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

# Hugging Face Transformers: State-of-the-Art Models for NLP, Computer Vision, and More

**Unlock the power of cutting-edge AI with Hugging Face Transformers, your go-to library for easily accessing and utilizing pre-trained models for diverse tasks like text generation, image classification, and audio processing.**  [Get Started](https://github.com/huggingface/transformers)

<h3 align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_as_a_model_definition.png"/>
</h3>

Hugging Face Transformers provides a unified framework for utilizing state-of-the-art machine learning models across various modalities, including text, computer vision, audio, video, and multimodal applications. It is the central point of reference for defining models, ensuring compatibility across training frameworks (Axolotl, Unsloth, DeepSpeed, etc.), inference engines (vLLM, SGLang, TGI, etc.), and related libraries (llama.cpp, mlx, etc.).

## Key Features

*   **Extensive Model Support:** Access a vast library of over 1 million pre-trained model checkpoints on the [Hugging Face Hub](https://huggingface.co/models).
*   **Unified API:** Utilize a consistent API for all supported models, simplifying your workflow.
*   **Cross-Framework Compatibility:** Seamlessly integrate with popular training frameworks, inference engines, and related libraries.
*   **Easy Customization:** Adapt models and examples to your specific needs with accessible model internals.
*   **Reduced Compute Costs:** Leverage pre-trained models to significantly reduce training time and associated costs.
*   **Community-Driven:** Join a vibrant community of developers, researchers, and engineers building innovative projects.

## Installation

To get started with Transformers, follow these steps:

1.  **Prerequisites:** Ensure you have Python 3.9+ and one of the following deep learning frameworks installed: PyTorch 2.1+, TensorFlow 2.6+, or Flax 0.4.1+.

2.  **Virtual Environment (Recommended):** Create and activate a virtual environment using `venv` or `uv` to manage dependencies.

    ```bash
    # venv
    python -m venv .my-env
    source .my-env/bin/activate

    # uv
    uv venv .my-env
    source .my-env/bin/activate
    ```

3.  **Install Transformers:** Install the library using `pip` or `uv`.
    ```bash
    # pip
    pip install "transformers[torch]"

    # uv
    uv pip install "transformers[torch]"
    ```

4.  **Install from Source (Optional):** For the latest changes or contributions, install directly from the source code. Be aware that this may not be the most stable version.
    ```bash
    git clone https://github.com/huggingface/transformers.git
    cd transformers

    # pip
    pip install .[torch]

    # uv
    uv pip install .[torch]
    ```

## Quickstart

The [Pipeline](https://huggingface.co/docs/transformers/pipeline_tutorial) API is a high-level tool that makes it easy to start using Transformers models right away.

```python
from transformers import pipeline

# Text Generation Example
pipeline = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B")
print(pipeline("the secret to baking a really good cake is ")
```

## Examples and Use Cases

Explore how to use the `Pipeline` API for a variety of tasks:

<details>
<summary>Automatic Speech Recognition</summary>

```python
from transformers import pipeline

pipeline = pipeline(task="automatic-speech-recognition", model="openai/whisper-large-v3")
print(pipeline("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
```

</details>

<details>
<summary>Image Classification</summary>

<h3 align="center">
    <a><img src="https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"></a>
</h3>

```python
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="facebook/dinov2-small-imagenet1k-1-layer")
print(pipeline("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
```

</details>

<details>
<summary>Visual Question Answering</summary>

<h3 align="center">
    <a><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-few-shot.jpg"></a>
</h3>

```python
from transformers import pipeline

pipeline = pipeline(task="visual-question-answering", model="Salesforce/blip-vqa-base")
print(pipeline(
    image="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-few-shot.jpg",
    question="What is in the image?",
))
```

</details>

## Why Choose Transformers?

*   **Ease of Use:** Simple, intuitive API for accessing state-of-the-art models.
*   **Efficiency:** Reduce compute costs by utilizing pre-trained models and training from scratch less frequently.
*   **Flexibility:** Train, evaluate, and deploy models using your preferred framework.
*   **Customization:** Adapt models to your specific use cases, experiments, or projects.

## Why You Might Not Choose Transformers

*   **Not a Modular Toolbox:** The library is designed for rapid iteration on existing models, not for building custom neural network architectures from scratch.
*   **Training API Focus:** The training API is primarily designed for PyTorch models provided by Transformers.
*   **Example Scripts:** The example scripts might require modifications to fit your particular project.

## Projects Built with Transformers

Explore the [awesome-transformers](./awesome-transformers.md) page to discover 100+ incredible projects built using this library.  Contribute your project!

## Example Models

Explore model hubs for examples across different modalities.

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