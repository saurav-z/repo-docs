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

<p align="center">
  <b><a href="https://github.com/huggingface/transformers">Hugging Face Transformers</a> empowers you to easily access and utilize the latest pre-trained models for a wide range of AI tasks.</b>
</p>

<h3 align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_as_a_model_definition.png"/>
</h3>

## Key Features

*   **Comprehensive Model Coverage:** Access over 1 million pre-trained models across various modalities like text, vision, audio, and multimodal tasks, all available on the [Hugging Face Hub](https://huggingface.co/models).
*   **Unified API:** Use a single, intuitive API for both inference and training, simplifying your workflow.
*   **Easy-to-Use Pipelines:** Leverage high-level pipelines for tasks like text generation, image classification, and speech recognition with minimal code.
*   **Flexible Framework Compatibility:** Seamlessly integrate with popular frameworks like PyTorch, TensorFlow, and Flax.
*   **Simplified Customization:**  Adapt models to your specific needs with easily customizable components.
*   **Lower Compute Costs and Reduced Carbon Footprint**: Leverage shared, pre-trained models to decrease the need for training from scratch.

## Getting Started

### Installation

Install Transformers with the following methods:

**Using `pip`:**

```bash
pip install "transformers[torch]"
```

**From Source:**

```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install .[torch]
```

**uv installation**

```bash
uv venv .my-env
source .my-env/bin/activate
uv pip install "transformers[torch]"
```

### Quickstart with Pipelines

The `Pipeline` API provides a straightforward way to utilize pre-trained models for various tasks. Here's how to get started with text generation:

```python
from transformers import pipeline

generator = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B")
result = generator("the secret to baking a really good cake is ")
print(result[0]['generated_text'])
```

**Example Tasks:**

*   **Automatic Speech Recognition:**

    ```python
    from transformers import pipeline

    asr = pipeline(task="automatic-speech-recognition", model="openai/whisper-large-v3")
    result = asr("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
    print(result['text'])
    ```
*   **Image Classification:**

    ```python
    from transformers import pipeline

    classifier = pipeline(task="image-classification", model="facebook/dinov2-small-imagenet1k-1-layer")
    result = classifier("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
    print(result)
    ```
*   **Visual Question Answering:**

    ```python
    from transformers import pipeline

    vqa = pipeline(task="visual-question-answering", model="Salesforce/blip-vqa-base")
    result = vqa(
        image="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/idefics-few-shot.jpg",
        question="What is in the image?",
    )
    print(result)
    ```

## Why Use Transformers?

*   **Ease of Use:** Simple API, high-performance models, and low barrier to entry.
*   **Cost-Effective:** Leverage pre-trained models to reduce compute costs and carbon footprint.
*   **Framework Flexibility:** Train, evaluate, and deploy models using your preferred framework.
*   **Customization:** Easily tailor models to your specific needs.

## Why Might You Not Use Transformers?

*   This library is not a modular toolbox of building blocks for neural nets. The code in the model files is not refactored with additional abstractions on purpose, so that researchers can quickly iterate on each of the models without diving into additional abstractions/files.
*   The training API is optimized to work with PyTorch models provided by Transformers. For generic machine learning loops, you should use another library like [Accelerate](https://huggingface.co/docs/accelerate).
*   The [example scripts](https://github.com/huggingface/transformers/tree/main/examples) are only *examples*. They may not necessarily work out-of-the-box on your specific use case and you'll need to adapt the code for it to work.

## Explore Further

*   **[Hugging Face Hub](https://huggingface.co/models):** Discover over a million pre-trained models.
*   **Documentation:** Explore the official [documentation](https://huggingface.co/docs/transformers/index) for detailed information and tutorials.
*   **Example Models:**
    *   [Audio Models](https://huggingface.co/models?pipeline_tag=audio-classification)
    *   [Computer Vision Models](https://huggingface.co/models?pipeline_tag=image-classification)
    *   [Multimodal Models](https://huggingface.co/models?pipeline_tag=image-to-text)
    *   [NLP Models](https://huggingface.co/models?pipeline_tag=text-classification)
*   **[Awesome Transformers](awesome-transformers.md):** Explore community projects built with Transformers.

## Citation

If you use the 🤗 Transformers library, please cite the following paper:

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