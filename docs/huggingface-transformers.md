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
  Easily access and utilize the most advanced pre-trained models for a wide range of AI tasks.
</p>

<h3 align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_as_a_model_definition.png"/>
</h3>

<p align="center">
    <b><a href="https://github.com/huggingface/transformers">Explore the Transformers Repository</a></b>
</p>

## Key Features

*   **Extensive Model Library:** Access a vast collection of pre-trained models for text, vision, audio, video, and multimodal tasks.
*   **Unified API:** Simplify your workflow with a consistent and intuitive API for all models.
*   **Easy Customization:** Adapt models to your specific needs and fine-tune them with minimal code.
*   **Cross-Framework Compatibility:** Seamlessly integrate models with popular frameworks like PyTorch, TensorFlow, and Flax.
*   **Lower Compute Costs:** Leverage pre-trained models to reduce training time and resource consumption.
*   **Community Driven:** Benefit from a large and active community with extensive model checkpoints on the [Hugging Face Hub](https://huggingface.co/models)
*   **Wide Support:** Supports a plethora of adjacent modeling libraries.

## Installation

Get started quickly with Transformers!

### Prerequisites
*   Python 3.9+
*   [PyTorch](https://pytorch.org/get-started/locally/) 2.1+ or [TensorFlow](https://www.tensorflow.org/install/pip) 2.6+ or [Flax](https://flax.readthedocs.io/en/latest/) 0.4.1+

1.  **Create and activate a virtual environment** (using [venv](https://docs.python.org/3/library/venv.html) or [uv](https://docs.astral.sh/uv/)).

    ```bash
    # venv
    python -m venv .my-env
    source .my-env/bin/activate
    # uv
    uv venv .my-env
    source .my-env/bin/activate
    ```

2.  **Install Transformers.**

    ```bash
    # pip
    pip install "transformers[torch]"

    # uv
    uv pip install "transformers[torch]"
    ```

3.  **Install from Source (for the latest changes - may be unstable):**

    ```bash
    git clone https://github.com/huggingface/transformers.git
    cd transformers

    # pip
    pip install .[torch]

    # uv
    uv pip install .[torch]
    ```

## Quickstart

The `Pipeline` API provides a simple way to get started with Transformers.  It simplifies the process of using pre-trained models for inference.

```python
from transformers import pipeline

# Text Generation Example
generator = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B")
result = generator("the secret to baking a really good cake is ")
print(result[0]['generated_text'])

# Chat Example
import torch
chat = [
    {"role": "system", "content": "You are a sassy, wise-cracking robot as imagined by Hollywood circa 1986."},
    {"role": "user", "content": "Hey, can you tell me any fun things to do in New York?"}
]
pipeline = pipeline(task="text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
response = pipeline(chat, max_new_tokens=512)
print(response[0]["generated_text"][-1]["content"])
```

Explore the examples to see how the `Pipeline` API works with different modalities and tasks, including automatic speech recognition, image classification, and visual question answering.

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

*   **Ease of Use:**  Simple, state-of-the-art models with a user-friendly API.
*   **Efficiency:** Leverage pre-trained models to save time and resources.
*   **Flexibility:** Train and use models with PyTorch, JAX, or TF2.0.
*   **Customization:** Adapt models and examples to fit your specific needs.

## Why Not Use Transformers?

*   Not a modular toolbox for building neural networks (focused on existing models).
*   The training API is optimized for PyTorch models provided by Transformers, use [Accelerate](https://huggingface.co/docs/accelerate) for generic ML loops.
*   Example scripts may need adaptation for your specific use case.

## 100+ Projects with Transformers

Explore the community-driven [awesome-transformers](./awesome-transformers.md) page to discover incredible projects built with Transformers.  Contribute your own!

## Example Models

Discover a variety of models across different modalities.  Visit the [Hugging Face Hub](https://huggingface.co/models) for more!

*   **(Audio)**:  Audio classification with [Whisper](https://huggingface.co/openai/whisper-large-v3-turbo), automatic speech recognition with [Moonshine](https://huggingface.co/UsefulSensors/moonshine), and more.
*   **(Computer Vision)**:  Image classification with [DINO v2](https://huggingface.co/facebook/dinov2-base), object detection with [RT-DETRv2](https://huggingface.co/PekingU/rtdetr_v2_r50vd), and more.
*   **(Multimodal)**:  Image captioning [BLIP-2](https://huggingface.co/Salesforce/blip2-opt-2.7b), visual question answering with [Llava](https://huggingface.co/llava-hf/llava-1.5-7b-hf), and more.
*   **(NLP)**:  Text generation with [Llama](https://huggingface.co/meta-llama/Llama-3.2-1B), summarization with [BART](https://huggingface.co/facebook/bart-large-cnn), and more.

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