---
# Transformers: State-of-the-Art Models for NLP, Computer Vision, and More

Harness the power of cutting-edge AI with the Hugging Face Transformers library, a comprehensive toolkit for easily accessing and using pre-trained models.  [Explore the original repo here](https://github.com/huggingface/transformers).

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-light.svg">
    <img alt="Hugging Face Transformers Library" src="https://huggingface.co/datasets/huggingface/documentation-images/raw/main/transformers-logo-light.svg" width="352" height="59" style="max-width: 100%;">
  </picture>
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

*   **Wide Range of Models:** Access to a vast collection of pre-trained models for various tasks across text, computer vision, audio, video, and multimodal applications.
*   **Easy-to-Use Interface:**  Simple and intuitive API for both inference and training, allowing you to quickly get started.
*   **Flexibility and Customization:** Easily adapt models and examples to your specific needs, with consistent exposure of model internals.
*   **Cross-Framework Compatibility:** Seamlessly switch between PyTorch, TensorFlow, and Flax.
*   **Pre-trained Checkpoints:** Leverage over 1 million pre-trained models available on the Hugging Face Hub.
*   **Community & Resources:** Access a large community with a dedicated page listing incredible projects built with Transformers.

## Installation

### Requirements
Transformers requires:
*   Python 3.9+
*   PyTorch 2.1+ or TensorFlow 2.6+ or Flax 0.4.1+

### Step-by-step installation
1.  **Virtual Environment:** Create and activate a virtual environment using `venv` or `uv`.

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

    *Install from source if you want the latest changes (may not be stable):*
    ```bash
    git clone https://github.com/huggingface/transformers.git
    cd transformers
    # pip
    pip install .[torch]
    # uv
    uv pip install .[torch]
    ```
## Quickstart: Get Started with the Pipeline API

Quickly leverage state-of-the-art models using the `Pipeline` API for various tasks.

```python
from transformers import pipeline

# Example for text generation
pipeline = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B")
print(pipeline("the secret to baking a really good cake is "))

# Chat example
import torch
chat = [
    {"role": "system", "content": "You are a sassy, wise-cracking robot as imagined by Hollywood circa 1986."},
    {"role": "user", "content": "Hey, can you tell me any fun things to do in New York?"}
]
pipeline = pipeline(task="text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
response = pipeline(chat, max_new_tokens=512)
print(response[0]["generated_text"][-1]["content"])

```

## Why Use Transformers?

*   **Simplified Development:**  Easy access to state-of-the-art models with a unified API.
*   **Reduced Costs:**  Leverage pre-trained models to minimize training time and infrastructure expenses.
*   **Framework Agnostic:**  Seamlessly train and deploy models across various frameworks.
*   **Customizable:**  Adapt and fine-tune models to your specific needs and use cases.

## Why *Not* to Use Transformers

*   **Not a modular building block toolbox:** The codebase is not refactored with additional abstractions on purpose, so that researchers can quickly iterate on each of the models without diving into additional abstractions/files.
*   **Training API limitations:** The training API is optimized to work with PyTorch models provided by Transformers. For generic machine learning loops, you should use another library like [Accelerate](https://huggingface.co/docs/accelerate).
*   **Example scripts:** The example scripts are only *examples*. They may not necessarily work out-of-the-box on your specific use case and you'll need to adapt the code for it to work.

## Example Models
*   **Audio:**  Whisper, Moonshine, Wav2Vec2, Moshi, MusicGen, Bark
*   **Computer Vision:**  SAM, DepthPro, DINO v2, SuperGlue, RT-DETRv2, VitPose, OneFormer, VideoMAE
*   **Multimodal:**  Qwen2-Audio, LayoutLMv3, Qwen-VL, BLIP-2, GOT-OCR2, TAPAS, Emu3, Llava-OneVision, Llava, Kosmos-2
*   **NLP:**  ModernBERT, Gemma, Mixtral, BART, T5, Llama, Qwen

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