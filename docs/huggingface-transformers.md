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
    Empower your projects with cutting-edge AI using the Hugging Face Transformers library, providing access to a vast ecosystem of pretrained models.  
</p>

<h3 align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_as_a_model_definition.png"/>
</h3>

<p align="center">
    <a href="https://github.com/huggingface/transformers">
        <img alt="GitHub" src="https://img.shields.io/badge/View%20on%20GitHub-gray?logo=github">
    </a>
</p>

## Key Features

*   **Comprehensive Model Support:** Access a wide range of state-of-the-art models for text, computer vision, audio, video, and multimodal tasks.
*   **Easy to Use:** A unified API and high-level `Pipeline` for quick inference.
*   **Cost-Effective:** Leverage pretrained models to reduce compute time and costs.
*   **Framework Agnostic:** Seamlessly move models between PyTorch, TensorFlow, and Flax.
*   **Customizable:** Fine-tune and adapt models to your specific needs with provided examples.
*   **Vast Community:** Benefit from a thriving community with over 1 million model checkpoints on the Hugging Face Hub.

## What is Hugging Face Transformers?

The Hugging Face Transformers library is a model-definition framework, providing tools for both inference and training across various modalities. It serves as a central hub for model definitions, ensuring compatibility across different training frameworks (Axolotl, Unsloth, DeepSpeed, etc.), inference engines (vLLM, SGLang, etc.), and related libraries (llama.cpp, mlx, etc.). Transformers is designed to simplify and democratize the use of state-of-the-art models, offering simple, customizable, and efficient solutions.

## Installation

Install Transformers with:

```bash
pip install "transformers[torch]"
```

For detailed installation instructions, including setting up virtual environments and installing from source, refer to the [original repository](https://github.com/huggingface/transformers).

## Quickstart

Get started using the `Pipeline` API:

```python
from transformers import pipeline

# Text generation
pipeline = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B")
pipeline("the secret to baking a really good cake is ")

# Chat with a model
import torch
chat = [
    {"role": "system", "content": "You are a sassy, wise-cracking robot as imagined by Hollywood circa 1986."},
    {"role": "user", "content": "Hey, can you tell me any fun things to do in New York?"}
]
pipeline = pipeline(task="text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
response = pipeline(chat, max_new_tokens=512)
print(response[0]["generated_text"][-1]["content"])
```

## Example Models

Explore a variety of models across different modalities on the Hugging Face Hub, including:

*   **Audio:** Whisper, Moonshine, Wav2Vec2, Moshi, MusicGen, Bark
*   **Computer Vision:** SAM, DepthPro, DINO v2, SuperPoint, SuperGlue, RT-DETRv2, VitPose, OneFormer, VideoMAE
*   **Multimodal:** Qwen2-Audio, LayoutLMv3, Qwen-VL, BLIP-2, GOT-OCR2, TAPAS, Emu3, Llava-OneVision, Llava, Kosmos-2
*   **NLP:** ModernBERT, Gemma, Mixtral, BART, T5, Llama, Qwen

## Why Use Transformers?

*   **Ease of Use:** Simple API and high-level abstractions for rapid development.
*   **Efficiency:** Leverage pretrained models to reduce compute and costs.
*   **Flexibility:** Train and deploy models in various frameworks.
*   **Community Support:** Access a vast ecosystem of models and resources.

## Citation

If you use the Transformers library in your research, please cite the following paper:

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