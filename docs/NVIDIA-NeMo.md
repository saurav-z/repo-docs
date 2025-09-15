# NVIDIA NeMo: Build, Customize, and Deploy Generative AI Models with Ease

NVIDIA NeMo is a cloud-native framework designed to empower researchers and developers to efficiently create, adapt, and deploy state-of-the-art generative AI models. [Access the original repo](https://github.com/NVIDIA/NeMo).

## Key Features

*   **Large Language Models (LLMs):** Train, fine-tune, and deploy powerful LLMs with optimized performance.
*   **Multimodal Models (MMs):** Develop models that process and generate text, images, and more.
*   **Automatic Speech Recognition (ASR):** Build accurate and efficient speech recognition systems.
*   **Text-to-Speech (TTS):** Create high-quality, natural-sounding speech synthesis models.
*   **Computer Vision (CV):** Leverage pre-trained models and tools for various computer vision tasks.
*   **Modular Design:** Built with PyTorch Lightning, offering flexibility and ease of customization.
*   **Scalability:** Designed for training across thousands of GPUs.
*   **Pre-trained Models:** Get started quickly with readily available models on Hugging Face Hub and NVIDIA NGC.
*   **Nemo-Run:** Simplified experiment management.
*   **Parameter Efficient Fine-Tuning (PEFT):** Supports techniques like LoRA, Adapters, and IA3.
*   **State-of-the-Art Techniques:** Leverages techniques like Tensor Parallelism, Pipeline Parallelism, FSDP, MoE, and Mixed Precision Training.

## Latest Updates

*   **Hugging Face Integration:** Seamlessly pretrain and fine-tune Hugging Face models using AutoModel, with specific support for AutoModelForCausalLM and AutoModelForImageTextToText.
*   **Blackwell Support:** Optimized performance benchmarks on GB200 & B200.
*   **New Models Support:** Includes Llama 4, Flux, Llama Nemotron, Hyena & Evo2, Qwen2-VL, Qwen2.5, Gemma3, Qwen3-30B&32B.
*   **Cosmos World Foundation Models Support:** Training and customization of the NVIDIA Cosmos models.
*   **Speech Recognition Improvements:** Up to 10x inference speed-up for ASR models.
*   **NeMo 2.0:** Major update prioritizing modularity and ease of use with Python-based configuration and modular abstractions.

## Getting Started

*   **Documentation:** Access the comprehensive [NeMo Framework User Guide](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/) for detailed technical information.
*   **Quickstart:** Utilize the [Quickstart](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/quickstart.html) guide for hands-on examples.
*   **Pre-trained Models:** Explore the vast library of pre-trained models on [Hugging Face Hub](https://huggingface.co/models?library=nemo&sort=downloads&search=nvidia) and [NVIDIA NGC](https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC).

## Installation

Choose the installation method that best fits your needs:

*   **Conda / Pip:**  Recommended for ASR and TTS domains.
*   **NGC PyTorch Container:** Install from source within a highly optimized NVIDIA PyTorch container.
*   **NGC NeMo Container:** Ready-to-go solution for optimal performance.

### Requirements

*   Python 3.10 or above
*   Pytorch 2.5 or above
*   NVIDIA GPU (for model training)

## Contribute

Contribute to the NeMo community!  See [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md) for details.

## Stay Connected

*   **Discussions:** Find answers and engage with the community on the NeMo [Discussions board](https://github.com/NVIDIA/NeMo/discussions).
*   **Publications:** Explore a growing list of [publications](https://nvidia.github.io/NeMo/publications/) using NeMo.