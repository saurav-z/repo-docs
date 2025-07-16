[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# NVIDIA NeMo: The AI Framework for LLMs, Multimodal AI, and Speech Applications

**NVIDIA NeMo is a powerful, cloud-native framework designed to accelerate the development, customization, and deployment of cutting-edge generative AI models, with support for Large Language Models (LLMs), Multimodal Models (MMs), and speech applications.**

For the full details, visit the original [NVIDIA NeMo repository](https://github.com/NVIDIA/NeMo).

## Key Features

*   **Large Language Models (LLMs):** Build, train, and fine-tune powerful LLMs with optimized performance and scalability.
*   **Multimodal Models (MMs):** Develop models that combine multiple data types like text, images, and audio.
*   **Automatic Speech Recognition (ASR):** Create and deploy state-of-the-art ASR models for accurate speech-to-text transcription.
*   **Text-to-Speech (TTS):** Generate high-quality synthetic speech from text.
*   **Computer Vision (CV):** Implement and train computer vision models.
*   **Scalable Training:** Leverage distributed training techniques to scale experiments across thousands of GPUs.
*   **Pre-trained Models:** Utilize a wide range of pre-trained models available on Hugging Face Hub and NVIDIA NGC.
*   **Optimized for NVIDIA Hardware:** Benefit from optimized performance on NVIDIA GPUs and the NVIDIA ecosystem.
*   **Cloud-Native:** Designed for easy deployment in cloud environments.
*   **Modular and Flexible:** Offers a modular architecture with Python-based configuration for customization.

## What's New

*   **Hugging Face Model Integration:** Supports pretraining and finetuning Hugging Face models with AutoModel, starting with AutoModelForCausalLM and AutoModelForImageTextToText. ([Blog](https://developer.nvidia.com/blog/run-hugging-face-models-instantly-with-day-0-support-from-nvidia-nemo-framework))
*   **Blackwell Support:** Performance benchmarks on GB200 & B200.
*   **Performance Tuning Guide:** A comprehensive guide for performance tuning is available. ([Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance-guide.html))
*   **New Model Support:** Support for Llama 4, Flux, Llama Nemotron, Hyena & Evo2, Qwen2-VL, Qwen2.5, Gemma3, Qwen3-30B&32B.
*   **NeMo 2.0:** Focuses on modularity and ease-of-use.
*   **Cosmos World Foundation Model Support:**  Training and customization of the NVIDIA Cosmos collection of world foundation models.

## Getting Started

### Installation

Choose your preferred installation method:

*   **Conda / Pip:**  Install using `pip install "nemo_toolkit[all]"` within a Conda environment.
*   **NGC PyTorch Container:**  Install from source within a base NVIDIA PyTorch container (nvcr.io/nvidia/pytorch:25.01-py3).
*   **NGC NeMo Container:** Use a pre-built, optimized NeMo container (nvcr.io/nvidia/nemo:25.02).

### Resources

*   **User Guide:** Access comprehensive documentation for the NeMo Framework. ([NeMo Framework User Guide](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/))
*   **Tutorials:** Follow tutorials on Google Colab or using the NGC NeMo Framework Container.
*   **Example Scripts:** Explore example scripts for advanced training and fine-tuning.
*   **Discussions Board:** Find answers to your questions or start discussions. ([NeMo Discussions board](https://github.com/NVIDIA/NeMo/discussions))

## Contribute

We welcome community contributions!  See [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md) for details.

## Licenses

NeMo is licensed under the [Apache License 2.0](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file).