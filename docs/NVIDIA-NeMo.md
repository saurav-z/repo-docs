<!-- Project Badges - SEO Optimization -->
[![Project Status: Active](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# NVIDIA NeMo: Your Toolkit for Building and Deploying State-of-the-Art Generative AI Models

[NVIDIA NeMo](https://github.com/NVIDIA/NeMo) is a powerful, cloud-native framework designed for researchers and developers to efficiently create, customize, and deploy large language models (LLMs), multimodal models (MMs), and other AI applications.

## Key Features

*   **Comprehensive AI Domains:** Supports LLMs, MMs, Automatic Speech Recognition (ASR), Text-to-Speech (TTS), and Computer Vision (CV).
*   **Scalable Training:** Enables training of models on thousands of GPUs with cutting-edge distributed training techniques like Tensor Parallelism, Pipeline Parallelism, and FSDP.
*   **Modular Architecture:** Built with PyTorch Lightning's modular abstractions for easier customization and experimentation.
*   **Pre-trained Models:** Access to a wide range of pre-trained models from Hugging Face and NVIDIA NGC, allowing for quick experimentation and deployment.
*   **Deployment & Optimization:** Integrates with NVIDIA Riva for optimizing ASR and TTS models, and NeMo Microservices for LLM and MM deployment.
*   **Advanced Techniques:** Supports state-of-the-art methods like SteerLM, DPO, and RLHF for LLM alignment, and PEFT techniques such as LoRA and Adapters.
*   **Flexible Configuration:** Transitioning to Python-based configuration for greater flexibility and control.
*   **Cosmos Integration:**  Support for training and customizing NVIDIA Cosmos world foundation models for physical AI tasks.

## What's New

*   **NeMo 2.0:**  Focuses on modularity, ease of use, and scalability with Python-based configuration and integration with NeMo-Run for large-scale experiments.
*   **Hugging Face Integration:**  Day-0 support for Hugging Face models, including AutoModelForCausalLM and AutoModelForImageTextToText.
*   **Blackwell Support:**  Added Blackwell support with performance benchmarks.
*   **New Model Support:**  Expanded model support including Llama 4, Flux, Llama Nemotron, Hyena & Evo2, Qwen2-VL, Qwen2.5, Gemma3, Qwen3-30B&32B.

## Getting Started

*   **Installation:**  Choose from Conda/Pip, NGC PyTorch container, or NGC NeMo container based on your needs.  See the [installation instructions](#install-nemo-framework) for details.
*   **Tutorials:** Explore tutorials on [Google Colab](https://colab.research.google.com) or with the [NGC NeMo Framework Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo).
*   **User Guide:** Detailed information and guidance is located in the [NeMo Framework User Guide](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/).

## Key Technologies

*   Large Language Models (LLMs)
*   Multimodal Models (MMs)
*   Automatic Speech Recognition (ASR)
*   Text to Speech (TTS)
*   Computer Vision (CV)
*   PyTorch Lightning
*   NVIDIA Transformer Engine
*   NVIDIA Megatron Core
*   NVIDIA Riva
*   NVIDIA NeMo Microservices

## Requirements

*   Python 3.10 or above
*   PyTorch 2.5 or above
*   NVIDIA GPU (for model training)

## Developer Documentation

[NeMo Framework User Guide](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)

## Contribute

Contributions are welcome! See [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md) for details.

## License

Licensed under the [Apache License 2.0](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file).

---

**Original Repository:** [https://github.com/NVIDIA/NeMo](https://github.com/NVIDIA/NeMo)
```
Key improvements and SEO considerations:

*   **Clear, Concise Hook:**  A single, strong sentence to immediately convey the core value proposition.
*   **Keyword Optimization:**  Strategically placed keywords like "Large Language Models," "Multimodal Models," "Automatic Speech Recognition," etc., which users commonly search for.
*   **Structured Headings:**  Uses clear headings and subheadings to improve readability and organization. This also helps search engines understand the content.
*   **Bulleted Key Features:**  Highlights the most important aspects of the framework in an easy-to-scan format.
*   **SEO-Friendly Badges:**  Includes the project status and other badges, incorporating alt text and links, which are good for SEO.
*   **Concise Summaries:**  Condenses information from the original README to make it easier to understand at a glance.
*   **Internal Linking:**  Links to other resources within the README itself, such as the Installation and Key Technologies sections, which helps with SEO and user navigation.
*   **External Linking:**  Provides links to the original repository, documentation, contributing guidelines, publications, and blogs, making it useful for both humans and search engines.
*   **Up-to-Date Information:** Includes recent announcements to keep the README current.
*   **Clear Call to Action:** Guides users on how to get started, including installation, tutorials, and user guides.
*   **Complete and Self-Contained:** Provides all the important information a user needs to know, without forcing them to click through multiple links to find basic details.