[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# NVIDIA NeMo: Build, Customize, and Deploy Generative AI Models at Scale

NVIDIA NeMo is a cloud-native framework empowering researchers and developers to build, customize, and deploy state-of-the-art generative AI models for diverse applications.  [Explore the NeMo repository](https://github.com/NVIDIA/NeMo).

## Key Features

*   **Large Language Models (LLMs):** Train and customize powerful LLMs.
*   **Multimodal Models (MMs):** Develop AI models that process and generate multiple data types, such as text and images.
*   **Automatic Speech Recognition (ASR):** Build accurate speech recognition models.
*   **Text-to-Speech (TTS):** Create high-quality speech synthesis systems.
*   **Computer Vision (CV):**  Develop and deploy advanced computer vision models.
*   **Flexible Configuration:** Python-based configuration for customization.
*   **Scalability:**  Seamlessly scale experiments across thousands of GPUs.
*   **Integration:** Compatible with Hugging Face models via AutoModel.

## Latest Updates

*   **Support for Hugging Face Models:**  Instantly run Hugging Face models with day-0 support. (2025-05-19)
*   **Blackwell Support:** Performance benchmarks added on GB200 & B200. (2025-05-19)
*   **Performance Tuning Guide:** Comprehensive guide for optimal throughput. (2025-05-19)
*   **New Model Support:**  Support for Llama 4, Flux, Llama Nemotron, Hyena & Evo2, Qwen2-VL, Qwen2.5, Gemma3, Qwen3-30B&32B. (2025-05-19)
*   **NeMo 2.0 Release:**  Focus on modularity and ease-of-use.

## Getting Started

*   **User Guide:** [NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html)
*   **Quickstart:** [NeMo 2.0 Quickstart](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/quickstart.html)
*   **Examples and Recipes:** Explore the [NeMo 2.0 Recipes](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/recipes) and [example scripts](https://github.com/NVIDIA/NeMo/tree/main/examples) for large-scale runs.
*   **NGC & Hugging Face:** Access pretrained models on [Hugging Face Hub](https://huggingface.co/models?library=nemo&sort=downloads&search=nvidia) and [NVIDIA NGC](https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC).

## Installation

Choose your installation method based on your needs:

*   **Conda / Pip:** For exploring NeMo on supported platforms. Recommended for ASR and TTS.
*   **NGC PyTorch Container:** Install from source in a highly optimized container.
*   **NGC NeMo Container:** Ready-to-use solution for maximum performance.

### Detailed Installation Instructions

[See the original README for detailed installation steps using Conda/Pip and NGC containers.](#install-nemo-framework)

## Developer Documentation

*   **Latest:** [![Documentation Status](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
*   **Stable:** [![Documentation Status](https://readthedocs.com/projects/nvidia-nemo/badge/?version=stable)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/)

## Contribute

We welcome contributions! Please refer to [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md).

## Resources

*   **Discussions:**  Get answers to your questions on the [Discussions board](https://github.com/NVIDIA/NeMo/discussions).
*   **Publications:** Explore [publications](https://nvidia.github.io/NeMo/publications/) utilizing the NeMo Framework.
*   **Blogs:** Learn more through the [Blogs section](#blogs).

## License

NeMo is licensed under the [Apache License 2.0](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file).