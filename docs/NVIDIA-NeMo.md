[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# NVIDIA NeMo: Build, Customize, and Deploy Generative AI Models with Ease

NVIDIA NeMo is a powerful, cloud-native framework enabling researchers and developers to efficiently create, customize, and deploy state-of-the-art generative AI models.  [Explore the original repository](https://github.com/NVIDIA/NeMo).

## Key Features

*   **Large Language Models (LLMs):** Train and fine-tune powerful LLMs.
*   **Multimodal Models (MMs):** Develop models that process multiple data types.
*   **Automatic Speech Recognition (ASR):** Build and optimize speech recognition models.
*   **Text-to-Speech (TTS):** Create high-quality speech synthesis models.
*   **Computer Vision (CV):** Leverage pre-trained models and build custom CV solutions.
*   **Modular Design:** Utilize PyTorch Lightning's modular abstractions for flexibility.
*   **Scalability:** Scale experiments across thousands of GPUs using NeMo-Run.
*   **Integration with Hugging Face:** Pretrain and finetune Hugging Face models.

## What's New

*   **Blackwell Support:** NeMo Framework has added Blackwell support, with performance benchmarks on GB200 & B200.
*   **Hugging Face Integration:** AutoModel support, enabling easy use of Hugging Face models.
*   **Performance Tuning Guide:** Comprehensive guide for performance tuning.
*   **Latest Model Support:** Support for Llama 4, Flux, Llama Nemotron, Hyena & Evo2, Qwen2-VL, Qwen2.5, Gemma3, Qwen3-30B&32B.
*   **NVIDIA Cosmos World Foundation Models:**  Support for training and customizing these models for video generation.
*   **NeMo 2.0:** Prioritizes modularity and ease-of-use with Python-based configuration and modular abstractions.

## Introduction

NVIDIA NeMo is designed to streamline the development of generative AI models across various domains, including LLMs, MMs, ASR, TTS, and CV. It provides a scalable and user-friendly environment for creating, customizing, and deploying AI models. The framework leverages existing code, pre-trained models, and cutting-edge techniques to accelerate AI development.  For technical documentation, please see the [NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html).

## Getting Started

*   **Pre-trained Models:** Access state-of-the-art, pre-trained models on [Hugging Face Hub](https://huggingface.co/models?library=nemo&sort=downloads&search=nvidia) and [NVIDIA NGC](https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC).
*   **Tutorials:** Follow the [tutorials](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/starthere/tutorials.html) to get started quickly.
*   **Playbooks:** Use the [playbooks](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html) to train NeMo models with the NeMo Framework Launcher.
*   **Example Scripts:** Explore [example scripts](https://github.com/NVIDIA/NeMo/tree/main/examples) for advanced use cases.

## Installation

Choose the installation method that best suits your needs:

*   **Conda / Pip:** Ideal for general exploration and ASR/TTS domains. See the detailed instructions in the original README.
*   **NGC PyTorch Container:**  Recommended for those wanting to install from source within an optimized container. See the original README for instructions.
*   **NGC NeMo Container:** A ready-to-go solution for peak performance.

## Requirements

*   Python 3.10 or above
*   PyTorch 2.5 or above
*   NVIDIA GPU (for model training)

## Developer Documentation

Refer to the documentation for more details:

| Version | Status                                                                                                                                                              | Description                                                                                                                    |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Latest  | [![Documentation Status](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)     | [Documentation of the latest (i.e. main) branch.](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)          |
| Stable  | [![Documentation Status](https://readthedocs.com/projects/nvidia-nemo/badge/?version=stable)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/) | [Documentation of the stable (i.e. most recent release)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/) |

## Contribute

We welcome community contributions! See [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md) for guidelines.

## Stay Updated

*   Check the [Discussions board](https://github.com/NVIDIA/NeMo/discussions) for FAQs and community engagement.
*   Explore the growing list of [publications](https://nvidia.github.io/NeMo/publications/) that use NeMo.
*   Read the [blogs](https://developer.nvidia.com/blog/nemo-amazon-titan/) for the latest advancements and use cases.

## License

NeMo is licensed under the [Apache License 2.0](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file).