[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# NVIDIA NeMo: Build, Customize, and Deploy Generative AI Models with Ease

**NVIDIA NeMo is a powerful, cloud-native framework designed to accelerate the development of Large Language Models (LLMs), Multimodal Models (MMs), and more.** ([Original Repo](https://github.com/NVIDIA/NeMo))

## Key Features

*   **Comprehensive Domain Support:** LLMs, MMs, Automatic Speech Recognition (ASR), Text-to-Speech (TTS), and Computer Vision (CV).
*   **Modular and Scalable:** Built with PyTorch Lightning and designed for efficient scaling across thousands of GPUs.
*   **Pre-trained Models:** Access a wide range of state-of-the-art pre-trained models on Hugging Face Hub and NVIDIA NGC.
*   **Customization and Fine-tuning:**  Easily customize and fine-tune models with cutting-edge techniques like LoRA, DPO, and RLHF.
*   **Deployment and Optimization:** Leverage NVIDIA Riva and NeMo Microservices for optimized inference and deployment.
*   **NVIDIA Transformer Engine Integration:** Optimize performance with FP8 training on NVIDIA Hopper GPUs and leverage NVIDIA Megatron Core for large-scale training.

## What's New

*   **Hugging Face Integration:**  Seamlessly work with Hugging Face models using AutoModel.
*   **Blackwell Support:**  Benefit from optimized performance on GB200 & B200.
*   **Performance Tuning Guide:** Access a comprehensive guide to optimize throughput.
*   **New Models:** Support for cutting-edge community models like Llama 4, Flux, and more.
*   **NeMo 2.0:**  A major release focusing on modularity, ease-of-use, and enhanced scalability.
*   **Cosmos Integration:**  Support for NVIDIA Cosmos world foundation models and video processing.

## Getting Started

### NeMo 2.0

*   [Quickstart](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/quickstart.html)
*   [NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/index.html)
*   [NeMo 2.0 Recipes](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/recipes)
*   [Feature Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/features/index.html#feature-guide)
*   [Migration Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/migration/index.html#migration-guide)

### Cosmos

*   [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/cosmos/collections/cosmos)
*   [Hugging Face](https://huggingface.co/collections/nvidia/cosmos-6751e884dc10e013a0a0d8e6)
*   [NeMo Curator](https://developer.nvidia.com/nemo-curator)
*   [Cosmos Diffusion models](https://github.com/NVIDIA/Cosmos/blob/main/cosmos1/models/diffusion/nemo/post_training/README.md)
*   [Cosmos Autoregressive models](https://github.com/NVIDIA/Cosmos/blob/main/cosmos1/models/autoregressive/nemo/post_training/README.md)

### Access Pre-trained Models and Tutorials

*   **Pretrained Models:** [Hugging Face Hub](https://huggingface.co/models?library=nemo&sort=downloads&search=nvidia), [NVIDIA NGC](https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC)
*   **Tutorials:** [NeMo Tutorials](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/starthere/tutorials.html)
*   **Example Scripts:** [Example Scripts](https://github.com/NVIDIA/NeMo/tree/main/examples)

## Installation

Choose the installation method that best fits your needs:

*   **Conda / Pip:**  Recommended for ASR and TTS, and to explore NeMo.
*   **NGC PyTorch Container:** Install from source within a highly optimized container.
*   **NGC NeMo Container:**  Ready-to-use, pre-built container for maximum performance.

**[Installation Instructions](#install-nemo-framework)**

## Requirements

*   Python 3.10 or above
*   PyTorch 2.5 or above
*   NVIDIA GPU (for model training)

## Developer Documentation

**[Developer Documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)**

## Contribute

We welcome contributions! See our [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md) for details.

## Publications

Explore the list of [publications](https://nvidia.github.io/NeMo/publications/) utilizing the NeMo Framework.

## Discussions Board

Find answers to your questions on the NeMo [Discussions Board](https://github.com/NVIDIA/NeMo/discussions).

## License

NVIDIA NeMo is licensed under the [Apache License 2.0](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file).