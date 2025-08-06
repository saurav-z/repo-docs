[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# NVIDIA NeMo: Build, Customize, and Deploy State-of-the-Art Generative AI Models

## Introduction

NVIDIA NeMo is a powerful, cloud-native framework designed to accelerate the development of Large Language Models (LLMs), Multimodal Models (MMs), Automatic Speech Recognition (ASR), Text-to-Speech (TTS), and Computer Vision (CV) applications.  Built for researchers and PyTorch developers, NeMo enables you to efficiently create, customize, and deploy generative AI models leveraging existing code and pre-trained checkpoints. Access the source code at the [NVIDIA/NeMo](https://github.com/NVIDIA/NeMo) repository.

## Key Features

*   **Large Language Models (LLMs):** Train, fine-tune, and deploy cutting-edge LLMs.
*   **Multimodal Models (MMs):** Develop models that integrate multiple data types (text, images, audio).
*   **Automatic Speech Recognition (ASR):** Build and deploy advanced ASR models.
*   **Text-to-Speech (TTS):** Create high-quality, natural-sounding speech synthesis.
*   **Computer Vision (CV):** Implement CV models for various applications.
*   **Modular Design:** Built with PyTorch Lightning's modular abstractions for easy customization.
*   **Scalability:** Seamlessly scale training across thousands of GPUs using NeMo-Run.
*   **Optimized Performance:** Leverages NVIDIA Transformer Engine and Megatron Core for FP8 training and efficient scaling.
*   **PEFT Support:** Includes support for techniques like LoRA, P-Tuning, and Adapters for efficient fine-tuning.
*   **Deployment:**  Deploy and optimize models using NVIDIA NeMo Microservices and NVIDIA Riva.
*   **Cosmos Support:**  Full support for training and customizing NVIDIA Cosmos world foundation models.

## What's New in NeMo 2.0

NeMo 2.0 introduces several key enhancements:

*   **Python-Based Configuration:** Increased flexibility and control with Python configurations.
*   **Modular Abstractions:** Simplified adaptation and experimentation through PyTorch Lightning's modular approach.
*   **Scalability:** Efficiently run large-scale experiments using NeMo-Run.

For detailed information, refer to the [NeMo Framework User Guide](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/starthere/tutorials.html).

## Getting Started

### Installation

Choose the installation method that best suits your needs:

*   **Conda / Pip:** Install NeMo using `pip install "nemo_toolkit[all]"` (recommended for ASR and TTS). See [Conda / Pip](#conda--pip) for full instructions.
*   **NGC PyTorch Container:** Install from source with feature-completeness into a highly optimized container.
*   **NGC NeMo Container:** Use a pre-built, ready-to-go container for optimal performance.

#### NeMo 2.0 Quickstart
*   Refer to the [Quickstart](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/quickstart.html) for examples of using NeMo-Run to launch NeMo 2.0 experiments.
*   [NeMo 2.0 Recipes](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/recipes) contains additional examples of launching large-scale runs using NeMo 2.0 and NeMo-Run.
*   For an in-depth exploration of the main features of NeMo 2.0, see the [Feature Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/features/index.html#feature-guide).
*   To transition from NeMo 1.0 to 2.0, see the [Migration Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/migration/index.html#migration-guide) for step-by-step instructions.

### Getting Started with Cosmos

NeMo Curator and NeMo Framework support video curation and post-training of the Cosmos World Foundation Models, which are open and available on [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/cosmos/collections/cosmos) and [Hugging Face](https://huggingface.co/collections/nvidia/cosmos-6751e884dc10e013a0a0d8e6). For more information on video datasets, refer to [NeMo Curator](https://developer.nvidia.com/nemo-curator). To post-train World Foundation Models using the NeMo Framework for your custom physical AI tasks, see the [Cosmos Diffusion models](https://github.com/NVIDIA/Cosmos/blob/main/cosmos1/models/diffusion/nemo/post_training/README.md) and the [Cosmos Autoregressive models](https://github.com/NVIDIA/Cosmos/blob/main/cosmos1/models/autoregressive/nemo/post_training/README.md).

### Pre-trained Models and Tutorials

Leverage pre-trained NeMo models available on [Hugging Face Hub](https://huggingface.co/models?library=nemo&sort=downloads&search=nvidia) and [NVIDIA NGC](https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC). Explore our extensive [tutorials](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/starthere/tutorials.html) and [playbooks](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html) for quickstart guides and training recipes.

## Requirements

*   Python 3.10 or above
*   PyTorch 2.5 or above
*   NVIDIA GPU (for training)

## Developer Documentation

Access comprehensive documentation for the latest and stable versions:

| Version | Status                                                                                                                                                              | Description                                                                                                                    |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Latest  | [![Documentation Status](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)     | [Documentation of the latest (i.e. main) branch.](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)          |
| Stable  | [![Documentation Status](https://readthedocs.com/projects/nvidia-nemo/badge/?version=stable)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/) | [Documentation of the stable (i.e. most recent release)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/) |

## Contribute

We welcome community contributions!  Please review our [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md) for guidelines.

## Publications

Explore the growing list of [publications](https://nvidia.github.io/NeMo/publications/) that utilize the NeMo Framework.

## Community

*   Ask questions and engage with the community on the NeMo [Discussions board](https://github.com/NVIDIA/NeMo/discussions).

## License

NVIDIA NeMo is licensed under the [Apache License 2.0](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file).