[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# NVIDIA NeMo: Build, Customize, and Deploy Generative AI Models

NVIDIA NeMo is a cloud-native, open-source framework designed to streamline the development of Large Language Models (LLMs), Multimodal Models (MMs), and other AI applications.  [Explore the NeMo Repository](https://github.com/NVIDIA/NeMo).

## Key Features:

*   **Comprehensive AI Domains:**  Supports LLMs, MMs, Automatic Speech Recognition (ASR), Text-to-Speech (TTS), and Computer Vision (CV).
*   **Scalable Training:**  Train models efficiently across thousands of GPUs.
*   **Modular Architecture:**  Built with PyTorch Lightning for flexibility and ease of use.
*   **Pre-trained Models & Recipes:** Access pre-trained models and example scripts for quick starts.
*   **Model Alignment:**  Supports state-of-the-art alignment methods like SteerLM, DPO, and RLHF.
*   **PEFT Support:**  Integrates Parameter Efficient Fine-Tuning (PEFT) techniques such as LoRA, Adapters, and more.
*   **Deployment and Optimization:** Deploy and optimize models with NVIDIA NeMo Microservices and Riva.
*   **Integration with Cutting-Edge Technology:** Leverages NVIDIA Transformer Engine, NVIDIA Megatron Core, and NVIDIA H200 GPUs for optimal performance.

## What's New

*   **NeMo 2.0:**  Focuses on modularity and ease-of-use, featuring a Python-based configuration, modular abstractions, and scalability.
*   **Hugging Face Integration:** Seamless support for Hugging Face models via AutoModel.
*   **Blackwell Support:** Enhanced performance on GB200 & B200.
*   **New Model Support:** Expanded support for community models like Llama 4, Flux, Hyena & Evo2, Qwen2-VL, Qwen2.5, Gemma3, Qwen3-30B&32B.

### Latest News
*   Pretrain and finetune Hugging Face models via AutoModel (2025-05-19)
*   Training on Blackwell using Nemo (2025-05-19)
*   Training Performance on GPU Tuning Guide (2025-05-19)
*   New Models Support (2025-05-19)
*   NeMo Framework 2.0 (2025-05-19)
*   New Cosmos World Foundation Models Support (2025-01-09)
*   Large Language Models and Multimodal Models Updates (2024/06/12 - 2024/11/06)
*   Speech Recognition Enhancements (2024/04/18 - 2024/09/24)

## Getting Started

NeMo provides multiple ways to get started:

*   **Pre-trained Models:** Leverage readily available models on [Hugging Face Hub](https://huggingface.co/models?library=nemo&sort=downloads&search=nvidia) and [NVIDIA NGC](https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC).
*   **Tutorials:**  Follow comprehensive [tutorials](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/starthere/tutorials.html) using Google Colab or the NGC NeMo Framework Container.
*   **Playbooks:** Utilize [playbooks](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html) to train models with the NeMo Framework Launcher (v1.0).
*   **Example Scripts:** Access [example scripts](https://github.com/NVIDIA/NeMo/tree/main/examples) for advanced training and fine-tuning.

### Installation

Choose the installation method that best suits your needs:

*   **Conda / Pip:** Recommended for ASR and TTS, limited feature-completeness.
*   **NGC PyTorch Container:** Source installation into a highly optimized container.
*   **NGC NeMo Container:** Ready-to-go solution for maximum performance.

[Installation Instructions](#install-nemo-framework)

### Requirements
*   Python 3.10 or above
*   Pytorch 2.5 or above
*   NVIDIA GPU (for training)

## Developer Documentation

Access detailed documentation for the latest and stable versions:

| Version | Status | Description |
| ------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Latest  | [![Documentation Status](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)     | [Documentation of the latest (i.e. main) branch.](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)          |
| Stable  | [![Documentation Status](https://readthedocs.com/projects/nvidia-nemo/badge/?version=stable)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/) | [Documentation of the stable (i.e. most recent release)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/) |

## Contribute

We welcome community contributions!  Refer to [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md) for details.

## Stay Connected

*   **Discussions:**  Find answers and join discussions on the [NeMo Discussions board](https://github.com/NVIDIA/NeMo/discussions).
*   **Publications:** Explore publications utilizing the NeMo Framework [publications](https://nvidia.github.io/NeMo/publications/).
*   **Blogs:** Stay up-to-date with the latest developments via NVIDIA's blogs and other resources.

## Licenses

NeMo is licensed under the [Apache License 2.0](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file).