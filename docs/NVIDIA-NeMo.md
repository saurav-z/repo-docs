[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# NVIDIA NeMo: Build, Customize, and Deploy Generative AI Models with Ease

NVIDIA NeMo is a powerful, cloud-native framework empowering researchers and developers to efficiently build, customize, and deploy cutting-edge generative AI models. ([See the original repository](https://github.com/NVIDIA/NeMo)).

## Key Features

*   **Large Language Models (LLMs):** Train, customize, and deploy state-of-the-art LLMs.
*   **Multimodal Models (MMs):** Develop models that combine different data types like text and images.
*   **Automatic Speech Recognition (ASR):** Build and optimize models for speech-to-text tasks.
*   **Text-to-Speech (TTS):** Create high-quality speech synthesis systems.
*   **Computer Vision (CV):** Implement and train advanced computer vision models.
*   **Modular and Flexible:** Designed for ease of use and experimentation with PyTorch Lightning and Python-based configurations.
*   **Scalable Training:** Supports training across thousands of GPUs with techniques like Tensor Parallelism, Pipeline Parallelism, FSDP, and MoE.
*   **Performance Optimization:** Integrates NVIDIA Transformer Engine for FP8 training and leverages NVIDIA Megatron Core for scaling.
*   **Deployment and Optimization:** Integrates with NVIDIA Riva for ASR/TTS and NeMo Microservices for LLMs/MMs.
*   **Community Support:** Active discussions and a dedicated forum for support and collaboration.

## What's New

*   **Hugging Face Integration:** Seamlessly pretrain and fine-tune Hugging Face models using AutoModel.
*   **Blackwell Support:** Performance benchmarks for GB200 & B200.
*   **Performance Tuning Guide:** Comprehensive guide for achieving optimal throughput.
*   **New Model Support:** Llama 4, Flux, Llama Nemotron, Hyena & Evo2, Qwen2-VL, Qwen2.5, Gemma3, Qwen3-30B&32B.
*   **NeMo 2.0:** Enhanced modularity and ease-of-use.

## Getting Started

*   **Quickstart:** Get started with NeMo 2.0 experiments using NeMo-Run: [Quickstart](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/quickstart.html)
*   **User Guide:** Comprehensive documentation: [NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/index.html).
*   **Recipes:** Large-scale run examples using NeMo 2.0 and NeMo-Run: [NeMo 2.0 Recipes](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/recipes).
*   **Feature Guide:** In-depth exploration of NeMo 2.0 features: [Feature Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/features/index.html#feature-guide).
*   **Migration Guide:** Instructions for transitioning from NeMo 1.0 to 2.0: [Migration Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/migration/index.html#migration-guide)

## Cosmos Support

*   **Cosmos World Foundation Models:** Support for training and customizing NVIDIA Cosmos.
*   **NeMo Curator:** Accelerate video processing with NeMo Curator: [NeMo Curator](https://developer.nvidia.com/nemo-curator).

## Training, Alignment, and Customization

*   Leverage [Lightning](https://github.com/Lightning-AI/lightning) for scalable training.
*   Utilize cutting-edge distributed training techniques.
*   Integrates [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine) and [NVIDIA Megatron Core](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core).
*   Supports alignment methods such as SteerLM, DPO, and RLHF (see [NVIDIA NeMo Aligner](https://github.com/NVIDIA/NeMo-Aligner)).
*   Supports PEFT techniques like LoRA, P-Tuning, Adapters, and IA3.

## Deployment and Optimization

*   Deploy and optimize LLMs and MMs with [NVIDIA NeMo Microservices](https://developer.nvidia.com/nemo-microservices-early-access).
*   Optimize ASR and TTS models with [NVIDIA Riva](https://developer.nvidia.com/riva).

## Installation

Choose the installation method that best suits your needs:

*   **Conda / Pip:** Recommended for ASR and TTS domains. Install with `pip install "nemo_toolkit[all]"` or use a Git reference (e.g., `pip install ".[all]"`).
*   **NGC PyTorch Container:** Install from source with feature-completeness into a highly optimized container.
*   **NGC NeMo Container:** Ready-to-go solution for highest performance.
  See the original README for detailed install instructions

## Requirements

*   Python 3.10 or above
*   Pytorch 2.5 or above
*   NVIDIA GPU (for training)

## Developer Documentation

*   [Latest Documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
*   [Stable Documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/)

## Contribute

We welcome contributions!  See [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md) for details.

## Publications

Explore publications utilizing NeMo Framework: [Publications](https://nvidia.github.io/NeMo/publications/)