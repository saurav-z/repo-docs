[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# NVIDIA NeMo: Build, Customize, and Deploy Generative AI Models with Ease

NVIDIA NeMo is a powerful, cloud-native framework enabling researchers and developers to efficiently create, customize, and deploy cutting-edge generative AI models for LLMs, Multimodal Models, ASR, TTS, and CV.  [Explore the original repo here](https://github.com/NVIDIA/NeMo).

## Key Features:

*   **Large Language Models (LLMs):** Train, fine-tune, and deploy state-of-the-art LLMs.
*   **Multimodal Models (MMs):** Develop models that process and generate information across multiple modalities (text, images, audio, video).
*   **Automatic Speech Recognition (ASR):** Build and optimize speech recognition models for high accuracy and speed.
*   **Text-to-Speech (TTS):** Create and deploy high-quality speech synthesis models.
*   **Computer Vision (CV):** Develop and train models for various computer vision tasks.
*   **Modular and Flexible:** Supports Python-based configurations for ease of use.
*   **Scalable Training:** Seamlessly scales training across thousands of GPUs using NeMo-Run and advanced parallelism techniques.
*   **Integration with Hugging Face:** Supports pre-training and fine-tuning of Hugging Face models.

## What's New

*   **Blackwell Support:** NeMo now supports the Blackwell architecture, with performance benchmarks available.
*   **Hugging Face Models via AutoModel:** Expanded support for Hugging Face models, including AutoModelForCausalLM and AutoModelForImageTextToText.
*   **New Model Support:** Includes support for Llama 4, Flux, Llama Nemotron, Hyena & Evo2, Qwen2-VL, Qwen2.5, Gemma3, and Qwen3-30B&32B.
*   **NeMo 2.0:**  Prioritizes modularity and ease-of-use.

## Getting Started

*   **Quickstart:** Access examples of using NeMo-Run to launch NeMo 2.0 experiments locally and on a slurm cluster.
*   **User Guide:** Detailed documentation on all features and capabilities.
*   **Recipes:** Find examples of large-scale runs using NeMo 2.0 and NeMo-Run.
*   **Migration Guide:** Step-by-step instructions for transitioning from NeMo 1.0 to 2.0.
*   **NGC and Hugging Face:** Explore a variety of pre-trained models available.

## Installation

NVIDIA NeMo Framework offers multiple installation methods to accommodate different needs:

*   **Conda / Pip:** Installs NeMo-Framework with native Pip into a virtual environment. (Recommended for ASR and TTS)
*   **NGC PyTorch container:** Install NeMo-Framework from source with feature-completeness into a highly optimized container.
*   **NGC NeMo container:** Ready-to-go solution of NeMo-Framework

### Requirements

*   Python 3.10 or above
*   Pytorch 2.5 or above
*   NVIDIA GPU (if you intend to do model training)

### Detailed Installation Guides

*   **Conda / Pip:** Detailed steps to install using Conda or Pip, including specific domain installations.
*   **NGC PyTorch container:** Instructions for installing NeMo within a base NVIDIA PyTorch container.
*   **NGC NeMo container:** Instructions to utilize a pre-built container with pre-installed dependencies.

## Deep Dive

*   **LLMs and MMs Training, Alignment, and Customization:** Describes the training process and support for advanced techniques like Tensor Parallelism, Pipeline Parallelism, Fully Sharded Data Parallelism, Mixed Precision Training and more.
*   **LLMs and MMs Deployment and Optimization:** Highlights the capabilities of optimizing models for deployment.
*   **Speech AI:** Explains optimizations for inference and deployment.
*   **NeMo Framework Launcher:** Introduces the cloud-native tool for streamlining the NeMo Framework experience and explains its recipes, scripts, and utilities.

## Resources:

*   [NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html)
*   [Discussions Board](https://github.com/NVIDIA/NeMo/discussions)
*   [Contribute to NeMo](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md)
*   [Publications](https://nvidia.github.io/NeMo/publications/)
*   [Blogs](https://developer.nvidia.com/blog/tag/nemo)

## License

NVIDIA NeMo is licensed under the [Apache License 2.0](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file).