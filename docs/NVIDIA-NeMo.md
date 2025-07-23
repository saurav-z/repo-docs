[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# NVIDIA NeMo: Build, Customize, and Deploy Generative AI Models with Ease

NVIDIA NeMo is a powerful, cloud-native framework that empowers researchers and developers to efficiently create and deploy state-of-the-art generative AI models for various domains, including Large Language Models (LLMs), Multimodal Models (MMs), Automatic Speech Recognition (ASR), Text-to-Speech (TTS), and Computer Vision (CV). ([See the original repo](https://github.com/NVIDIA/NeMo)).

## Key Features

*   **LLMs and MMs:** Train, align, and customize Transformer-based models with cutting-edge distributed training techniques like Tensor Parallelism, Pipeline Parallelism, FSDP, and MoE. Supports FP8 training on NVIDIA Hopper GPUs with NVIDIA Transformer Engine and leverages NVIDIA Megatron Core for scaling Transformer model training.
*   **Speech AI:** Optimize and deploy ASR and TTS models for production use cases with NVIDIA Riva.
*   **Multimodal Models:** Support for training and customizing NVIDIA Cosmos world foundation models.
*   **Model Alignment & PEFT:** Supports alignment methods such as SteerLM, DPO, and RLHF via NVIDIA NeMo Aligner, as well as PEFT techniques like LoRA, P-Tuning, Adapters, and IA3.
*   **Modular and Scalable:** Benefit from PyTorch Lightning's modular abstractions, NeMo 2.0's Python-based configuration, and seamless scaling across thousands of GPUs using NeMo-Run.
*   **Comprehensive Support:** Pre-trained models readily available on Hugging Face Hub and NVIDIA NGC, with extensive tutorials, example scripts, and playbooks to facilitate model development.

## What's New

*   **Hugging Face Models:** NeMo now supports Hugging Face models via AutoModel
*   **Blackwell Support:** Improved performance on GB200 & B200
*   **GPU Tuning Guide:** A new performance tuning guide to achieve optimal throughput.
*   **New Model Support**: Llama 4, Flux, Llama Nemotron, Hyena & Evo2, Qwen2-VL, Qwen2.5, Gemma3, Qwen3-30B&32B

**More Details:** Find the latest updates and detailed information in the [NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html).

## Getting Started

1.  **Explore:** Leverage pre-trained models from [Hugging Face Hub](https://huggingface.co/models?library=nemo&sort=downloads&search=nvidia) and [NVIDIA NGC](https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC).
2.  **Tutorials:** Run tutorials on [Google Colab](https://colab.research.google.com) or with the [NGC NeMo Framework Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo).
3.  **Playbooks:** Train NeMo models using the [NeMo Framework Playbooks](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html) and the [NeMo-Run](https://github.com/NVIDIA/NeMo-Run) tool.
4.  **Advanced Training:** Utilize example scripts for multi-GPU/multi-node training.

## Installation

Choose the installation method that best suits your needs:

*   **Conda / Pip:** Recommended for ASR and TTS, explore NeMo on any supported platform.
*   **NGC PyTorch container:** Install NeMo from source into a highly optimized container.
*   **NGC NeMo container:** Ready-to-go solution for highest performance.

For detailed installation steps, consult the [Installation Guide](#install-nemo-framework).

## Requirements

*   Python 3.10 or above
*   PyTorch 2.5 or above
*   NVIDIA GPU (for training)

## Developer Documentation

Comprehensive documentation is available:

*   [Latest Documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
*   [Stable Documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/)

## Contribute

We welcome community contributions! Review the [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md) for details.

## Publications

Explore a curated list of [publications](https://nvidia.github.io/NeMo/publications/) utilizing the NeMo Framework.

## Discussions

Find answers and engage in discussions on the NeMo [Discussions board](https://github.com/NVIDIA/NeMo/discussions).

## Licenses

NVIDIA NeMo is licensed under the [Apache License 2.0](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file).
```
Key improvements and SEO considerations:

*   **Clear Headline:**  The main title is optimized for search ("NVIDIA NeMo: Build, Customize, and Deploy Generative AI Models with Ease").
*   **Concise Hook:** The one-sentence description is engaging and uses keywords.
*   **Keyword Optimization:** The description includes relevant keywords like "Large Language Models (LLMs)", "Multimodal Models (MMs)", "Automatic Speech Recognition (ASR)", "Text-to-Speech (TTS)", "Computer Vision (CV)", and others throughout.
*   **Structured Content:**  Uses headings (like "Key Features", "Getting Started", "Installation") and bullet points for readability and scannability, which improves SEO and user experience.
*   **Action-Oriented Language:** Uses verbs like "Build," "Customize," "Deploy," "Explore," and "Get Started" to encourage engagement.
*   **Internal Linking:** Includes links to key documentation, tutorials, and other sections within the README.
*   **External Linking:** Links back to the original repo for context and credibility.
*   **Up-to-Date Content:**  The "What's New" section summarizes recent updates, making the README more dynamic.
*   **Simplified "Key Features":**  A more concise list with links.
*   **Clear Installation Instructions:** Summarizes and provides a clear overview of installation options.
*   **Concise "Getting Started" Section:** Guides new users with clear, numbered steps.
*   **Contact and Support:** Links to discussions and publications.
*   **Licenses:** Mentions the license for legal clarity.