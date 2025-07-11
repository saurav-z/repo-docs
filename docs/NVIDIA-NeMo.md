# NVIDIA NeMo: Build, Customize, and Deploy Generative AI Models 

NVIDIA NeMo is a cloud-native framework empowering researchers and developers to build, customize, and deploy state-of-the-art generative AI models across various domains. Access the original repository [here](https://github.com/NVIDIA/NeMo).

[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Key Features:

*   **Modular and Flexible:**  Utilize PyTorch Lightning's modular design for easy customization and experimentation.
*   **Scalable Training:** Seamlessly train models across thousands of GPUs with NeMo-Run and other techniques like TP, PP, FSDP, MoE, and mixed precision.
*   **Wide Domain Support:** Includes pre-built models and tools for:
    *   Large Language Models (LLMs)
    *   Multimodal Models (MMs)
    *   Automatic Speech Recognition (ASR)
    *   Text-to-Speech (TTS)
    *   Computer Vision (CV)
*   **Advanced Training Techniques:** Supports cutting-edge techniques like FP8 training with NVIDIA Transformer Engine and integration with NVIDIA Megatron Core for scaling Transformer model training.
*   **Alignment and Fine-tuning:**  Implement state-of-the-art alignment methods such as SteerLM, DPO, RLHF, and parameter-efficient fine-tuning techniques (PEFT) like LoRA, P-Tuning, Adapters, and IA3.
*   **Deployment and Optimization:** Deploy and optimize LLMs and MMs with NVIDIA NeMo Microservices and ASR/TTS models with NVIDIA Riva.
*   **Cosmos Support:** Training and customization for the NVIDIA Cosmos collection of world foundation models, including video processing with NeMo Curator.
*   **Easy Access:** Access pretrained models on Hugging Face Hub and NVIDIA NGC.

## What's New

*   **Hugging Face integration:** NeMo Framework's latest feature AutoModel enables broad support for Hugging Face models,
*   **Blackwell support:** NeMo Framework has added Blackwell support.
*   **Performance guide:** NeMo Framework has published a comprehensive guide for performance tuning to achieve optimal throughput.

## Getting Started

*   **Documentation:** Explore the comprehensive [NeMo Framework User Guide](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/).
*   **Quickstart:** Get started with NeMo 2.0 using the [Quickstart](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/quickstart.html).
*   **Examples and Recipes:** Explore training examples and recipes at [NeMo 2.0 Recipes](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/recipes) and [example scripts](https://github.com/NVIDIA/NeMo/tree/main/examples)
*   **Tutorials:** Run tutorials on [Google Colab](https://colab.research.google.com) or with the [NGC NeMo Framework Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo).

## Installation

Choose your installation method based on your needs:

*   **Conda / Pip:** Recommended for ASR and TTS domains.  Explore NeMo on any supported platform.
*   **NGC PyTorch container:** Install from source in a highly optimized container.
*   **NGC NeMo container:**  Ready-to-go solution for highest performance.

Detailed installation instructions and support matrices are available in the original README.

## Contribute

Contribute to the NeMo project by following the guidelines in [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md).

## Explore Further

*   [Publications](https://nvidia.github.io/NeMo/publications/)
*   [Discussions Board](https://github.com/NVIDIA/NeMo/discussions)
*   [Blogs](https://developer.nvidia.com/blog/nvidia-nemo-framework-features-and-nvidia-h200-supercharge-llm-training-performance-and-versatility/)

## License

NeMo is licensed under the [Apache License 2.0](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file).