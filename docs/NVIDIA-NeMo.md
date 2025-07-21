[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# NVIDIA NeMo: Build and Deploy State-of-the-Art Generative AI Models

NVIDIA NeMo is a flexible, cloud-native framework enabling researchers and developers to build, customize, and deploy cutting-edge generative AI models for LLMs, MMs, ASR, TTS, and CV domains.

**[Go to the original repository](https://github.com/NVIDIA/NeMo) to contribute and view all the updates.**

## Key Features

*   **Large Language Models (LLMs):** Train and customize powerful language models.
*   **Multimodal Models (MMs):** Develop models that combine different data modalities.
*   **Automatic Speech Recognition (ASR):** Build and deploy high-accuracy speech recognition systems.
*   **Text-to-Speech (TTS):** Create realistic and engaging speech synthesis.
*   **Computer Vision (CV):** Explore a wide range of computer vision tasks.
*   **Modular Design:**  Built on PyTorch Lightning for ease of use and customization.
*   **Scalability:** Designed for training large-scale models across thousands of GPUs with NeMo-Run.
*   **Pre-trained Models & Recipes:**  Get started quickly with pre-trained models and example scripts.
*   **Optimizations:** Leverages NVIDIA Transformer Engine and Megatron Core for performance.
*   **Deployment & Optimization:**  Integrates with NVIDIA Riva and NeMo Microservices for production use.

## Latest Updates

*   **Support for Hugging Face Models:** Pretrain and finetune Hugging Face models using AutoModel.
*   **Blackwell Support:** Added Blackwell support with performance benchmarks on GB200 & B200.
*   **Training Performance Guide:** A comprehensive guide for performance tuning to achieve optimal throughput.
*   **New Models Support:** Support for community models like Llama 4, Flux, Llama Nemotron, Hyena & Evo2, Qwen2-VL, Qwen2.5, Gemma3, Qwen3-30B&32B.
*   **Cosmos World Foundation Models:** Training and customizing the NVIDIA Cosmos collection of world foundation models.
*   **NeMo 2.0 Release:** Prioritizes modularity and ease-of-use, with Python-based configuration and modular abstractions.

## Getting Started

### Installation

Choose your preferred installation method based on your needs:

*   **Conda / Pip:** Install NeMo via pip into a virtual environment. Recommended for ASR and TTS domains.
    *   `pip install "nemo_toolkit[all]"` (or a specific version)
*   **NGC PyTorch Container:** For users that want to install from source in a highly optimized container.
    *   Refer to the documentation on how to build the container.
*   **NGC NeMo Container:**  Ready-to-go solution with all dependencies pre-installed and optimized.
    *   `docker run ... nvcr.io/nvidia/pytorch:${NV_PYTORCH_TAG:-'nvcr.io/nvidia/nemo:25.02'}` (See full command in the original README)

### Key Resources

*   **User Guide:**  [NeMo Framework User Guide](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
*   **Quickstart:** [Quickstart](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/quickstart.html) for NeMo 2.0
*   **Examples:** [Example Scripts](https://github.com/NVIDIA/NeMo/tree/main/examples)
*   **Pre-trained Models:** [Hugging Face Hub](https://huggingface.co/models?library=nemo&sort=downloads&search=nvidia) and [NVIDIA NGC](https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC)

### Requirements

*   Python 3.10 or above
*   PyTorch 2.5 or above
*   NVIDIA GPU (for model training)

## Community and Support

*   **Discussions Board:** [NeMo Discussions board](https://github.com/NVIDIA/NeMo/discussions) - Ask questions and engage with the community.
*   **Contribute:** [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md) - Learn how to contribute to the project.
*   **Publications:**  [Publications](https://nvidia.github.io/NeMo/publications/) - Explore articles that utilize NeMo.
*   **Blogs:**  Stay updated with the latest developments through the provided blog links.

## Licenses

NeMo is licensed under the [Apache License 2.0](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file).