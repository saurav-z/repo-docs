[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# NVIDIA NeMo: Build and Deploy Generative AI Models at Scale

NVIDIA NeMo is a cloud-native framework empowering researchers and developers to create, customize, and deploy state-of-the-art generative AI models across various domains. [See the original repo](https://github.com/NVIDIA/NeMo) for more details.

## Key Features:

*   **Large Language Models (LLMs):** Train, fine-tune, and deploy LLMs with advanced techniques like Transformer Engine and Megatron Core.
*   **Multimodal Models (MMs):** Develop models that combine different data modalities like text, images, and video.
*   **Automatic Speech Recognition (ASR):** Build and optimize ASR models with cutting-edge accuracy and speed.
*   **Text-to-Speech (TTS):** Create high-quality speech synthesis models.
*   **Computer Vision (CV):** Explore and implement computer vision models.
*   **Nemo 2.0**: Upgraded framework with Python-Based Configuration, Modular Abstractions, and Scalability.
*   **Cosmos Support**: Support for training and customizing NVIDIA Cosmos models

## What's New

*   **Support for latest community models**
*   **New support for Hugging Face Models**
*   **Support for Blackwell and Performance Benchmarks**

## Getting Started

###  Installation
Choose the most suitable installation method:
*   **Conda / Pip:** Install NeMo-Framework into a virtual environment for exploring or use ASR and TTS.
*   **NGC PyTorch Container:** Install from source into an optimized container.
*   **NGC NeMo Container:** Use a pre-built container for optimal performance.

See the [Installation](#install-nemo-framework) section for detailed instructions and platform support.

###  Quick Start

1.  **Explore Pre-trained Models:** Access pre-trained models on [Hugging Face Hub](https://huggingface.co/models?library=nemo&sort=downloads&search=nvidia) and [NVIDIA NGC](https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC).
2.  **Tutorials:** Run tutorials on [Google Colab](https://colab.research.google.com) or using the [NGC NeMo Framework Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo).
3.  **Playbooks and Examples:** Utilize [playbooks](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html) and example scripts for training models from scratch or fine-tuning.

## Developer Resources

*   **Documentation:** Comprehensive documentation is available [here](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/).
*   **Example Scripts:** Find example scripts for multi-GPU/multi-node training [here](https://github.com/NVIDIA/NeMo/tree/main/examples).
*   **Publications:** Explore a growing list of publications utilizing the NeMo Framework: [publications](https://nvidia.github.io/NeMo/publications/)
*   **Blogs:** Read the latest blogs [here](#blogs)

## Contribute

We welcome community contributions; refer to [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md) for guidelines.

## License

NeMo is licensed under the [Apache License 2.0](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file).