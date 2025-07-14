# NVIDIA NeMo Framework: Build, Customize, and Deploy State-of-the-Art Generative AI Models

[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**NVIDIA NeMo is a powerful, cloud-native framework designed to accelerate the development and deployment of Large Language Models (LLMs), Multimodal Models (MMs), and other generative AI models.**  Explore the original repository: [NVIDIA/NeMo](https://github.com/NVIDIA/NeMo).

## Key Features:

*   **Comprehensive Support:** Build, customize, and deploy cutting-edge generative AI models.
*   **Scalable Training:** Train models efficiently across thousands of GPUs with built-in parallelism strategies.
*   **Modular Design:** NeMo 2.0 offers Python-based configuration and modular abstractions for flexibility and experimentation.
*   **Wide Range of Applications:** Supports LLMs, MMs, Automatic Speech Recognition (ASR), Text-to-Speech (TTS), and Computer Vision (CV).
*   **Pre-trained Models & Tutorials:** Leverage pre-trained models from Hugging Face Hub and NVIDIA NGC, and get started quickly with extensive tutorials.
*   **Optimized Deployment:** Deploy and optimize models with NVIDIA Riva and NeMo Microservices.
*   **Performance Benchmarks:** Access performance benchmarks and tuning guides to achieve optimal throughput.

## Latest Updates:

*   **Hugging Face Integration:**  Seamlessly pretrain and fine-tune Hugging Face models with AutoModel support.
*   **Blackwell Support:** Optimized training performance on NVIDIA Blackwell (GB200 & B200) GPUs.
*   **Performance Tuning Guide:** Comprehensive guide available for optimizing training throughput.
*   **New Model Support:** Expanding support for the latest community models, including Llama 4, Flux, Hyena, Qwen2, and more.
*   **Cosmos Integration:** Support for training and customizing NVIDIA Cosmos world foundation models, enhancing video generation capabilities.
*   **NeMo 2.0 Release:**  Focuses on modularity and ease-of-use.

## Getting Started:

*   **Documentation:**  Access the latest documentation at [https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/).
*   **User Guide:**  Find the user guide at [https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html).
*   **Pre-trained Models:** Explore pre-trained models on [Hugging Face Hub](https://huggingface.co/models?library=nemo&sort=downloads&search=nvidia) and [NVIDIA NGC](https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC).

## Installation:

Choose the installation method that best fits your needs:

*   **Conda / Pip:**  Recommended for ASR and TTS and for exploring NeMo (limited feature completeness for other domains).
*   **NGC PyTorch Container:**  Install from source within a highly optimized container.
*   **NGC NeMo Container:** Ready-to-go solution for maximum performance.

## Requirements:

*   Python 3.10 or above
*   Pytorch 2.5 or above
*   NVIDIA GPU (for model training)

## Contribution and Community:

*   **Contribute:**  Join the community and contribute to NeMo. Refer to [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md) for guidelines.
*   **Discussions:**  Ask questions and join discussions on the [Discussions board](https://github.com/NVIDIA/NeMo/discussions).
*   **Publications:** Explore a growing list of publications utilizing the NeMo Framework.

## Licenses

*   NeMo is licensed under the [Apache License 2.0](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file).