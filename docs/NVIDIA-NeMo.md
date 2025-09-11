# NVIDIA NeMo: Build, Customize, and Deploy State-of-the-Art Generative AI Models

[NVIDIA NeMo](https://github.com/NVIDIA/NeMo) is a powerful, cloud-native framework that empowers researchers and developers to create cutting-edge generative AI models for various domains, including Large Language Models (LLMs), Multimodal Models (MMs), Automatic Speech Recognition (ASR), Text-to-Speech (TTS), and Computer Vision (CV).

[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Key Features

*   **LLMs & MMs:** Train, fine-tune, and align large language and multimodal models with cutting-edge techniques, including support for state-of-the-art methods like SteerLM and DPO.
*   **ASR:**  Optimize and deploy Automatic Speech Recognition models, accelerating performance with up to 10x inference speed-up.
*   **TTS:** Utilize Text-to-Speech models, optimized for inference and ready for production deployment.
*   **Computer Vision:** Access tools for computer vision tasks.
*   **Modular and Scalable:**  Leverage a modular architecture and scalable training capabilities to efficiently handle large-scale experiments.
*   **Pre-trained Models:** Get started quickly with pre-trained models available on Hugging Face Hub and NVIDIA NGC.
*   **Deployment and Optimization:** Utilize NVIDIA NeMo Microservices for deployment and optimization of LLMs and MMs, and NVIDIA Riva for ASR and TTS models.

## What's New

*   **Hugging Face Integration:** Broad support for Hugging Face models with AutoModel capabilities.
*   **Blackwell Support:** Includes performance benchmarks for GB200 & B200.
*   **New Model Support:** Including Llama 4, Flux, Llama Nemotron, Hyena & Evo2, Qwen2-VL, Qwen2.5, Gemma3, Qwen3-30B&32B.
*   **Cosmos World Foundation Models:** Support for training and customizing NVIDIA Cosmos models for video generation.
*   **NeMo 2.0:** Modular, Python-based configuration for enhanced flexibility, performance and scalability.

## Getting Started

*   Explore comprehensive [tutorials](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/starthere/tutorials.html) on Google Colab or with the NGC NeMo Framework Container.
*   Find pre-trained models on [Hugging Face Hub](https://huggingface.co/models?library=nemo&sort=downloads&search=nvidia) and [NVIDIA NGC](https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC).
*   Refer to [NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html) and [NeMo 2.0 Quickstart](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/quickstart.html).

## Installation

Choose the method that best suits your needs:

*   **Conda / Pip:** Install with `pip install "nemo_toolkit[all]"` (recommended for ASR/TTS).
*   **NGC PyTorch Container:** Install from source within a pre-built NVIDIA PyTorch container.
*   **NGC NeMo Container:** Utilize a pre-built, optimized container for maximum performance.

## Resources

*   **Documentation:** [NeMo Framework User Guide](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
*   **Tutorials:** [Tutorials](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/starthere/tutorials.html)
*   **Publications:** [Publications](https://nvidia.github.io/NeMo/publications/)
*   **Discussions:** [Discussions board](https://github.com/NVIDIA/NeMo/discussions)
*   **Contribute:** [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md)