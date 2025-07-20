[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# NVIDIA NeMo: Build, Customize, and Deploy Generative AI Models with Ease

NVIDIA NeMo is a versatile and scalable framework, accelerating the development of Large Language Models (LLMs), Multimodal Models (MMs), and more, offering researchers and developers a streamlined path from concept to deployment. [Explore the original repository](https://github.com/NVIDIA/NeMo) for the latest updates and contributions.

## Key Features

*   **LLMs & MMs:** Powerful tools for training, alignment, and customization of large language and multimodal models.
*   **Automatic Speech Recognition (ASR):** State-of-the-art models for speech-to-text tasks.
*   **Text-to-Speech (TTS):** High-quality speech synthesis capabilities.
*   **Computer Vision (CV):** Support for various computer vision tasks.
*   **Modular Design:** Python-based configuration and PyTorch Lightning integration for flexibility.
*   **Scalability:** Efficiently scales training across thousands of GPUs using NeMo-Run and advanced parallelism techniques.
*   **Deployment & Optimization:** Integration with NVIDIA Riva for optimized inference and production deployment.
*   **Pre-trained Models:** Access to a wide range of pre-trained models on Hugging Face Hub and NVIDIA NGC.

## What's New

*   **[Day-0 Support for Hugging Face Models with AutoModel](https://developer.nvidia.com/blog/run-hugging-face-models-instantly-with-day-0-support-from-nvidia-nemo-framework):** Enables broad support for Hugging Face models, including text generation and image-to-text models.
*   **[Blackwell Support with Performance Benchmarks](https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance_summary.html):** Added support for Blackwell, with performance benchmarks on GB200 & B200, with ongoing optimizations in future releases.
*   **[Training Performance on GPU Tuning Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance-guide.html):** A comprehensive guide for performance tuning to achieve optimal throughput.
*   **[New Models Support](https://docs.nvidia.com/nemo-framework/user-guide/latest/vlms/llama4.html):** Support for latest community models - Llama 4, Flux, Llama Nemotron, Hyena & Evo2, Qwen2-VL, Qwen2.5, Gemma3, Qwen3-30B&32B.
*   **[NVIDIA Cosmos World Foundation Models Support](https://developer.nvidia.com/blog/advancing-physical-ai-with-nvidia-cosmos-world-foundation-model-platform):** Accelerates world model development for physical AI systems with state-of-the-art world foundation models.
*   **[NeMo 2.0](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/index.html):**  Enhances flexibility, performance, and scalability, with a transition to Python-based configurations and modular abstractions.
*   **[Large Language Models and Multimodal Models](https://developer.nvidia.com/blog/state-of-the-art-multimodal-generative-ai-model-development-with-nvidia-nemo/):** Enhancements to the NeMo platform, focusing on multimodal generative AI models, including NeMo Curator and the Cosmos tokenizer.
*   **[Speech Recognition](https://developer.nvidia.com/blog/turbocharge-asr-accuracy-and-speed-with-nvidia-nemo-parakeet-tdt/):** NeMo ASR models now have inference optimizations.
*   **[NeMo-Run](https://github.com/NVIDIA/NeMo-Run):**  Recommended for launching experiments using NeMo 2.0.

## Getting Started

*   **Installation:**
    *   [Conda / Pip](#conda--pip): Install NeMo with native Pip into a virtual environment.
    *   [NGC PyTorch container](#ngc-pytorch-container): Install NeMo-Framework from source with feature-completeness into a highly optimized container.
    *   [NGC NeMo container](#ngc-nemo-container): Ready-to-go solution of NeMo-Framework
*   **Tutorials:** Access extensive tutorials on [Google Colab](https://colab.research.google.com) and with our [NGC NeMo Framework Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo).
*   **Examples:** Find example scripts supporting multi-GPU/multi-node training in the [examples directory](https://github.com/NVIDIA/NeMo/tree/main/examples).

## Requirements

*   Python 3.10 or above
*   PyTorch 2.5 or above
*   NVIDIA GPU (for model training)

## Developer Documentation

[See the documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/) for detailed information.

## Contribution & Community

*   **Contribute:** We welcome community contributions; see [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md) for details.
*   **Discussions:** Ask questions and start discussions on the [Discussions board](https://github.com/NVIDIA/NeMo/discussions).

## Publications & Blogs

Explore the [Publications](https://nvidia.github.io/NeMo/publications/) and [Blogs](https://blogs.nvidia.com/blog/) to see how NeMo is used in cutting-edge research and industry applications.

## Licenses

NeMo is licensed under the [Apache License 2.0](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file).