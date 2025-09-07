[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# NVIDIA NeMo: Build, Customize, and Deploy Generative AI Models at Scale

NVIDIA NeMo is a powerful, cloud-native framework empowering researchers and developers to efficiently create, customize, and deploy state-of-the-art generative AI models. [Learn more at the original repo](https://github.com/NVIDIA/NeMo).

## Key Features

*   **Large Language Models (LLMs):** Train, fine-tune, and deploy LLMs with cutting-edge techniques.
*   **Multimodal Models (MMs):** Develop models that combine different data modalities (e.g., text, images, and video).
*   **Automatic Speech Recognition (ASR):** Build and optimize ASR models for various applications.
*   **Text-to-Speech (TTS):** Create high-quality speech synthesis models.
*   **Computer Vision (CV):** Explore and implement computer vision solutions with pre-trained models and example scripts.
*   **Seamless Scalability:** Train models across thousands of GPUs using advanced distributed training techniques.
*   **Modular and Flexible:** Leverage Python-based configuration and modular abstractions.
*   **Pre-trained Models:** Access a wide range of pre-trained models on Hugging Face Hub and NVIDIA NGC.
*   **Model Optimization and Deployment:** Deploy and optimize models with NVIDIA Riva and NeMo Microservices.
*   **Comprehensive Tooling:** Utilize the NeMo Framework Launcher and NeMo-Run for streamlined training and experimentation.

## Latest Updates

*   **[New Models Support]** - Llama 4, Flux, Llama Nemotron, Hyena & Evo2, Qwen2-VL, Qwen2.5, Gemma3, Qwen3-30B&32B.
*   **[Training on Blackwell using Nemo]** - Performance benchmarks on GB200 & B200.
*   **[Pretrain and finetune Hugging Face models via AutoModel]** - AutoModelForCausalLM and AutoModelForImageTextToText support.
*   **[Training Performance on GPU Tuning Guide]** - A comprehensive guide for performance tuning to achieve optimal throughput.
*   **[NVIDIA Cosmos World Foundation Models Support]** - Training and customizing the NVIDIA Cosmos collection of world foundation models.
*   **[Large Language Models and Multimodal Models]** - New Llama 3.1 support, NVIDIA releases 340B models, and more.
*   **[Speech Recognition]** - Inference optimizations for ASR models, the NeMo Canary multilingual model, and the Parakeet ASR models.

## Getting Started

NeMo provides extensive resources to get you started:

*   **Documentation:** Explore the [NeMo Framework User Guide](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/) for detailed information.
*   **Quickstart:** Use the [Quickstart](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/quickstart.html) for examples of using NeMo-Run.
*   **Examples & Recipes:** Find practical examples and recipes to launch large-scale runs, available in the [NeMo 2.0 Recipes](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/recipes)
*   **Tutorials:** Access tutorials on [Google Colab](https://colab.research.google.com) or with our [NGC NeMo Framework Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo).
*   **Pre-trained Models:** Utilize pre-trained models from [Hugging Face Hub](https://huggingface.co/models?library=nemo&sort=downloads&search=nvidia) and [NVIDIA NGC](https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC).

## Installation

NeMo offers several installation methods:

*   **Conda / Pip:** Recommended for ASR and TTS, and limited feature-completeness for other domains.
    ```bash
    conda create --name nemo python==3.10.12
    conda activate nemo
    pip install "nemo_toolkit[all]"
    ```
*   **NGC PyTorch Container:** For installation from source in a highly optimized container.
    ```bash
    docker run ...  # Follow the instructions in the original README
    ```
*   **NGC NeMo Container:** Ready-to-go solution, recommended for highest performance.
    ```bash
    docker run ...  # Follow the instructions in the original README
    ```
    Consult the [original README](https://github.com/NVIDIA/NeMo) for detailed setup instructions for each method.

## Requirements

*   Python 3.10 or above
*   PyTorch 2.5 or above
*   NVIDIA GPU (for model training)

## Contribution

We welcome community contributions!  See [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md) for guidelines.

## Resources

*   **Developer Documentation:** [NeMo Framework User Guide](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
*   **Discussions Board:** [NeMo Discussions Board](https://github.com/NVIDIA/NeMo/discussions)
*   **Publications:** [Publications](https://nvidia.github.io/NeMo/publications/)