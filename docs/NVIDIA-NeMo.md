[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# NVIDIA NeMo: Unleash the Power of Generative AI (LLMs, MMs, ASR, TTS, CV)

NVIDIA NeMo is a cloud-native, end-to-end framework that simplifies the development, customization, and deployment of state-of-the-art generative AI models, including Large Language Models (LLMs), Multimodal Models (MMs), Automatic Speech Recognition (ASR), Text-to-Speech (TTS), and Computer Vision (CV) models. Explore the original repo [here](https://github.com/NVIDIA/NeMo).

## Key Features:

*   **Comprehensive Support:**  Develop and deploy AI models across LLMs, MMs, ASR, TTS, and CV domains.
*   **Cloud-Native Design:** Optimized for cloud environments, ensuring scalability and ease of use.
*   **Modular Architecture:**  Built on PyTorch Lightning, NeMo promotes modularity for easier customization and experimentation.
*   **Pre-trained Models:** Leverage a vast library of pre-trained models available on Hugging Face Hub and NVIDIA NGC, accelerating your development.
*   **Scalable Training:** Supports training on thousands of GPUs with cutting-edge techniques like Tensor Parallelism, Pipeline Parallelism, FSDP, MoE, and mixed precision training.
*   **Advanced Optimization:**  Integrates with NVIDIA Transformer Engine and Megatron Core for efficient FP8 and large-scale Transformer model training.
*   **Deployment Ready:**  Deploy and optimize LLMs and MMs with NVIDIA NeMo Microservices and ASR/TTS models with NVIDIA Riva.
*   **PEFT & Alignment Support:** Includes support for Parameter Efficient Fine-Tuning (PEFT) techniques like LoRA and methods such as DPO, and RLHF.

## What's New

*   **Pretrain and finetune :hugs:Hugging Face models via AutoModel:**  NeMo Framework's latest feature AutoModel enables broad support for :hugs:Hugging Face models, with 25.04 focusing on `AutoModelForCausalLM` and  `AutoModelForImageTextToText` models.
*   **Training on Blackwell using Nemo:**  NeMo Framework has added Blackwell support, with performance benchmarks on GB200 & B200. 
*   **Training Performance on GPU Tuning Guide:** NeMo Framework has published a comprehensive guide for performance tuning to achieve optimal throughput.
*   **New Models Support:**  NeMo Framework has added support for latest community models - Llama 4, Flux, Llama Nemotron, Hyena & Evo2, Qwen2-VL, Qwen2.5, Gemma3, Qwen3-30B&32B.
*   **NeMo Framework 2.0:**  Major update emphasizing modularity and ease-of-use.  
*   **Cosmos World Foundation Models Support:** Expanded support for the NVIDIA Cosmos platform.

## Getting Started

*   **Documentation:** Access comprehensive documentation and user guides at [NeMo Framework User Guide](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/).
*   **Quickstart:** Get started quickly with [Quickstart](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/quickstart.html).
*   **Tutorials:**  Run tutorials on Google Colab or with the NGC NeMo Framework Container.
*   **Pre-trained Models:**  Explore available models on [Hugging Face Hub](https://huggingface.co/models?library=nemo&sort=downloads&search=nvidia) and [NVIDIA NGC](https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC).

## Installation

Choose an installation method based on your needs:

*   **Conda / Pip:** Recommended for ASR and TTS and general exploration, install using `pip install "nemo_toolkit[all]"`.
*   **NGC PyTorch Container:** For source installation within a highly optimized container (NeMo-Toolkit 2.3.0+).
*   **NGC NeMo Container:** Pre-built, optimized container for maximum performance.

## Requirements:

*   Python 3.10+
*   PyTorch 2.5+
*   NVIDIA GPU (for training)

## Resources:

*   **Developer Documentation:** Access the latest documentation at [Documentation of the latest (i.e. main) branch.](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/).
*   **Discussions Board:**  Find answers and engage with the community on the [NeMo Discussions board](https://github.com/NVIDIA/NeMo/discussions).
*   **Publications:**  Explore a list of [publications](https://nvidia.github.io/NeMo/publications/) that use the NeMo Framework.
*   **Contribute:**  Contribute to NeMo! See [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md) for details.