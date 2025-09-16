# NVIDIA NeMo: Build, Customize, and Deploy Generative AI Models 

**NVIDIA NeMo is a versatile framework empowering researchers and developers to efficiently create, train, and deploy state-of-the-art generative AI models for various domains.** ([Original Repository](https://github.com/NVIDIA/NeMo))

[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Key Features

*   **LLMs & MMs Support:** Train and customize large language models and multimodal models with state-of-the-art techniques, including support for Hugging Face models.
*   **Speech AI:** Build and deploy high-performance Automatic Speech Recognition (ASR) and Text-to-Speech (TTS) models.
*   **Cloud-Native & Scalable:** Designed for efficient training across multiple GPUs and in cloud environments, including support for Amazon EKS and GKE.
*   **Modular Design:** NeMo 2.0 offers a Python-based configuration and modular abstractions for easier customization.
*   **Performance Optimization:** Leverage NVIDIA Transformer Engine, Megatron Core, and TensorRT-LLM for optimized training and inference.
*   **Wide Range of Techniques:** Supports cutting-edge techniques like FSDP, MoE, and PEFT methods (LoRA, etc.) to fine-tune and deploy models.
*   **Pre-trained Models:** Access a wide range of pre-trained models on Hugging Face Hub and NVIDIA NGC, enabling quick starts.
*   **Deployment Options:** Utilize NVIDIA Riva for speech AI deployment and NeMo Microservices for LLM/MM deployment.
*   **Community Support:** Engage with the community via the Discussions board and access a rich set of tutorials and examples.

## What's New

*   **Hugging Face Integration:**  Day-0 support for Hugging Face models (e.g., `AutoModelForCausalLM` and `AutoModelForImageTextToText`)
*   **Blackwell Support:**  Added support for Blackwell with performance benchmarks on GB200 & B200.
*   **Performance Tuning Guide:**  Comprehensive guide for performance tuning to achieve optimal throughput is available.
*   **New Model Support:**  Added support for latest community models - Llama 4, Flux, Llama Nemotron, Hyena & Evo2, Qwen2-VL, Qwen2.5, Gemma3, Qwen3-30B&32B.
*   **NeMo 2.0:** Prioritizes modularity and ease-of-use.

## Getting Started

*   **Documentation:**  [NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html)
*   **Tutorials:**  Extensive tutorials can be run on [Google Colab](https://colab.research.google.com) or with the [NGC NeMo Framework Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo).
*   **Examples:**  Example scripts support multi-GPU/multi-node training:  [Example Scripts](https://github.com/NVIDIA/NeMo/tree/main/examples)
*   **NGC & Hugging Face:** State-of-the-art pretrained NeMo models are freely available on [Hugging Face Hub](https://huggingface.co/models?library=nemo&sort=downloads&search=nvidia) and [NVIDIA NGC](https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC).
*   **Cosmos Foundation Models:** Support video curation and post-training of the Cosmos World Foundation Models, which are open and available on [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/cosmos/collections/cosmos) and [Hugging Face](https://huggingface.co/collections/nvidia/cosmos-6751e884dc10e013a0a0d8e6).

## Installation

Choose your installation method:

*   **Conda / Pip:**  Recommended for ASR and TTS, and for exploring NeMo on any supported platform.
*   **NGC PyTorch Container:** Install from source for feature-completeness in a highly optimized container.
*   **NGC NeMo Container:**  Ready-to-go solution for maximum performance.

See the original repository for more detailed installation instructions.

## Requirements

*   Python 3.10 or above
*   Pytorch 2.5 or above
*   NVIDIA GPU (for model training)

## Contribute

Contributions are welcome!  See [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md) for guidelines.

## Publications

See the [publications](https://nvidia.github.io/NeMo/publications/) that utilize the NeMo Framework.

## License

NVIDIA NeMo is licensed under the [Apache License 2.0](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file).