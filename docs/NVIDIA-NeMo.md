[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# NVIDIA NeMo: Build and Deploy Generative AI Models at Scale

NVIDIA NeMo is a powerful, cloud-native framework for researchers and developers to efficiently create, customize, and deploy state-of-the-art generative AI models.  For more information and detailed examples, see the original [NVIDIA NeMo repository](https://github.com/NVIDIA/NeMo).

**Key Features:**

*   **Large Language Models (LLMs):** Train, fine-tune, and deploy LLMs with cutting-edge techniques.
*   **Multimodal Models (MMs):** Develop models that combine text, images, and other modalities.
*   **Automatic Speech Recognition (ASR):** Build and optimize accurate speech recognition models.
*   **Text-to-Speech (TTS):** Create high-quality and natural-sounding speech synthesis systems.
*   **Computer Vision (CV):** Implement and experiment with advanced computer vision models.
*   **Scalable Training:** Leverage distributed training techniques for massive model sizes.
*   **Modular Design:**  Simplify model building and experimentation with PyTorch Lightning's modularity.
*   **Optimized Deployment:** Deploy and optimize NeMo models with NVIDIA Riva and NeMo Microservices.

## Core Capabilities

NeMo offers a comprehensive toolkit for building and deploying AI models across various domains.

*   **LLMs and MMs Training, Alignment, and Customization**: NeMo supports training, fine-tuning, and alignment of LLMs with state-of-the-art methods. Leverage techniques like SteerLM, DPO, and RLHF, and PEFT methods such as LoRA, P-Tuning, and Adapters. Training is automatically scalable to thousands of GPUs.
*   **LLMs and MMs Deployment and Optimization**: Deploy and optimize NeMo LLMs and MMs with NVIDIA NeMo Microservices.
*   **Speech AI**: NeMo ASR and TTS models can be optimized for inference and deployed for production use cases with NVIDIA Riva.

## Latest Updates

*   **[Pretrain and Finetune Hugging Face models](https://developer.nvidia.com/blog/run-hugging-face-models-instantly-with-day-0-support-from-nvidia-nemo-framework)**  (May 19, 2025):  NeMo now supports pretraining and finetuning of Hugging Face models, including AutoModelForCausalLM and AutoModelForImageTextToText.
*   **[Blackwell Support](https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance_summary.html)** (May 19, 2025):  NeMo Framework has added Blackwell support, with performance benchmarks on GB200 & B200.
*   **[Training Performance on GPU Tuning Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance-guide.html)** (May 19, 2025):  A comprehensive guide for performance tuning to achieve optimal throughput is available.
*   **[New Models Support](https://docs.nvidia.com/nemo-framework/user-guide/latest/vlms/llama4.html)** (May 19, 2025):  Support added for latest community models like Llama 4, Flux, and more.
*   **[NeMo 2.0](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/index.html)**:  Released with a focus on modularity and ease-of-use.

## Get Started

*   **Documentation:** [NeMo Framework User Guide](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
*   **Quickstart:**  [NeMo 2.0 Quickstart](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/quickstart.html)
*   **Pre-trained Models:** Available on [Hugging Face Hub](https://huggingface.co/models?library=nemo&sort=downloads&search=nvidia) and [NVIDIA NGC](https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC).
*   **Examples:** Explore the [example scripts](https://github.com/NVIDIA/NeMo/tree/main/examples) for advanced training.
*   **Training guide**: [NeMo Framework Playbooks](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html)

## Installation

Choose the installation method that best suits your needs:

*   **Conda / Pip:**  For general use and exploring NeMo, especially for ASR and TTS.
    *   `pip install "nemo_toolkit[all]"`  (for latest release)
    *   Or install from a specific Git reference:
        ```bash
        git clone https://github.com/NVIDIA/NeMo
        cd NeMo
        git checkout @${REF:-'main'}
        pip install '.[all]'
        ```
*   **NGC PyTorch container (Recommended):** For feature-completeness and optimized performance.
    *   Follow the instructions to install using a base NVIDIA PyTorch container (nvcr.io/nvidia/pytorch:25.01-py3).
*   **NGC NeMo container:** For the highest performance, a ready-to-go solution.
    *   Run the provided docker run command (nvcr.io/nvidia/pytorch:${NV_PYTORCH_TAG:-'nvcr.io/nvidia/nemo:25.02'}).

## Requirements

*   Python 3.10 or above
*   PyTorch 2.5 or above
*   NVIDIA GPU (for training)

## Contribution & Community

*   **Contribute:** See [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md).
*   **Discussions:**  [NeMo Discussions board](https://github.com/NVIDIA/NeMo/discussions)
*   **Publications:** Explore a list of [publications](https://nvidia.github.io/NeMo/publications/) using NeMo.

## Licensing

NVIDIA NeMo is licensed under the [Apache License 2.0](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file).