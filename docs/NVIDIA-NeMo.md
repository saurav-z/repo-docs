[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# NVIDIA NeMo: Build, Customize, and Deploy Generative AI Models at Scale

NVIDIA NeMo is a powerful, cloud-native framework empowering researchers and developers to create, customize, and deploy state-of-the-art generative AI models for various applications.  Access the original repository [here](https://github.com/NVIDIA/NeMo).

## Key Features

*   **Large Language Models (LLMs):** Efficiently train and fine-tune large language models.
*   **Multimodal Models (MMs):** Develop models that process and generate data across different modalities, like text and images.
*   **Automatic Speech Recognition (ASR):** Create and optimize models for accurate speech-to-text transcription.
*   **Text-to-Speech (TTS):** Generate natural-sounding speech from text.
*   **Computer Vision (CV):** Build models for image recognition, object detection, and more.
*   **Modular Architecture:** Built on PyTorch Lightning, NeMo offers a modular and flexible approach to model development.
*   **Scalability:** Seamlessly scale your experiments across thousands of GPUs using NeMo-Run.
*   **Pre-trained Models:** Leverage a wide range of pre-trained models available on Hugging Face Hub and NVIDIA NGC.
*   **Parameter Efficient Fine-tuning (PEFT):** Supports techniques like LoRA and Adapters for efficient model customization.
*   **Deployment and Optimization:** Optimize and deploy your LLMs and MMs with NVIDIA NeMo Microservices and ASR/TTS models with NVIDIA Riva.

## What's New

*   **Hugging Face Model Support:** Seamlessly integrate and fine-tune Hugging Face models, including AutoModelForCausalLM and AutoModelForImageTextToText.
*   **Blackwell Support:** Optimized performance on NVIDIA Blackwell architecture.
*   **Performance Tuning Guide:** Comprehensive guide for optimizing throughput.
*   **New Model Support:** Integration of the latest community models such as Llama 4, Flux, Qwen2-VL, and more.
*   **NeMo 2.0:** Introduces a Python-based configuration, modular abstractions, and scalability enhancements.
*   **Cosmos World Foundation Models:** Training and customization support for NVIDIA Cosmos models.

## Getting Started

### Installation

Choose your installation method:

*   **Conda/Pip:** For general use and ASR/TTS domains.

    ```bash
    conda create --name nemo python==3.10.12
    conda activate nemo
    pip install "nemo_toolkit[all]"  # Or install from a specific Git reference
    ```

    Install domain-specific dependencies like this:
    ```bash
    pip install nemo_toolkit['asr']
    ```

*   **NGC PyTorch Container:** Optimized container for source installations.
    ```bash
    docker run --gpus all -it --rm --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/pytorch:${NV_PYTORCH_TAG:-'nvcr.io/nvidia/pytorch:25.01-py3'}
    ```
    Then:
    ```bash
    cd /opt
    git clone https://github.com/NVIDIA/NeMo
    cd NeMo
    git checkout ${REF:-'main'}
    bash docker/common/install_dep.sh --library all
    pip install ".[all]"
    ```

*   **NGC NeMo Container:** For highest performance and ready-to-go solutions.
    ```bash
    docker run --gpus all -it --rm --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/pytorch:${NV_PYTORCH_TAG:-'nvcr.io/nvidia/nemo:25.02'}
    ```

### Tutorials and Resources

*   **Documentation:** [NeMo Framework User Guide](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
*   **Quickstart:** Get started with NeMo 2.0 [Quickstart](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/quickstart.html)
*   **Tutorials:**  [Tutorials](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/starthere/tutorials.html)
*   **Examples:** [Example Scripts](https://github.com/NVIDIA/NeMo/tree/main/examples)
*   **Pre-trained Models:** Available on [Hugging Face Hub](https://huggingface.co/models?library=nemo&sort=downloads&search=nvidia) and [NVIDIA NGC](https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC).

## Requirements

*   Python 3.10 or above
*   PyTorch 2.5 or above
*   NVIDIA GPU (for model training)

## Development

*   **Developer Documentation:** Access the latest documentation [here](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/).
*   **Contribute:**  Refer to [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md) for guidelines.

## Get Involved

*   **Discussions:**  Join the conversation on the [Discussions board](https://github.com/NVIDIA/NeMo/discussions).

## Licenses

NeMo is licensed under the [Apache License 2.0](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file).