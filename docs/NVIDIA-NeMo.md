[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# NVIDIA NeMo: Build, Customize, and Deploy Generative AI Models at Scale

NVIDIA NeMo is a powerful, cloud-native framework designed to accelerate the development of Large Language Models (LLMs), Multimodal Models (MMs), and more.  **[Explore the NeMo Framework on GitHub](https://github.com/NVIDIA/NeMo)!**

## Key Features

*   **LLMs and MMs:** Train, align, and customize large language and multimodal models.
*   **Automatic Speech Recognition (ASR):** Build and optimize ASR models.
*   **Text-to-Speech (TTS):**  Create and deploy high-quality TTS models.
*   **Computer Vision (CV):** Support for vision models.
*   **Modular & Scalable:** Built on PyTorch Lightning with built-in support for scaling experiments across thousands of GPUs using NeMo-Run.
*   **Integration with Hugging Face:** Easily pretrain and fine-tune Hugging Face models via AutoModel.
*   **Deployment & Optimization:** Deploy and optimize models with NVIDIA NeMo Microservices and Riva.
*   **Model Alignment:** Fine-tune with cutting-edge methods, including SteerLM, Direct Preference Optimization (DPO), and Reinforcement Learning from Human Feedback (RLHF).
*   **Parameter-Efficient Fine-tuning (PEFT):** Supports methods such as LoRA, P-Tuning, Adapters, and IA3.
*   **Cosmos World Foundation Models:** Supports training and customizing NVIDIA Cosmos models.

## What's New

*   **Support for Blackwell:** Updated performance benchmarks on GB200 & B200.
*   **New Models:** Support for latest community models, including Llama 4, Flux, Llama Nemotron, Hyena & Evo2, Qwen2-VL, Qwen2.5, Gemma3, Qwen3-30B&32B.
*   **Hugging Face Integration:**  Enhanced support for pretraining and fine-tuning Hugging Face models, including AutoModelForCausalLM and AutoModelForImageTextToText.
*   **NeMo 2.0:**  Prioritizes modularity, ease-of-use and provides Python-based configuration.

## Getting Started

*   **[Quickstart](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/quickstart.html):** Example using NeMo-Run.
*   **[NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/index.html):**  Complete documentation.
*   **[NeMo 2.0 Recipes](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/recipes):** Examples of launching large-scale runs.
*   **[Feature Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/features/index.html#feature-guide):**  Main features of NeMo 2.0.
*   **[Migration Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/migration/index.html#migration-guide):** Transition from NeMo 1.0 to 2.0.
*   **Pre-trained Models:** Download state-of-the-art pretrained NeMo models from [Hugging Face Hub](https://huggingface.co/models?library=nemo&sort=downloads&search=nvidia) and [NVIDIA NGC](https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC).

## Installation

Choose the installation method that best fits your needs:

*   **Conda / Pip:**  Ideal for exploring NeMo and for ASR and TTS development, using native Pip into a virtual environment.
*   **NGC PyTorch Container:** Install from source within a highly optimized container.
*   **NGC NeMo Container:**  Ready-to-use solution for optimal performance.

### Requirements

*   Python 3.10 or above
*   Pytorch 2.5 or above
*   NVIDIA GPU (for model training)

### Installation Steps

1.  **Conda / Pip:**
    ```bash
    conda create --name nemo python==3.10.12
    conda activate nemo
    pip install "nemo_toolkit[all]"
    ```
    or if you prefer a specific commit:
    ```bash
    git clone https://github.com/NVIDIA/NeMo
    cd NeMo
    git checkout @${REF:-'main'}
    pip install '.[all]'
    ```
    or a domain specific install
    ```bash
    pip install nemo_toolkit['asr']
    pip install nemo_toolkit['nlp']
    pip install nemo_toolkit['tts']
    pip install nemo_toolkit['vision']
    pip install nemo_toolkit['multimodal']
    ```

2.  **NGC PyTorch Container:**
    ```bash
    docker run \
      --gpus all \
      -it \
      --rm \
      --shm-size=16g \
      --ulimit memlock=-1 \
      --ulimit stack=67108864 \
      nvcr.io/nvidia/pytorch:${NV_PYTORCH_TAG:-'nvcr.io/nvidia/pytorch:25.01-py3'}
    cd /opt
    git clone https://github.com/NVIDIA/NeMo
    cd NeMo
    git checkout ${REF:-'main'}
    bash docker/common/install_dep.sh --library all
    pip install ".[all]"
    ```

3.  **NGC NeMo Container:**
    ```bash
    docker run \
      --gpus all \
      -it \
      --rm \
      --shm-size=16g \
      --ulimit memlock=-1 \
      --ulimit stack=67108864 \
      nvcr.io/nvidia/pytorch:${NV_PYTORCH_TAG:-'nvcr.io/nvidia/nemo:25.02'}
    ```

## Developer Documentation

[Latest Documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)

## Contribute

We welcome community contributions!  See [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md) for details.

## Discussions

Visit the [Discussions board](https://github.com/NVIDIA/NeMo/discussions) to ask questions and join discussions.

## Publications

Explore publications utilizing NeMo:  [Publications](https://nvidia.github.io/NeMo/publications/)

## License

NVIDIA NeMo is licensed under the [Apache License 2.0](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file).