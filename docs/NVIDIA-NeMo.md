[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# NVIDIA NeMo: Build, Customize, and Deploy State-of-the-Art Generative AI Models

NVIDIA NeMo is a powerful, cloud-native framework empowering researchers and developers to build, customize, and deploy cutting-edge generative AI models for Large Language Models (LLMs), Multimodal Models (MMs), Automatic Speech Recognition (ASR), Text-to-Speech (TTS), and Computer Vision (CV) applications. Explore the original repo [here](https://github.com/NVIDIA/NeMo).

## Key Features

*   **Large Language Models (LLMs):** Train and fine-tune LLMs efficiently.
*   **Multimodal Models (MMs):** Develop models that process multiple data types (text, images, etc.).
*   **Automatic Speech Recognition (ASR):** Build accurate speech recognition systems.
*   **Text-to-Speech (TTS):** Create high-quality text-to-speech models.
*   **Computer Vision (CV):** Implement advanced computer vision tasks.
*   **Modular and Scalable:** Built on PyTorch Lightning for ease of use and scalability.
*   **Performance Optimization:** Leverages NVIDIA Transformer Engine and Megatron Core for FP8 training and efficient scaling.
*   **Alignment Techniques:** Supports state-of-the-art alignment methods like SteerLM, DPO, and RLHF.
*   **PEFT Support:** Integrated with Parameter-Efficient Fine-tuning (PEFT) methods like LoRA.
*   **Deployment and Optimization:** Integrates with NVIDIA Riva for ASR/TTS deployment.
*   **Nemo 2.0:** A modern framework with Python-based configuration and modular abstractions.
*   **Cosmos Support:** Post-training of Cosmos World Foundation Models.

## What's New

### NeMo 2.0 Highlights:
  *   **Python-Based Configuration:** Increased flexibility and programmatic control.
  *   **Modular Abstractions:** Simplifies adaptation and experimentation.
  *   **Scalability:** Enables large-scale experiments across thousands of GPUs using NeMo-Run.
  *   **Supported Collections:** LLMs and VLMs.
  *   **Get Started:** Quickstart, User Guide, Feature Guide, Migration Guide

### Current Updates:

  *  Support for Hugging Face models, Blackwell support, and GPU tuning guide
  *  Support for latest community models - Llama 4, Flux, Llama Nemotron, Hyena & Evo2, Qwen2-VL, Qwen2.5, Gemma3, Qwen3-30B&32B.

## Getting Started

### Installation

#### Conda / Pip

```bash
conda create --name nemo python==3.10.12
conda activate nemo
pip install "nemo_toolkit[all]" # or specify domain like "nemo_toolkit['asr']"
```

#### NGC PyTorch container (Recommended)

```bash
docker run \
  --gpus all \
  -it \
  --rm \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  nvcr.io/nvidia/pytorch:${NV_PYTORCH_TAG:-'nvcr.io/nvidia/pytorch:25.01-py3'}
```

#### NGC NeMo container

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

### Resources

*   **Documentation:** [NeMo Framework User Guide](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
*   **Tutorials:** [Tutorials](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/starthere/tutorials.html)
*   **Pre-trained Models:** [Hugging Face Hub](https://huggingface.co/models?library=nemo&sort=downloads&search=nvidia) and [NVIDIA NGC](https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC)
*   **Discussions:** [Discussions board](https://github.com/NVIDIA/NeMo/discussions)
*   **Publications:** [Publications](https://nvidia.github.io/NeMo/publications/)

## Requirements

*   Python 3.10 or above
*   PyTorch 2.5 or above
*   NVIDIA GPU (for training)

## Contribution & Community

Contributions are welcome! See [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md) for guidelines. Ask questions and start discussions on the NeMo [Discussions board](https://github.com/NVIDIA/NeMo/discussions).