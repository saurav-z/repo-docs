<!-- Improved README.md -->
[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# NVIDIA NeMo: Revolutionizing Generative AI with a Unified Framework

NVIDIA NeMo is a comprehensive framework designed to accelerate the development and deployment of Large Language Models (LLMs), Multimodal Models (MMs), and other generative AI models. **For more details, visit the original repository: [NVIDIA/NeMo](https://github.com/NVIDIA/NeMo).**

## Key Features

*   **Modular Architecture:** Build and customize generative AI models efficiently with PyTorch Lightning's modular abstractions.
*   **Scalable Training:** Train models across thousands of GPUs with optimized distributed training techniques, including Tensor Parallelism, Pipeline Parallelism, and FSDP.
*   **Pre-trained Models:** Leverage a wide range of pre-trained models from Hugging Face Hub and NVIDIA NGC, streamlining your AI development process.
*   **Model Alignment:** Utilize state-of-the-art alignment methods such as SteerLM, Direct Preference Optimization (DPO), and Reinforcement Learning from Human Feedback (RLHF).
*   **Parameter-Efficient Fine-tuning:** Implement PEFT techniques like LoRA, P-Tuning, and Adapters for optimized model customization.
*   **Speech AI Capabilities:** Optimize ASR and TTS models for production with NVIDIA Riva.
*   **Flexible Deployment:** Deploy and optimize models with NVIDIA NeMo Microservices.
*   **Comprehensive Ecosystem:** Includes the NeMo Framework Launcher for cloud-native training on CSPs and Slurm clusters.

## What's New

*   **Hugging Face Integration:**  Seamlessly integrate and fine-tune Hugging Face models with AutoModel support, including AutoModelForCausalLM and AutoModelForImageTextToText.
*   **Blackwell Support:** Benefit from improved performance on Blackwell architecture, with upcoming optimizations.
*   **Performance Tuning Guide:** Access a detailed guide for performance tuning to achieve optimal throughput.
*   **New Model Support:** Utilize support for the latest community models like Llama 4, Flux, Llama Nemotron, Hyena & Evo2, Qwen2-VL, Qwen2.5, Gemma3, Qwen3-30B&32B.
*   **NeMo 2.0:** Experience enhanced modularity and ease of use with the latest release, including Python-based configurations, improved modular abstractions, and scalability via NeMo-Run.
*   **Cosmos Integration:** Leverage the Cosmos platform for world model development in physical AI systems, including support for training and customizing NVIDIA Cosmos models.

## Getting Started

1.  **Installation:** Choose your preferred method:
    *   **Conda / Pip:** Recommended for exploring NeMo, particularly for ASR and TTS domains (see detailed instructions below).
    *   **NGC PyTorch Container:** Install from source within a highly optimized container.
    *   **NGC NeMo Container:** Utilize a ready-to-go solution for optimal performance.
2.  **Pre-trained Models:** Explore and utilize pre-trained models from [Hugging Face Hub](https://huggingface.co/models?library=nemo&sort=downloads&search=nvidia) and [NVIDIA NGC](https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC).
3.  **Tutorials & Documentation:** Access extensive [tutorials](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/starthere/tutorials.html) and [playbooks](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html) to guide your learning and training process.

## Installation Options

### Conda / Pip

Install NeMo in a fresh Conda environment:

```bash
conda create --name nemo python==3.10.12
conda activate nemo
```

#### Pick the right version

```bash
pip install "nemo_toolkit[all]"
```

#### Install a specific Domain

```bash
pip install nemo_toolkit['all'] # or pip install "nemo_toolkit['all']@git+https://github.com/NVIDIA/NeMo@${REF:-'main'}"
pip install nemo_toolkit['asr'] # or pip install "nemo_toolkit['asr']@git+https://github.com/NVIDIA/NeMo@$REF:-'main'}"
pip install nemo_toolkit['nlp'] # or pip install "nemo_toolkit['nlp']@git+https://github.com/NVIDIA/NeMo@${REF:-'main'}"
pip install nemo_toolkit['tts'] # or pip install "nemo_toolkit['tts']@git+https://github.com/NVIDIA/NeMo@${REF:-'main'}"
pip install nemo_toolkit['vision'] # or pip install "nemo_toolkit['vision']@git+https://github.com/NVIDIA/NeMo@${REF:-'main'}"
pip install nemo_toolkit['multimodal'] # or pip install "nemo_toolkit['multimodal']@git+https://github.com/NVIDIA/NeMo@${REF:-'main'}"
```

### NGC PyTorch container

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

```bash
cd /opt
git clone https://github.com/NVIDIA/NeMo
cd NeMo
git checkout ${REF:-'main'}
bash docker/common/install_dep.sh --library all
pip install ".[all]"
```

### NGC NeMo container

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

## Key Features

*   [Large Language Models](nemo/collections/nlp/README.md)
*   [Multimodal Models](nemo/collections/multimodal/README.md)
*   [Automatic Speech Recognition](nemo/collections/asr/README.md)
*   [Text to Speech](nemo/collections/tts/README.md)
*   [Computer Vision](nemo/collections/vision/README.md)

## Requirements

*   Python 3.10 or above
*   Pytorch 2.5 or above
*   NVIDIA GPU (for training)

## Developer Documentation

Refer to the official [NVIDIA NeMo Documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/) for the latest information.

## Contribute

We welcome community contributions! Please refer to [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md) for details.

## Publications

Explore the [publications](https://nvidia.github.io/NeMo/publications/) that utilize the NeMo Framework.

## Blogs

*   [Bria Builds Responsible Generative AI for Enterprises Using NVIDIA NeMo, Picasso](https://blogs.nvidia.com/blog/bria-builds-responsible-generative-ai-using-nemo-picasso/)
*   [New NVIDIA NeMo Framework Features and NVIDIA H200 Supercharge LLM Training Performance and Versatility](https://developer.nvidia.com/blog/new-nvidia-nemo-framework-features-and-nvidia-h200-supercharge-llm-training-performance-and-versatility)
*   [NVIDIA now powers training for Amazon Titan Foundation models](https://blogs.nvidia.com/blog/nemo-amazon-titan/)

## License

NeMo is licensed under the [Apache License 2.0](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file).