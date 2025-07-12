# NVIDIA NeMo: Build, Customize, and Deploy Generative AI Models with Ease

**NVIDIA NeMo is a powerful, cloud-native framework that simplifies the development of state-of-the-art generative AI models across various domains.** [Visit the original repository](https://github.com/NVIDIA/NeMo) for more details.

[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Key Features

*   **Large Language Models (LLMs):** Train, fine-tune, and deploy cutting-edge LLMs.
*   **Multimodal Models (MMs):** Develop AI models that process and generate data across multiple modalities.
*   **Automatic Speech Recognition (ASR):** Build and deploy high-accuracy speech recognition systems.
*   **Text-to-Speech (TTS):** Create realistic and expressive text-to-speech applications.
*   **Computer Vision (CV):** Implement advanced computer vision solutions.
*   **Modular and Extensible:** Leverage PyTorch Lightning's modular design for easy customization.
*   **Scalable Training:** Train models efficiently across thousands of GPUs using NeMo-Run and advanced parallelism techniques.
*   **Pre-trained Models:** Access a wide variety of pre-trained models on Hugging Face Hub and NVIDIA NGC.
*   **Deployment and Optimization:** Utilize NVIDIA Riva and NeMo Microservices for production-ready deployment.
*   **Integration with Cutting-Edge Techniques:** Includes support for FP8 training, MoE, RLHF, and PEFT methods.

## What's New

Stay up-to-date with the latest advancements in NeMo:

*   **[Pretrain and finetune Hugging Face models via AutoModel]** (https://developer.nvidia.com/blog/run-hugging-face-models-instantly-with-day-0-support-from-nvidia-nemo-framework)
*   **[Training on Blackwell using Nemo]** ([Performance benchmarks on GB200 & B200](https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance_summary.html)
*   **[Training Performance on GPU Tuning Guide]** ([a comprehensive guide for performance tuning](https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance-guide.html)
*   **[New Models Support]** ([Llama 4, Flux, Llama Nemotron, Hyena & Evo2, Qwen2-VL, Qwen2.5, Gemma3, Qwen3-30B&32B](https://docs.nvidia.com/nemo-framework/user-guide/latest/vlms/llama4.html)
*   **[NeMo Framework 2.0]** ([Refer to the NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/index.html)
*   **[New Cosmos World Foundation Models Support]** ([Advancing Physical AI with NVIDIA Cosmos World Foundation Model Platform](https://developer.nvidia.com/blog/advancing-physical-ai-with-nvidia-cosmos-world-foundation-model-platform)
*   **[Large Language Models and Multimodal Models]** ([State-of-the-Art Multimodal Generative AI Model Development with NVIDIA NeMo](https://developer.nvidia.com/blog/state-of-the-art-multimodal-generative-ai-model-development-with-nvidia-nemo/)
*   **[Speech Recognition]** ([Accelerating Leaderboard-Topping ASR Models 10x with NVIDIA NeMo](https://developer.nvidia.com/blog/accelerating-leaderboard-topping-asr-models-10x-with-nvidia-nemo/)

## Getting Started

*   **Tutorials:** Explore comprehensive tutorials that can be run on Google Colab or with our NGC NeMo Framework Container ([Tutorials](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/starthere/tutorials.html))
*   **Example Scripts:** Utilize example scripts for multi-GPU/multi-node training ([Example Scripts](https://github.com/NVIDIA/NeMo/tree/main/examples))
*   **Playbooks:** Leverage pre-built playbooks for streamlined model training ([Playbooks](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html))

## Installation

Choose the installation method that best fits your needs:

### Conda / Pip

```bash
conda create --name nemo python==3.10.12
conda activate nemo
pip install "nemo_toolkit[all]"
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

## Resources

*   **Documentation:** Comprehensive user guide and API reference ([Developer Documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/))
*   **Discussions:** Engage with the community and ask questions ([Discussions Board](https://github.com/NVIDIA/NeMo/discussions))
*   **Publications:** Explore research papers utilizing NeMo ([Publications](https://nvidia.github.io/NeMo/publications/))
*   **Contribute:** Learn how to contribute to the project ([CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md))

## Licenses

NeMo is licensed under the [Apache License 2.0](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file).