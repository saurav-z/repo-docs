[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# NVIDIA NeMo: Build, Customize, and Deploy Generative AI Models

**NVIDIA NeMo is a cloud-native framework that simplifies the development and deployment of state-of-the-art generative AI models; find the original repo [here](https://github.com/NVIDIA/NeMo).**

## Key Features

*   **Comprehensive AI Domains:** Supports Large Language Models (LLMs), Multimodal Models (MMs), Automatic Speech Recognition (ASR), Text-to-Speech (TTS), and Computer Vision (CV).
*   **Scalable Training:** Train models efficiently across thousands of GPUs with techniques like Tensor Parallelism, Pipeline Parallelism, FSDP, and Mixed Precision Training.
*   **Modular and Flexible:** Utilize Python-based configuration and modular abstractions for easy customization and experimentation.
*   **Pre-trained Models & Integration:** Leverage readily available pre-trained models from Hugging Face Hub and NVIDIA NGC and seamlessly integrate with NVIDIA Riva for deployment.
*   **Parameter-Efficient Fine-tuning (PEFT):** Supports LoRA, P-Tuning, Adapters, and IA3 for efficient model customization.
*   **Model Alignment:** Tools available to align LLMs with SteerLM, Direct Preference Optimization (DPO), and Reinforcement Learning from Human Feedback (RLHF)
*   **Advanced Deployment:** Deploy and optimize models with NVIDIA NeMo Microservices.

## Latest Updates

*   **Support for Hugging Face Models:**  NeMo's AutoModel enables broad support for Hugging Face models, including *AutoModelForCausalLM* and *AutoModelForImageTextToText*. ([Blog](https://developer.nvidia.com/blog/run-hugging-face-models-instantly-with-day-0-support-from-nvidia-nemo-framework))
*   **Blackwell Support and GPU Tuning Guide:** NeMo now supports Blackwell with performance benchmarks and a comprehensive guide to performance tuning. ([Performance Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance-guide.html))
*   **Expanded Model Support:** Added support for new community models, including Llama 4, Flux, Llama Nemotron, Hyena & Evo2, Qwen2-VL, Qwen2.5, Gemma3, Qwen3-30B&32B.
*   **NeMo 2.0 Release:**  Focuses on modularity and ease-of-use, featuring Python-based configuration and modular abstractions. ([NeMo 2.0 User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/index.html))
*   **Cosmos World Foundation Model Support:** Supports training and customization of NVIDIA Cosmos models. ([Cosmos Blog](https://developer.nvidia.com/blog/advancing-physical-ai-with-nvidia-cosmos-world-foundation-model-platform))
*   **Performance Records:** Achieved new generative AI performance records in MLPerf Training v4.0.

## Introduction

NVIDIA NeMo Framework is a scalable and cloud-native generative AI framework built for researchers and PyTorch developers working on Large Language Models (LLMs), Multimodal Models (MMs), Automatic Speech Recognition (ASR), Text to Speech (TTS), and Computer Vision (CV) domains. It is designed to help you efficiently create, customize, and deploy new generative AI models by leveraging existing code and pre-trained model checkpoints.

For technical documentation, please see the [NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html).

## Getting Started

### Quick Start
Quickstart guide for using NeMo-Run to launch NeMo 2.0 experiments locally and on a slurm cluster:
[Quickstart](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/quickstart.html)

### Additional Information

*   **NeMo 2.0 User Guide:** Comprehensive information on NeMo 2.0 features.
    [NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/index.html)
*   **NeMo 2.0 Recipes:** Examples of large-scale runs using NeMo 2.0 and NeMo-Run.
    [NeMo 2.0 Recipes](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/recipes)
*   **Feature Guide:** Detailed exploration of NeMo 2.0 main features.
    [Feature Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/features/index.html#feature-guide)
*   **Migration Guide:** Transition from NeMo 1.0 to 2.0.
    [Migration Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/migration/index.html#migration-guide)

### Pre-trained Models

Easily generate text, transcribe audio, or synthesize speech with pre-trained models available on:

*   [Hugging Face Hub](https://huggingface.co/models?library=nemo&sort=downloads&search=nvidia)
*   [NVIDIA NGC](https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC)

## LLMs, MMs, Speech, and Computer Vision

*   **Training:** Utilize Lightning for scalable training across thousands of GPUs, with techniques like Tensor Parallelism, Pipeline Parallelism, and Mixed Precision Training.
*   **Alignment & Customization:** Align LLMs with state-of-the-art methods and supports parameter-efficient fine-tuning (PEFT) techniques.
*   **Deployment and Optimization:** Deploy and optimize LLMs and MMs using NVIDIA NeMo Microservices and optimize ASR and TTS models with NVIDIA Riva.

## Installation

Choose an installation method based on your needs:

*   **Conda / Pip:** For general exploration and ASR/TTS domains.
    ```bash
    conda create --name nemo python==3.10.12
    conda activate nemo
    pip install "nemo_toolkit[all]"
    ```
    or
    ```bash
    git clone https://github.com/NVIDIA/NeMo
    cd NeMo
    git checkout @${REF:-'main'}
    pip install '.[all]'
    ```
*   **NGC PyTorch Container:** Install from source within an optimized container.
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
*   **NGC NeMo Container:** For highest performance, using a pre-built container.
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
    

## Future Work

The NeMo Framework Launcher is currently not available for ASR and TTS training, however, it is coming soon.

## Resources

*   **Discussions Board:** Find answers and engage with the community on the [Discussions board](https://github.com/NVIDIA/NeMo/discussions).
*   **Contribute:** Contribute to NeMo; see [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md).
*   **Publications:** Explore [publications](https://nvidia.github.io/NeMo/publications/) using NeMo.
*   **Blogs:**
    *   [Bria Builds Responsible Generative AI for Enterprises Using NVIDIA NeMo, Picasso](https://blogs.nvidia.com/blog/bria-builds-responsible-generative-ai-using-nemo-picasso/)
    *   [New NVIDIA NeMo Framework Features and NVIDIA H200](https://developer.nvidia.com/blog/new-nvidia-nemo-framework-features-and-nvidia-h200-supercharge-llm-training-performance-and-versatility/)
    *   [NVIDIA now powers training for Amazon Titan Foundation models](https://blogs.nvidia.com/blog/nemo-amazon-titan/)

## License

NeMo is licensed under the [Apache License 2.0](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file).