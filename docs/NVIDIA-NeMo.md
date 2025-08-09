<!-- Project Status Badges -->
[![Project Status: Active](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# NVIDIA NeMo: Build, Customize, and Deploy Generative AI Models at Scale

**NVIDIA NeMo is a cloud-native framework that empowers researchers and developers to efficiently build, customize, and deploy state-of-the-art generative AI models for LLMs, MMs, ASR, TTS, and CV.** [Explore the original NeMo repository](https://github.com/NVIDIA/NeMo).

## Key Features:

*   **Comprehensive Domain Support**: Build and customize models across Large Language Models (LLMs), Multimodal Models (MMs), Automatic Speech Recognition (ASR), Text-to-Speech (TTS), and Computer Vision (CV).
*   **Scalable Training**: Train models efficiently across thousands of GPUs with built-in support for distributed training techniques like Tensor Parallelism, Pipeline Parallelism, and FSDP.
*   **Modular Design**: Leverage PyTorch Lightning's modular abstractions for easier experimentation and customization.
*   **Pre-trained Models**: Utilize a wide range of pre-trained models available on Hugging Face Hub and NVIDIA NGC, enabling quickstart and transfer learning.
*   **Optimized Deployment**: Deploy and optimize models using NVIDIA Riva and NeMo Microservices for production-ready applications.
*   **Parameter Efficient Fine-tuning (PEFT)**: Supports the latest PEFT techniques such as LoRA, P-Tuning, Adapters, and IA3.
*   **NeMo-Run**: Seamlessly scale large-scale experiments across thousands of GPUs.

## Latest News and Updates:

*   **Hugging Face AutoModel Support:** Utilize AutoModel for CausalLM and AutoModelForImageTextToText models. ([Run Hugging Face Models Instantly with Day-0 Support from NVIDIA NeMo Framework](https://developer.nvidia.com/blog/run-hugging-face-models-instantly-with-day-0-support-from-nvidia-nemo-framework))
*   **Blackwell Support**: Added support for NVIDIA Blackwell architecture.
*   **Training Performance Guide**: Comprehensive guide published for performance tuning. ([Training Performance on GPU Tuning Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance-guide.html))
*   **New Models Support:** Support for Llama 4, Flux, Llama Nemotron, Hyena & Evo2, Qwen2-VL, Qwen2.5, Gemma3, Qwen3-30B&32B.
*   **NeMo 2.0 Release**: Major update focused on modularity and ease-of-use. ([NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/index.html))

## Introduction

The NVIDIA NeMo Framework is a cloud-native, end-to-end platform designed to simplify the development of generative AI models. Built for researchers and PyTorch developers, NeMo offers a streamlined approach to building, customizing, and deploying LLMs, MMs, ASR, TTS, and CV models.

For detailed technical information, please refer to the [NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html).

## What's New in NeMo 2.0

NeMo 2.0 introduces major advancements for enhanced flexibility, performance, and scalability.

*   **Python-Based Configuration**: Utilize a Python-based configuration system for improved control and customizability.
*   **Modular Abstractions**: Leverage PyTorch Lightning’s modular abstractions for simpler experimentation.
*   **Scalability**: Scale large-scale experiments using [NeMo-Run](https://github.com/NVIDIA/NeMo-Run).

> [!IMPORTANT]  
> NeMo 2.0 is currently supported by the LLM (large language model) and VLM (vision language model) collections.

### Get Started with NeMo 2.0

*   [Quickstart](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/quickstart.html)
*   [NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/index.html)
*   [NeMo 2.0 Recipes](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/recipes)
*   [Feature Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/features/index.html#feature-guide)
*   [Migration Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/migration/index.html#migration-guide)

### Get Started with Cosmos

Learn about video curation and post-training using the [Cosmos World Foundation Models](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/cosmos/collections/cosmos).

## LLMs and MMs Training, Alignment, and Customization

NeMo models are trained with [Lightning](https://github.com/Lightning-AI/lightning) and are scalable to thousands of GPUs.

NeMo models leverage cutting-edge distributed training techniques, including:
* Tensor Parallelism (TP)
* Pipeline Parallelism (PP)
* Fully Sharded Data Parallelism (FSDP)
* Mixture-of-Experts (MoE)
* Mixed Precision Training with BFloat16 and FP8

NeMo LLMs can be aligned with state-of-the-art methods such as SteerLM, Direct Preference Optimization (DPO), and Reinforcement Learning from Human Feedback (RLHF).

In addition to supervised fine-tuning (SFT), NeMo also supports the
latest parameter efficient fine-tuning (PEFT) techniques such as LoRA,
P-Tuning, Adapters, and IA3.

## LLMs and MMs Deployment and Optimization

NeMo LLMs and MMs can be deployed and optimized with [NVIDIA NeMo
Microservices](https://developer.nvidia.com/nemo-microservices-early-access).

## Speech AI

NeMo ASR and TTS models can be optimized for inference and deployed for
production use cases with [NVIDIA Riva](https://developer.nvidia.com/riva).

## Get Started with NeMo Framework

Get started quickly with pre-trained models from [Hugging Face Hub](https://huggingface.co/models?library=nemo&sort=downloads&search=nvidia)
and [NVIDIA NGC](https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC).

Extensive tutorials can be run on [Google Colab](https://colab.research.google.com) or
with our [NGC NeMo Framework
Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo).
We also have
[playbooks](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html)
for users who want to train NeMo models with the NeMo Framework
Launcher.

For advanced users who want to train NeMo models from scratch or
fine-tune existing NeMo models, we have a full suite of [example
scripts](https://github.com/NVIDIA/NeMo/tree/main/examples) that support
multi-GPU/multi-node training.

## Requirements

*   Python 3.10 or above
*   PyTorch 2.5 or above
*   NVIDIA GPU (required for model training)

## Install NeMo Framework

Choose the appropriate installation method based on your needs:
*   [Conda / Pip](#conda--pip):  Recommended for ASR and TTS domains.
*   [NGC PyTorch container](#ngc-pytorch-container):  Install from source into an optimized container.
*   [NGC NeMo container](#ngc-nemo-container):  Ready-to-go solution for high performance.

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

## Developer Documentation

| Version | Status                                                                                                                                                              | Description                                                                                                                    |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Latest  | [![Documentation Status](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)     | [Documentation of the latest (i.e. main) branch.](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)          |
| Stable  | [![Documentation Status](https://readthedocs.com/projects/nvidia-nemo/badge/?version=stable)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/) | [Documentation of the stable (i.e. most recent release)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/) |

## Contribute to NeMo

We welcome community contributions!  Refer to [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md) for the process.

## Discussions Board

Find answers to your questions and engage in discussions on the NeMo [Discussions
board](https://github.com/NVIDIA/NeMo/discussions).

## Publications

Explore the collection of [publications](https://nvidia.github.io/NeMo/publications/) that utilize the NeMo Framework.

## Blogs

**(Blog entries have been maintained, condensing the original format)**

*   [Bria Builds Responsible Generative AI for Enterprises Using NVIDIA NeMo, Picasso](https://blogs.nvidia.com/blog/bria-builds-responsible-generative-ai-using-nemo-picasso/)
*   [New NVIDIA NeMo Framework Features and NVIDIA H200](https://developer.nvidia.com/blog/new-nvidia-nemo-framework-features-and-nvidia-h200-supercharge-llm-training-performance-and-versatility/)
*   [NVIDIA now powers training for Amazon Titan Foundation models](https://blogs.nvidia.com/blog/nemo-amazon-titan/)

## Licenses

NVIDIA NeMo is licensed under the [Apache License 2.0](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file).