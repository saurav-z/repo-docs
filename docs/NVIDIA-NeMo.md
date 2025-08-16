# NVIDIA NeMo: Build, Customize, and Deploy Generative AI Models

**NVIDIA NeMo is a flexible, cloud-native framework for researchers and developers to build, customize, and deploy state-of-the-art generative AI models across various domains.**  Learn more about NVIDIA NeMo on the [original repository](https://github.com/NVIDIA/NeMo).

[![Project Status: Active](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Key Features

*   **Large Language Models (LLMs):**  Train, fine-tune, and deploy LLMs efficiently.
*   **Multimodal Models (MMs):**  Develop models that combine text, images, and more.
*   **Automatic Speech Recognition (ASR):**  Build and optimize cutting-edge ASR models.
*   **Text-to-Speech (TTS):**  Create high-quality speech synthesis systems.
*   **Computer Vision (CV):**  Implement advanced computer vision models.
*   **Modular Design:** Leverages PyTorch Lightning for ease of use and extensibility.
*   **Scalability:** Supports training across thousands of GPUs using NeMo-Run and advanced parallelism techniques.
*   **Pre-trained Models:** Access a wide array of pre-trained models on Hugging Face Hub and NVIDIA NGC.
*   **Deployment and Optimization:** Integrate with NVIDIA Riva for optimized inference and production deployment.

## What's New

*   **Hugging Face Support:**  Seamlessly pretrain and fine-tune Hugging Face models using AutoModel.
*   **Blackwell Support:**  Added support for NVIDIA Blackwell with performance benchmarks.
*   **Performance Tuning Guide:**  A comprehensive guide is available for performance tuning to achieve optimal throughput!
*   **New Model Support:** added support for latest community models - Llama 4, Flux, Llama Nemotron, Hyena & Evo2, Qwen2-VL, Qwen2.5, Gemma3, Qwen3-30B&32B.
*   **NeMo 2.0:**  Release of NeMo 2.0, emphasizing modularity and ease-of-use, with a transition to Python-based configuration.
*   **Cosmos World Foundation Models Support:** Expanded support for NVIDIA Cosmos models for physical AI.

## Introduction

The NVIDIA NeMo (Neural Modules) Framework is designed for researchers and PyTorch developers working on generative AI models. It offers a comprehensive toolkit for building, customizing, and deploying LLMs, MMs, ASR, TTS, and CV models.  NeMo leverages existing code and pre-trained checkpoints to streamline the development process.

For detailed technical information, consult the [NeMo Framework User Guide](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/).

## Getting Started

### Get Started with NeMo 2.0

-   Refer to the [Quickstart](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/quickstart.html) for examples of using NeMo-Run to launch NeMo 2.0 experiments locally and on a slurm cluster.
-   For more information about NeMo 2.0, see the [NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/index.html).
-   [NeMo 2.0 Recipes](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/recipes) contains additional examples of launching large-scale runs using NeMo 2.0 and NeMo-Run.
-   For an in-depth exploration of the main features of NeMo 2.0, see the [Feature Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/features/index.html#feature-guide).
-   To transition from NeMo 1.0 to 2.0, see the [Migration Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/migration/index.html#migration-guide) for step-by-step instructions.

### Get Started with Cosmos

NeMo Curator and NeMo Framework support video curation and post-training of the Cosmos World Foundation Models, which are open and available on [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/cosmos/collections/cosmos) and [Hugging Face](https://huggingface.co/collections/nvidia/cosmos-6751e884dc10e013a0a0d8e6). For more information on video datasets, refer to [NeMo Curator](https://developer.nvidia.com/nemo-curator). To post-train World Foundation Models using the NeMo Framework for your custom physical AI tasks, see the [Cosmos Diffusion models](https://github.com/NVIDIA/Cosmos/blob/main/cosmos1/models/diffusion/nemo/post_training/README.md) and the [Cosmos Autoregressive models](https://github.com/NVIDIA/Cosmos/blob/main/cosmos1/models/autoregressive/nemo/post_training/README.md).

## Training, Alignment, and Customization of LLMs and MMs

NeMo models utilize PyTorch Lightning for scalable training across many GPUs. Key training features include:

*   **Parallelism Strategies:** Tensor Parallelism (TP), Pipeline Parallelism (PP), Fully Sharded Data Parallelism (FSDP), Mixture-of-Experts (MoE), and mixed precision training.
*   **NVIDIA Transformer Engine:**  Utilized for FP8 training on NVIDIA Hopper GPUs.
*   **NVIDIA Megatron Core:** Used for scaling Transformer model training.
*   **Alignment Techniques:** Supports SteerLM, Direct Preference Optimization (DPO), and Reinforcement Learning from Human Feedback (RLHF) for LLM alignment.
*   **Parameter Efficient Fine-tuning (PEFT):** Includes support for LoRA, P-Tuning, Adapters, and IA3.

## Deployment and Optimization of LLMs and MMs

*   **NVIDIA NeMo Microservices:**  Deploy and optimize NeMo LLMs and MMs.
*   **NVIDIA Riva:** Optimize ASR and TTS models for inference and production use.

## Speech AI

NeMo ASR and TTS models can be optimized for inference and deployed for
production use cases with [NVIDIA Riva](https://developer.nvidia.com/riva).

## NeMo Framework Launcher

> [!IMPORTANT]  
> NeMo Framework Launcher is compatible with NeMo version 1.0 only. [NeMo-Run](https://github.com/NVIDIA/NeMo-Run) is recommended for launching experiments using NeMo 2.0.

[NeMo Framework
Launcher](https://github.com/NVIDIA/NeMo-Megatron-Launcher) is a
cloud-native tool that streamlines the NeMo Framework experience. It is
used for launching end-to-end NeMo Framework training jobs on CSPs and
Slurm clusters.

The NeMo Framework Launcher includes extensive recipes, scripts,
utilities, and documentation for training NeMo LLMs. It also includes
the NeMo Framework [Autoconfigurator](https://github.com/NVIDIA/NeMo-Megatron-Launcher#53-using-autoconfigurator-to-find-the-optimal-configuration),
which is designed to find the optimal model parallel configuration for
training on a specific cluster.

To get started quickly with the NeMo Framework Launcher, please see the
[NeMo Framework
Playbooks](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html).
The NeMo Framework Launcher does not currently support ASR and TTS
training, but it will soon.

## Requirements

*   Python 3.10 or above
*   PyTorch 2.5 or above
*   NVIDIA GPU (for model training)

## Install NeMo Framework

Choose the installation method that best fits your needs:

*   **Conda / Pip:** (Recommended for ASR and TTS) Install NeMo with native Pip into a virtual environment.
*   **NGC PyTorch container:** Install from source within an optimized NVIDIA PyTorch container.
*   **NGC NeMo container:** Use a pre-built, optimized container for maximum performance.

### Support matrix

NeMo-Framework provides tiers of support based on OS / Platform and mode of installation. Please refer the following overview of support levels:

- Fully supported: Max performance and feature-completeness.
- Limited supported: Used to explore NeMo.
- No support yet: In development.
- Deprecated: Support has reached end of life.

Please refer to the following table for current support levels:

| OS / Platform              | Install from PyPi | Source into NGC container |
|----------------------------|-------------------|---------------------------|
| `linux` - `amd64/x84_64`   | Limited support   | Full support              |
| `linux` - `arm64`          | Limited support   | Limited support           |
| `darwin` - `amd64/x64_64`  | Deprecated        | Deprecated                |
| `darwin` - `arm64`         | Limited support   | Limited support           |
| `windows` - `amd64/x64_64` | No support yet    | No support yet            |
| `windows` - `arm64`        | No support yet    | No support yet            |

### Conda / Pip

```bash
conda create --name nemo python==3.10.12
conda activate nemo
pip install "nemo_toolkit[all]"
```
or, using a specific Git reference:

```bash
git clone https://github.com/NVIDIA/NeMo
cd NeMo
git checkout @${REF:-'main'}
pip install '.[all]'
```
Install a specific domain:
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

## Developer Documentation

| Version | Status                                                                                                                                                              | Description                                                                                                                    |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Latest  | [![Documentation Status](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)     | [Documentation of the latest (i.e. main) branch.](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)          |
| Stable  | [![Documentation Status](https://readthedocs.com/projects/nvidia-nemo/badge/?version=stable)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/) | [Documentation of the stable (i.e. most recent release)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/) |

## Future Work

*   Continued improvements to NeMo Framework Launcher.

## Discussions Board

Find answers and participate in discussions on the NeMo [Discussions board](https://github.com/NVIDIA/NeMo/discussions).

## Contribute to NeMo

Contribute to the NeMo project by following the guidelines in [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md).

## Publications

Explore publications that utilize NeMo at [publications](https://nvidia.github.io/NeMo/publications/).  Contribute articles via pull requests to the `gh-pages-src` branch.

## Blogs

(See the original README for links to blog posts.)

## Licenses

NeMo is licensed under the [Apache License 2.0](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file).