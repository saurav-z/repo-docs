[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# NVIDIA NeMo: Build and Deploy Generative AI Models with Ease

NVIDIA NeMo is a cloud-native framework empowering researchers and developers to create, customize, and deploy state-of-the-art generative AI models.  [Explore the NeMo repository](https://github.com/NVIDIA/NeMo).

## Key Features

*   **Large Language Models (LLMs):**  Train and customize powerful LLMs.
*   **Multimodal Models (MMs):** Develop AI models that process multiple data types (text, images, etc.).
*   **Automatic Speech Recognition (ASR):**  Build and deploy advanced speech recognition systems.
*   **Text-to-Speech (TTS):** Create high-quality text-to-speech applications.
*   **Computer Vision (CV):** Leverage cutting-edge computer vision capabilities.

## What's New

*   **[Hugging Face Integration](https://developer.nvidia.com/blog/run-hugging-face-models-instantly-with-day-0-support-from-nvidia-nemo-framework):** Seamlessly integrate and fine-tune Hugging Face models.
*   **Blackwell Support:** Optimized performance on NVIDIA Blackwell hardware.
*   **Performance Tuning Guide:**  Comprehensive guide to achieving optimal throughput.
*   **New Model Support:**  Expanded support for cutting-edge community models like Llama 4, Flux, and more.
*   **NeMo 2.0:**  A major update focusing on modularity and ease of use.
*   **Cosmos World Foundation Models:** Support for training and customizing the NVIDIA Cosmos collection of world foundation models for physical AI applications.

## Introduction

NVIDIA NeMo Framework simplifies the development and deployment of generative AI models.  It is a cloud-native framework designed for researchers and PyTorch developers working with Large Language Models (LLMs), Multimodal Models (MMs), Automatic Speech Recognition (ASR), Text to Speech (TTS), and Computer Vision (CV) domains. NeMo helps you efficiently create, customize, and deploy new generative AI models by leveraging existing code and pre-trained model checkpoints.

For detailed technical information, refer to the [NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html).

### NeMo 2.0 Overview

NeMo 2.0 is a major upgrade with several key improvements:

*   **Python-Based Configuration:**  Offers more flexibility and programmatic control.
*   **Modular Abstractions:**  Simplifies experimentation and modification.
*   **Scalability:** Supports large-scale experiments using [NeMo-Run](https://github.com/NVIDIA/NeMo-Run).

> [!IMPORTANT]
>  NeMo 2.0 is currently supported by the LLM (large language model) and VLM (vision language model) collections.

#### Get Started with NeMo 2.0

*   [Quickstart](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/quickstart.html) - Using NeMo-Run.
*   [NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/index.html) - Learn more about NeMo 2.0.
*   [NeMo 2.0 Recipes](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/recipes) - Examples of large-scale runs.
*   [Feature Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/features/index.html#feature-guide) - Main features of NeMo 2.0.
*   [Migration Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/migration/index.html#migration-guide) - Migrate from NeMo 1.0 to 2.0.

### Get Started with Cosmos

NeMo Curator and NeMo Framework support video curation and post-training of the Cosmos World Foundation Models, which are open and available on [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/cosmos/collections/cosmos) and [Hugging Face](https://huggingface.co/collections/nvidia/cosmos-6751e884dc10e013a0a0d8e6). For more information on video datasets, refer to [NeMo Curator](https://developer.nvidia.com/nemo-curator). To post-train World Foundation Models using the NeMo Framework for your custom physical AI tasks, see the [Cosmos Diffusion models](https://github.com/NVIDIA/Cosmos/blob/main/cosmos1/models/diffusion/nemo/post_training/README.md) and the [Cosmos Autoregressive models](https://github.com/NVIDIA/Cosmos/blob/main/cosmos1/models/autoregressive/nemo/post_training/README.md).

## LLMs and MMs: Training, Alignment, and Customization

NeMo models are trained using [Lightning](https://github.com/Lightning-AI/lightning), enabling automatic scalability across thousands of GPUs.  Performance benchmarks are available [here](https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance_summary.html).

Key training features include:

*   **Distributed Training:** Utilizes parallelism strategies like TP, PP, FSDP, MoE, and mixed precision for efficient large-model training.
*   **NVIDIA Technologies:** Leverages NVIDIA Transformer Engine for FP8 training on Hopper GPUs and NVIDIA Megatron Core for scaling.
*   **Alignment Techniques:** Supports state-of-the-art methods such as SteerLM, DPO, and RLHF (via [NVIDIA NeMo Aligner](https://github.com/NVIDIA/NeMo-Aligner)).
*   **PEFT Support:** Implements parameter-efficient fine-tuning techniques (LoRA, P-Tuning, Adapters, IA3).

## LLMs and MMs: Deployment and Optimization

NeMo LLMs and MMs can be deployed and optimized using [NVIDIA NeMo Microservices](https://developer.nvidia.com/nemo-microservices-early-access).

## Speech AI

NeMo ASR and TTS models can be optimized for inference and deployed for production using [NVIDIA Riva](https://developer.nvidia.com/riva).

## NeMo Framework Launcher (v1.0 - for NeMo 1.0)

> [!IMPORTANT]  
> NeMo Framework Launcher is compatible with NeMo version 1.0 only. [NeMo-Run](https://github.com/NVIDIA/NeMo-Run) is recommended for launching experiments using NeMo 2.0.

[NeMo Framework Launcher](https://github.com/NVIDIA/NeMo-Megatron-Launcher) is a cloud-native tool that streamlines the NeMo Framework experience for launching end-to-end NeMo Framework training jobs on CSPs and Slurm clusters.

*   Includes extensive recipes, scripts, and utilities for training NeMo LLMs.
*   Features the [Autoconfigurator](https://github.com/NVIDIA/NeMo-Megatron-Launcher#53-using-autoconfigurator-to-find-the-optimal-configuration) to optimize model parallel configurations.
*   Get started with the [NeMo Framework Playbooks](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html).

## Get Started with NeMo Framework

Get started quickly with pre-trained models from:

*   [Hugging Face Hub](https://huggingface.co/models?library=nemo&sort=downloads&search=nvidia)
*   [NVIDIA NGC](https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC)

Access comprehensive resources:

*   [Tutorials](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/starthere/tutorials.html) on Google Colab or with the [NGC NeMo Framework Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo).
*   [Playbooks](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html) for training.
*   [Example Scripts](https://github.com/NVIDIA/NeMo/tree/main/examples) for advanced users.

## Requirements

*   Python 3.10 or above
*   PyTorch 2.5 or above
*   NVIDIA GPU (for model training)

## Developer Documentation

| Version | Status                                                                                                                                                              | Description                                                                                                                    |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Latest  | [![Documentation Status](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)     | [Documentation of the latest (i.e. main) branch.](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)          |
| Stable  | [![Documentation Status](https://readthedocs.com/projects/nvidia-nemo/badge/?version=stable)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/) | [Documentation of the stable (i.e. most recent release)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/) |

## Installation

Choose your installation method based on your needs:

*   **[Conda / Pip](#conda--pip):** Recommended for ASR and TTS, and for exploring NeMo.
*   **[NGC PyTorch Container](#ngc-pytorch-container):**  For installing from source within an optimized container.
*   **[NGC NeMo Container](#ngc-nemo-container):** For peak performance.

### Support Matrix

Provides tiers of support based on OS / Platform and mode of installation. Please refer the following overview of support levels:

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
```

#### Pick the right version

```bash
pip install "nemo_toolkit[all]"
```

If a more specific version is desired, we recommend a Pip-VCS install. From [NVIDIA/NeMo](github.com/NVIDIA/NeMo), fetch the commit, branch, or tag that you would like to install.  
To install nemo_toolkit from this Git reference `$REF`, use the following installation method:

```bash
git clone https://github.com/NVIDIA/NeMo
cd NeMo
git checkout @${REF:-'main'}
pip install '.[all]'
```

#### Install a specific Domain

To install a specific domain of NeMo, you must first install the
nemo_toolkit using the instructions listed above. Then, you run the
following domain-specific commands:

```bash
pip install nemo_toolkit['all'] # or pip install "nemo_toolkit['all']@git+https://github.com/NVIDIA/NeMo@${REF:-'main'}"
pip install nemo_toolkit['asr'] # or pip install "nemo_toolkit['asr']@git+https://github.com/NVIDIA/NeMo@$REF:-'main'}"
pip install nemo_toolkit['nlp'] # or pip install "nemo_toolkit['nlp']@git+https://github.com/NVIDIA/NeMo@${REF:-'main'}"
pip install nemo_toolkit['tts'] # or pip install "nemo_toolkit['tts']@git+https://github.com/NVIDIA/NeMo@${REF:-'main'}"
pip install nemo_toolkit['vision'] # or pip install "nemo_toolkit['vision']@git+https://github.com/NVIDIA/NeMo@${REF:-'main'}"
pip install nemo_toolkit['multimodal'] # or pip install "nemo_toolkit['multimodal']@git+https://github.com/NVIDIA/NeMo@${REF:-'main'}"
```

### NGC PyTorch container

**NOTE: The following steps are supported beginning with 24.04 (NeMo-Toolkit 2.3.0)**

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

From [NVIDIA/NeMo](github.com/NVIDIA/NeMo), fetch the commit/branch/tag that you want to install.  
To install nemo_toolkit including all of its dependencies from this Git reference `$REF`, use the following installation method:

```bash
cd /opt
git clone https://github.com/NVIDIA/NeMo
cd NeMo
git checkout ${REF:-'main'}
bash docker/common/install_dep.sh --library all
pip install ".[all]"
```

## NGC NeMo container

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

The NeMo Framework Launcher does not currently support ASR and TTS
training, but it will soon.

## Discussions Board

Find answers to frequently asked questions on the [NeMo Discussions board](https://github.com/NVIDIA/NeMo/discussions).  Ask questions or start discussions there.

## Contribute to NeMo

We welcome community contributions!  See [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md) for guidelines.

## Publications

Explore the growing list of [publications](https://nvidia.github.io/NeMo/publications/) utilizing the NeMo Framework.  Contribute your work via a pull request to the `gh-pages-src` branch.

## Blogs

```markdown
  **Large Language Models and Multimodal Models**
    *   **Bria Builds Responsible Generative AI for Enterprises Using NVIDIA NeMo, Picasso**
        (2024/03/06)

    *   **New NVIDIA NeMo Framework Features and NVIDIA H200**
        (2023/12/06)

    *   **NVIDIA now powers training for Amazon Titan Foundation models**
        (2023/11/28)
```

## Licenses

NeMo is licensed under the [Apache License 2.0](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file).