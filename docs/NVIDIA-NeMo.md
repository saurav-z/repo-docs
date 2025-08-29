[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# NVIDIA NeMo: Build and Deploy Generative AI Models with Ease

NVIDIA NeMo is a versatile framework empowering researchers and developers to create, customize, and deploy state-of-the-art generative AI models for LLMs, MMs, ASR, TTS, and CV.  Access the original repo [here](https://github.com/NVIDIA/NeMo).

## Key Features

*   **Large Language Models (LLMs):** Training, fine-tuning, and deployment.
*   **Multimodal Models (MMs):** Support for various data types and tasks.
*   **Automatic Speech Recognition (ASR):** Build and optimize speech recognition models.
*   **Text-to-Speech (TTS):** Develop and deploy high-quality text-to-speech systems.
*   **Computer Vision (CV):** Leverage the framework for computer vision tasks.
*   **Model Alignment:** Utilizes SteerLM, Direct Preference Optimization (DPO), and Reinforcement Learning from Human Feedback (RLHF)
*   **Parameter Efficient Fine-Tuning (PEFT):** Support for LoRA, P-Tuning, Adapters, and IA3.

## Latest Updates

*   **Pretrain and fine-tune Hugging Face models via AutoModel:** AutoModel support for models such as AutoModelForCausalLM and AutoModelForImageTextToText.
*   **Blackwell Support:** Adds support for Blackwell, with performance benchmarks on GB200 & B200.
*   **New Models Support:** Includes support for Llama 4, Flux, Llama Nemotron, Hyena & Evo2, Qwen2-VL, Qwen2.5, Gemma3, Qwen3-30B&32B.
*   **NVIDIA Cosmos World Foundation Models Support:** Accelerates video processing with optimized video processing and captioning features via NeMo Curator.
*   **Llama 3.1 Support:** Supports training and customizing the Llama 3.1 collection of LLMs from Meta.

## Introduction

The NVIDIA NeMo Framework is a cloud-native, scalable framework designed for researchers and PyTorch developers. It simplifies the process of creating, customizing, and deploying generative AI models across various domains. It provides efficient ways to train, fine-tune, and deploy models by leveraging existing code and pre-trained checkpoints.

For detailed technical documentation, please refer to the [NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html).

## What's New in NeMo 2.0

NVIDIA NeMo 2.0 offers significant improvements over its predecessor, including:

*   **Python-Based Configuration:** Provides more flexibility and control.
*   **Modular Abstractions:** Simplifies adaptation and experimentation through PyTorch Lightning’s modular approach.
*   **Scalability:** Seamlessly scales experiments across thousands of GPUs using [NeMo-Run](https://github.com/NVIDIA/NeMo-Run).

> [!IMPORTANT]  
> NeMo 2.0 is currently supported by the LLM (large language model) and VLM (vision language model) collections.

### Get Started with NeMo 2.0

*   Refer to the [Quickstart](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/quickstart.html) for examples of using NeMo-Run to launch NeMo 2.0 experiments locally and on a slurm cluster.
*   For more information about NeMo 2.0, see the [NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/index.html).
*   [NeMo 2.0 Recipes](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/recipes) contains additional examples of launching large-scale runs using NeMo 2.0 and NeMo-Run.
*   For an in-depth exploration of the main features of NeMo 2.0, see the [Feature Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/features/index.html#feature-guide).
*   To transition from NeMo 1.0 to 2.0, see the [Migration Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/migration/index.html#migration-guide) for step-by-step instructions.

### Get Started with Cosmos

*   NeMo Curator and NeMo Framework support video curation and post-training of the Cosmos World Foundation Models, which are open and available on [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/cosmos/collections/cosmos) and [Hugging Face](https://huggingface.co/collections/nvidia/cosmos-6751e884dc10e013a0a0d8e6).
*   For more information on video datasets, refer to [NeMo Curator](https://developer.nvidia.com/nemo-curator).
*   To post-train World Foundation Models using the NeMo Framework for your custom physical AI tasks, see the [Cosmos Diffusion models](https://github.com/NVIDIA/Cosmos/blob/main/cosmos1/models/diffusion/nemo/post_training/README.md) and the [Cosmos Autoregressive models](https://github.com/NVIDIA/Cosmos/blob/main/cosmos1/models/autoregressive/nemo/post_training/README.md).

## LLMs and MMs Training, Alignment, and Customization

NeMo models are built with [Lightning](https://github.com/Lightning-AI/lightning) and trained with cutting-edge distributed training techniques.  These techniques include Tensor Parallelism (TP), Pipeline Parallelism (PP), Fully Sharded Data Parallelism (FSDP), Mixture-of-Experts (MoE), and Mixed Precision Training with BFloat16 and FP8, as well as others.  NeMo LLMs can be aligned with state-of-the-art methods such as SteerLM, Direct Preference Optimization (DPO), and Reinforcement Learning from Human Feedback (RLHF). See [NVIDIA NeMo Aligner](https://github.com/NVIDIA/NeMo-Aligner) for more information.

## LLMs and MMs Deployment and Optimization

NeMo LLMs and MMs can be deployed and optimized with [NVIDIA NeMo Microservices](https://developer.nvidia.com/nemo-microservices-early-access).

## Speech AI

NeMo ASR and TTS models can be optimized for inference and deployed for production use cases with [NVIDIA Riva](https://developer.nvidia.com/riva).

## NeMo Framework Launcher (for NeMo 1.0)

> [!IMPORTANT]  
> NeMo Framework Launcher is compatible with NeMo version 1.0 only. [NeMo-Run](https://github.com/NVIDIA/NeMo-Run) is recommended for launching experiments using NeMo 2.0.

The [NeMo Framework Launcher](https://github.com/NVIDIA/NeMo-Megatron-Launcher) is a cloud-native tool for launching end-to-end NeMo Framework training jobs on CSPs and Slurm clusters.

To get started quickly with the NeMo Framework Launcher, please see the [NeMo Framework Playbooks](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html).

## Getting Started

Pretrained NeMo models are available on [Hugging Face Hub](https://huggingface.co/models?library=nemo&sort=downloads&search=nvidia) and [NVIDIA NGC](https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC). Explore comprehensive [tutorials](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/starthere/tutorials.html) on [Google Colab](https://colab.research.google.com) or with the [NGC NeMo Framework Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo).
For advanced users,  [example scripts](https://github.com/NVIDIA/NeMo/tree/main/examples) support multi-GPU/multi-node training.

## Requirements

*   Python 3.10 or above
*   PyTorch 2.5 or above
*   NVIDIA GPU (for model training)

## Developer Documentation

See the documentation for the latest and stable versions:

| Version | Documentation                                                                                                          |
| ------- | ---------------------------------------------------------------------------------------------------------------------- |
| Latest  | [Latest Branch Documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)                    |
| Stable  | [Stable Release Documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/)                 |

## Installation

Choose the installation method based on your needs:

*   **Conda / Pip:** Use for ASR and TTS domains, and to explore NeMo on supported platforms.
*   **NGC PyTorch container:** Install from source within a highly optimized container.
*   **NGC NeMo container:** Ready-to-go solution for optimal performance.

### Support matrix

*   Fully supported: Max performance and feature-completeness.
*   Limited supported: Used to explore NeMo.
*   No support yet: In development.
*   Deprecated: Support has reached end of life.

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
pip install "nemo_toolkit[all]" # or pip install "nemo_toolkit[all]@git+https://github.com/NVIDIA/NeMo@${REF:-'main'}"
```

### NGC PyTorch container

**(Requires 24.04 / NeMo-Toolkit 2.3.0 or later)**

1.  Launch a base NVIDIA PyTorch container:  `nvcr.io/nvidia/pytorch:25.01-py3`
2.  Inside the container:

    ```bash
    cd /opt
    git clone https://github.com/NVIDIA/NeMo
    cd NeMo
    git checkout ${REF:-'main'}
    bash docker/common/install_dep.sh --library all
    pip install ".[all]"
    ```

### NGC NeMo container

Use the latest consolidated Docker container:
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

Find answers and engage in discussions on the NeMo [Discussions board](https://github.com/NVIDIA/NeMo/discussions).

## Contribute

Contributions are welcome! Refer to [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md) for details.

## Publications

Explore the growing list of [publications](https://nvidia.github.io/NeMo/publications/) using the NeMo Framework.  Submit a pull request to the `gh-pages-src` branch to contribute an article.

## Blogs

**(Latest Blog Posts - Summarized)**

*   **Bria Builds Responsible Generative AI:** Leveraging NeMo and Picasso for enterprise visual AI.
*   **NVIDIA H200 and NeMo:** Enhancements include FSDP, MoE-based LLMs, RLHF, and speedups on H200 GPUs.
*   **Amazon Titan Foundation Models:** NeMo powers the training of Amazon Titan foundation models.

## Licenses

NeMo is licensed under the [Apache License 2.0](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file).