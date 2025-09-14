[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# NVIDIA NeMo: Build, Customize, and Deploy Generative AI Models with Ease

NVIDIA NeMo is a cloud-native framework, empowering researchers and developers to build, customize, and deploy cutting-edge generative AI models for LLMs, MMs, ASR, TTS, and CV, and more.  Learn more on the original [NVIDIA NeMo Repository](https://github.com/NVIDIA/NeMo).

## Key Features

*   **Modular Architecture:** Flexible and easy-to-use framework for designing and experimenting with AI models.
*   **Pre-trained Models & Checkpoints:** Leverage existing models and accelerate your projects.
*   **Scalable Training:** Seamlessly scale training across thousands of GPUs with NeMo-Run.
*   **Support for Various AI Domains:** Focused on Large Language Models (LLMs), Multimodal Models (MMs), Automatic Speech Recognition (ASR), Text-to-Speech (TTS), and Computer Vision (CV).
*   **Optimized for NVIDIA Hardware:** Maximizes performance on NVIDIA GPUs with techniques like Tensor Parallelism, Pipeline Parallelism, and FP8 training.
*   **Integration with NVIDIA Ecosystem:** Deploy and optimize models with NVIDIA Riva and NeMo Microservices.
*   **Parameter-Efficient Fine-tuning (PEFT):** Supports PEFT techniques like LoRA and Adapters.

## What's New in NeMo 2.0

NVIDIA NeMo 2.0 introduces significant improvements for enhanced flexibility, performance, and scalability.

*   **Python-Based Configuration:** Easier configuration and programmatic control.
*   **Modular Abstractions:** Simplifying experimentation with PyTorch Lightning.
*   **Scalability:** Efficiently scale experiments using NeMo-Run for large-scale runs.

## Getting Started

*   Refer to the [Quickstart](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/quickstart.html) to learn how to launch NeMo 2.0 experiments locally.
*   Explore the [NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/index.html).
*   Find examples of large-scale runs using NeMo 2.0 and NeMo-Run in the [NeMo 2.0 Recipes](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/recipes).
*   Explore main features of NeMo 2.0 through the [Feature Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/features/index.html#feature-guide).
*   Migrate from NeMo 1.0 to 2.0 with the [Migration Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/migration/index.html#migration-guide) .

## Cosmos

*   Support video curation and post-training of the [Cosmos World Foundation Models](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/cosmos/collections/cosmos)
*   Learn more about video datasets at [NeMo Curator](https://developer.nvidia.com/nemo-curator).
*   Post-train World Foundation Models for your custom physical AI tasks using the [Cosmos Diffusion models](https://github.com/NVIDIA/Cosmos/blob/main/cosmos1/models/diffusion/nemo/post_training/README.md) and the [Cosmos Autoregressive models](https://github.com/NVIDIA/Cosmos/blob/main/cosmos1/models/autoregressive/nemo/post_training/README.md).

## Training, Alignment, and Customization

*   All NeMo models are trained with [Lightning](https://github.com/Lightning-AI/lightning) and automatically scalable to 1000s of GPUs.
*   Find the performance benchmarks using the latest NeMo Framework container [here](https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance_summary.html).
*   Leverage cutting-edge distributed training techniques, incorporating parallelism strategies.
*   Utilize NVIDIA Transformer Engine and NVIDIA Megatron Core for efficient training.
*   NeMo LLMs can be aligned with state-of-the-art methods.
*   Supports the latest parameter efficient fine-tuning (PEFT) techniques such as LoRA, P-Tuning, Adapters, and IA3.

## Deployment and Optimization

*   Deploy and optimize NeMo LLMs and MMs with [NVIDIA NeMo Microservices](https://developer.nvidia.com/nemo-microservices-early-access).
*   Optimize NeMo ASR and TTS models for inference and deploy with [NVIDIA Riva](https://developer.nvidia.com/riva).

## Installation

Choose the method that best suits your needs:

*   **Conda / Pip:** Explore NeMo on any supported platform, recommended for ASR and TTS.
*   **NGC PyTorch Container:** Install from source with feature-completeness.
*   **NGC NeMo Container:** Ready-to-go solution for highest performance.

### Supported Platforms

*   linux - amd64/x84_64: Full support
*   linux - arm64: Limited support
*   darwin - amd64/x64_64: Deprecated
*   darwin - arm64: Limited support

### Conda / Pip Installation

1.  Create a Conda environment:

```bash
conda create --name nemo python==3.10.12
conda activate nemo
```

2.  Pick the right version
    *   Install from pre-built wheel

```bash
pip install "nemo_toolkit[all]"
```

    *   Install from a Git reference

```bash
git clone https://github.com/NVIDIA/NeMo
cd NeMo
git checkout @${REF:-'main'}
pip install '.[all]'
```

3.  Install a specific domain:

```bash
pip install nemo_toolkit['all']
pip install nemo_toolkit['asr']
pip install nemo_toolkit['nlp']
pip install nemo_toolkit['tts']
pip install nemo_toolkit['vision']
pip install nemo_toolkit['multimodal']
```

### NGC PyTorch Container

1.  Launch the container:

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

2.  Install nemo_toolkit:

```bash
cd /opt
git clone https://github.com/NVIDIA/NeMo
cd NeMo
git checkout ${REF:-'main'}
bash docker/common/install_dep.sh --library all
pip install ".[all]"
```

### NGC NeMo Container

Run the following to use a pre-built container:

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

## Requirements

*   Python 3.10 or above
*   PyTorch 2.5 or above
*   NVIDIA GPU (for training)

## Resources

*   **Documentation:** [NeMo Framework User Guide](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
*   **Tutorials:** Extensive [tutorials](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/starthere/tutorials.html)
*   **Example Scripts:** [Example Scripts](https://github.com/NVIDIA/NeMo/tree/main/examples)
*   **Discussions:** [Discussions Board](https://github.com/NVIDIA/NeMo/discussions)
*   **Publications:** [Publications](https://nvidia.github.io/NeMo/publications/)

## Contribute

We welcome community contributions! Refer to [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md) for the process.