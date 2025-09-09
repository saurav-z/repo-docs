[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# NVIDIA NeMo: Build, Customize, and Deploy State-of-the-Art Generative AI Models

NVIDIA NeMo is a cloud-native, open-source framework designed for researchers and developers to efficiently build, customize, and deploy large language models (LLMs), multimodal models (MMs), automatic speech recognition (ASR), text-to-speech (TTS), and computer vision (CV) models.  [Explore the original repository](https://github.com/NVIDIA/NeMo).

## Key Features

*   **Comprehensive Support:**  Build and customize LLMs, MMs, ASR, TTS, and CV models.
*   **Scalable Training:** Train models efficiently on thousands of GPUs with Lightning integration and advanced parallelism techniques.
*   **Pre-trained Models:** Leverage pre-trained models available on Hugging Face Hub and NVIDIA NGC.
*   **Model Optimization:** Deploy and optimize models with NVIDIA Riva for ASR/TTS and NeMo Microservices.
*   **Parameter-Efficient Fine-tuning:** Utilize techniques like LoRA and adapters for efficient model customization.
*   **Extensive Tutorials & Examples:** Get started quickly with tutorials on Google Colab and NGC containers, along with example scripts for multi-GPU/multi-node training.
*   **NeMo 2.0:** Enhanced modularity, Python-based configuration, and streamlined experimentation.
*   **Cosmos Integration:** Support for training and customizing NVIDIA Cosmos world foundation models.
*   **Latest Model Support:** Llama 4, Flux, Llama Nemotron, Hyena & Evo2, Qwen2-VL, Qwen2.5, Gemma3, Qwen3-30B&32B
*   **Hugging Face Models Support**: Pretrain and finetune Hugging Face Models via AutoModel

## What's New in NeMo 2.0

NVIDIA NeMo 2.0 introduces significant enhancements for improved flexibility, performance, and scalability:

*   **Python-Based Configuration:** Enjoy greater flexibility and control with Python-based configurations.
*   **Modular Abstractions:** Benefit from PyTorch Lightning's modular design for easier experimentation and modification.
*   **Scalability:** Seamlessly scale large-scale experiments using NeMo-Run.

## Getting Started

*   **Quickstart:**  Explore examples of using NeMo-Run to launch experiments locally and on a Slurm cluster.
*   **User Guide:** Dive into the [NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/index.html).
*   **Recipes:** Access [NeMo 2.0 Recipes](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/recipes) for large-scale run examples.
*   **Feature Guide:** Discover key features in the [Feature Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/features/index.html#feature-guide).
*   **Migration Guide:**  Transition from NeMo 1.0 to 2.0 with the [Migration Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/migration/index.html#migration-guide).
*   **Cosmos:** Explore [NVIDIA Cosmos World Foundation Models](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/cosmos/collections/cosmos) on NGC and Hugging Face, with information on video datasets, [NeMo Curator](https://developer.nvidia.com/nemo-curator), and post-training instructions using the NeMo Framework.

## Training and Customization

*   Utilizes [Lightning](https://github.com/Lightning-AI/lightning) for scalable training.
*   Employs parallelism strategies like Tensor Parallelism (TP), Pipeline Parallelism (PP), Fully Sharded Data Parallelism (FSDP), Mixture-of-Experts (MoE), and mixed precision (BFloat16 and FP8) for efficient training.
*   Leverages [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine) for FP8 training on Hopper GPUs and [NVIDIA Megatron Core](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core) for scaling transformer model training.
*   Supports alignment methods like SteerLM, Direct Preference Optimization (DPO), and Reinforcement Learning from Human Feedback (RLHF).  See [NVIDIA NeMo Aligner](https://github.com/NVIDIA/NeMo-Aligner).
*   Offers parameter-efficient fine-tuning (PEFT) techniques such as LoRA, P-Tuning, Adapters, and IA3.

## Deployment and Optimization

*   Deploy and optimize LLMs and MMs with [NVIDIA NeMo Microservices](https://developer.nvidia.com/nemo-microservices-early-access).
*   Optimize ASR and TTS models for inference and production using [NVIDIA Riva](https://developer.nvidia.com/riva).

## Installation

Choose the installation method that best suits your needs:

*   **Conda / Pip:**  For exploration on supported platforms. Recommended for ASR and TTS.
*   **NGC PyTorch Container:** Install from source in a highly optimized container.
*   **NGC NeMo Container:** Ready-to-go solution for optimal performance.

### Support Matrix

Fully supported: Max performance and feature-completeness.

Limited supported: Used to explore NeMo.

No support yet: In development.

Deprecated: Support has reached end of life.

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

### NGC PyTorch Container

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

### NGC NeMo Container

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

*   [Discussions Board](https://github.com/NVIDIA/NeMo/discussions): Ask questions and join discussions.
*   [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md): Contribute to the project.
*   [Publications](https://nvidia.github.io/NeMo/publications/): Explore publications utilizing NeMo.

## License

NVIDIA NeMo is licensed under the [Apache License 2.0](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file).