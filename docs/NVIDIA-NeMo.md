[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# NVIDIA NeMo: Your Gateway to Cutting-Edge Generative AI

**NVIDIA NeMo is a versatile and scalable framework for building, customizing, and deploying state-of-the-art generative AI models across various domains.** ([See the original repo](https://github.com/NVIDIA/NeMo))

## Key Features

*   **Large Language Models (LLMs):** Efficiently train and fine-tune powerful language models.
*   **Multimodal Models (MMs):** Develop AI models that process and generate data from multiple sources, such as text and images.
*   **Automatic Speech Recognition (ASR):** Build accurate and high-performing speech recognition systems.
*   **Text-to-Speech (TTS):** Create natural-sounding speech synthesis applications.
*   **Computer Vision (CV):** Develop and deploy computer vision models for various tasks.
*   **Model Customization and Alignment**: Use cutting edge methods such as SteerLM, Direct Preference Optimization (DPO), and Reinforcement Learning from Human Feedback (RLHF) for LLMs and MMs.
*   **Deployment and Optimization**: Deploy LLMs and MMs with NVIDIA NeMo Microservices and optimize ASR and TTS models with NVIDIA Riva.

## Latest Updates
*   **Hugging Face Integration:** Seamlessly pretrain and fine-tune Hugging Face models.
*   **Blackwell Support:** Benefit from enhanced performance benchmarks on GB200 & B200.
*   **Performance Tuning Guide:** Optimize throughput with a new performance tuning guide.
*   **New Model Support:** Access and utilize the latest community models, including Llama 4, Flux, Llama Nemotron, Hyena & Evo2, Qwen2-VL, Qwen2.5, Gemma3, Qwen3-30B&32B.

## What's New in NeMo 2.0

NeMo 2.0 introduces several improvements, including:

*   **Python-Based Configuration:** Offers more flexibility and control in model configurations.
*   **Modular Abstractions:** Simplifies model adaptation and experimentation through PyTorch Lightning's modular approach.
*   **Scalability:** Streamlines large-scale experiments across thousands of GPUs with NeMo-Run.

>   **Note:** NeMo 2.0 primarily supports LLM (large language model) and VLM (vision language model) collections.

### Get Started with NeMo 2.0
*   Refer to the [Quickstart](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/quickstart.html) for examples of using NeMo-Run to launch NeMo 2.0 experiments.
*   For more information about NeMo 2.0, see the [NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/index.html).
*   [NeMo 2.0 Recipes](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/recipes) contains additional examples of launching large-scale runs using NeMo 2.0 and NeMo-Run.
*   For an in-depth exploration of the main features of NeMo 2.0, see the [Feature Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/features/index.html#feature-guide).
*   To transition from NeMo 1.0 to 2.0, see the [Migration Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/migration/index.html#migration-guide) for step-by-step instructions.

### Get Started with Cosmos
The NeMo Framework and NeMo Curator support the post-training and video curation of the Cosmos World Foundation Models. More information on video datasets can be found at [NeMo Curator](https://developer.nvidia.com/nemo-curator). To post-train World Foundation Models for custom physical AI tasks, see the [Cosmos Diffusion models](https://github.com/NVIDIA/Cosmos/blob/main/cosmos1/models/diffusion/nemo/post_training/README.md) and the [Cosmos Autoregressive models](https://github.com/NVIDIA/Cosmos/blob/main/cosmos1/models/autoregressive/nemo/post_training/README.md).

## LLMs and MMs Training, Alignment, and Customization

*   **Scalable Training:** Train models across 1000s of GPUs.
*   **Advanced Parallelism:** Utilize techniques like Tensor Parallelism, Pipeline Parallelism, FSDP, MoE, and Mixed Precision Training.
*   **NVIDIA Technologies:** Leverages NVIDIA Transformer Engine and Megatron Core for FP8 training and scaling.
*   **Alignment:** Supports cutting-edge alignment methods such as SteerLM, Direct Preference Optimization (DPO), and Reinforcement Learning from Human Feedback (RLHF).
*   **Parameter Efficient Fine-tuning (PEFT):**  Supports LoRA, P-Tuning, Adapters, and IA3.

## LLMs and MMs Deployment and Optimization

*   **Deployment:** Deploy and optimize models with NVIDIA NeMo Microservices.

## Speech AI

*   **Optimization:** Optimize ASR and TTS models for inference.
*   **Deployment:** Deploy for production use cases with NVIDIA Riva.

## Get Started with NeMo Framework

*   **Pretrained Models:** Access state-of-the-art models on [Hugging Face Hub](https://huggingface.co/models?library=nemo&sort=downloads&search=nvidia) and [NVIDIA NGC](https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC).
*   **Tutorials:** Run tutorials on [Google Colab](https://colab.research.google.com) or with the [NGC NeMo Framework Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo).
*   **Playbooks:** Use playbooks for training models with the NeMo Framework Launcher.
*   **Example Scripts:** Utilize example scripts for multi-GPU/multi-node training.

## Requirements

*   Python 3.10 or above
*   PyTorch 2.5 or above
*   NVIDIA GPU (for training)

## Installation

Choose the installation method that best fits your needs:

*   **Conda / Pip:** Ideal for exploring NeMo.  Recommended for ASR and TTS.
*   **NGC PyTorch container:** Install NeMo from source within an optimized container.
*   **NGC NeMo container:** A ready-to-go solution for optimal performance.

### Support Matrix

NeMo-Framework provides tiers of support based on OS / Platform and mode of installation. Please refer the following overview of support levels:

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

#### Conda / Pip
```bash
conda create --name nemo python==3.10.12
conda activate nemo
pip install "nemo_toolkit[all]"
```

#### NGC PyTorch container
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

## Developer Documentation

See the table for documentation information:

| Version | Status                                                                                                                                                              | Description                                                                                                                    |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Latest  | [![Documentation Status](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)     | [Documentation of the latest (i.e. main) branch.](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)          |
| Stable  | [![Documentation Status](https://readthedocs.com/projects/nvidia-nemo/badge/?version=stable)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/) | [Documentation of the stable (i.e. most recent release)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/) |

## Contribute

We welcome community contributions; see [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md) for details.

## Discussions Board

Find answers and engage in discussions on the NeMo [Discussions board](https://github.com/NVIDIA/NeMo/discussions).

## Publications

Explore the [publications](https://nvidia.github.io/NeMo/publications/) that utilize the NeMo Framework.

## License

NeMo is licensed under the [Apache License 2.0](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file).