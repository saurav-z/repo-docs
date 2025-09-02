[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# NVIDIA NeMo: Unleash the Power of Generative AI with Ease

NVIDIA NeMo is a versatile, cloud-native framework empowering researchers and developers to build and deploy cutting-edge generative AI models for LLMs, MMs, ASR, TTS, and CV.  [Explore the original repository](https://github.com/NVIDIA/NeMo) for more details.

## Key Features

*   **Large Language Models (LLMs):**  Build, customize, and deploy powerful language models.
*   **Multimodal Models (MMs):**  Develop models that understand and generate content across multiple modalities (text, images, etc.).
*   **Automatic Speech Recognition (ASR):**  Create accurate and efficient speech-to-text models.
*   **Text-to-Speech (TTS):**  Generate realistic and natural-sounding speech from text.
*   **Computer Vision (CV):**  Implement advanced computer vision models.

## What's New

*   **Hugging Face Models via AutoModel:**  Easily integrate and fine-tune Hugging Face models.
*   **Blackwell Support:** Enhanced performance benchmarks on GB200 & B200.
*   **Training Performance Guide:**  Optimize your model training with the comprehensive performance tuning guide.
*   **Latest Model Support:**  Access the latest community models like Llama 4, Flux, Qwen2, and more.
*   **NeMo Framework 2.0:**  A modular and user-friendly framework prioritizing ease of use.
*   **Cosmos World Foundation Models Support:**  Train and customize video foundation models.
*   **New LLM and Multimodal Model Support:**  Includes support for Llama 3.1, state-of-the-art advancements.

## Introduction

NVIDIA NeMo (Neural Modules) is a comprehensive framework designed for researchers and developers working with generative AI.  It simplifies the process of creating, customizing, and deploying state-of-the-art models by leveraging existing code and pre-trained checkpoints. The framework is optimized for LLMs, MMs, ASR, TTS, and CV domains.

For in-depth technical details, consult the [NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html).

## NeMo 2.0: Enhanced Flexibility and Performance

NeMo 2.0 introduces significant improvements over its predecessor:

*   **Python-Based Configuration:**  Offers greater flexibility and programmatic control.
*   **Modular Abstractions:**  Simplifies adaptation and experimentation with PyTorch Lightning.
*   **Scalability:**  Easily scale experiments across thousands of GPUs using [NeMo-Run](https://github.com/NVIDIA/NeMo-Run).

> [!IMPORTANT]  
> NeMo 2.0 supports LLM (large language model) and VLM (vision language model) collections.

### Get Started with NeMo 2.0

*   [Quickstart](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/quickstart.html) - Launch experiments with NeMo-Run.
*   [NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/index.html) - Comprehensive documentation.
*   [NeMo 2.0 Recipes](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/recipes) - Large-scale run examples.
*   [Feature Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/features/index.html#feature-guide) - In-depth feature exploration.
*   [Migration Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/migration/index.html#migration-guide) - Transition from NeMo 1.0.

### Get Started with Cosmos

For more information on video datasets, refer to [NeMo Curator](https://developer.nvidia.com/nemo-curator). To post-train World Foundation Models using the NeMo Framework for your custom physical AI tasks, see the [Cosmos Diffusion models](https://github.com/NVIDIA/Cosmos/blob/main/cosmos1/models/diffusion/nemo/post_training/README.md) and the [Cosmos Autoregressive models](https://github.com/NVIDIA/Cosmos/blob/main/cosmos1/models/autoregressive/nemo/post_training/README.md).

## Training, Alignment, and Customization of LLMs and MMs

NeMo models are trained with [Lightning](https://github.com/Lightning-AI/lightning) and automatically scale to thousands of GPUs. Performance benchmarks are available [here](https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance_summary.html).

Key training techniques:

*   **Distributed Training:**  Utilizes Tensor Parallelism (TP), Pipeline Parallelism (PP), Fully Sharded Data Parallelism (FSDP), Mixture-of-Experts (MoE), and mixed precision training.
*   **NVIDIA Transformer Engine:**  Employs FP8 training on NVIDIA Hopper GPUs.
*   **NVIDIA Megatron Core:**  Scales Transformer model training.

NeMo LLMs support alignment methods like SteerLM, DPO, and RLHF. See [NVIDIA NeMo Aligner](https://github.com/NVIDIA/NeMo-Aligner).  Also supports PEFT methods such as LoRA, P-Tuning, Adapters, and IA3. See [NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/sft_peft/index.html).

## Deployment and Optimization of LLMs and MMs

Deploy and optimize NeMo LLMs and MMs using [NVIDIA NeMo Microservices](https://developer.nvidia.com/nemo-microservices-early-access).

## Speech AI

Optimize and deploy NeMo ASR and TTS models with [NVIDIA Riva](https://developer.nvidia.com/riva).

## NeMo Framework Launcher (for NeMo 1.0 - NeMo 2.0 uses NeMo-Run)

[NeMo Framework Launcher](https://github.com/NVIDIA/NeMo-Megatron-Launcher) streamlines NeMo Framework training on CSPs and Slurm clusters. It includes recipes, scripts, and the [Autoconfigurator](https://github.com/NVIDIA/NeMo-Megatron-Launcher#53-using-autoconfigurator-to-find-the-optimal-configuration).  Use [NeMo-Run](https://github.com/NVIDIA/NeMo-Run) for NeMo 2.0 experiments.

## Getting Started

Pre-trained NeMo models are available on [Hugging Face Hub](https://huggingface.co/models?library=nemo&sort=downloads&search=nvidia) and [NVIDIA NGC](https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC). Use tutorials on [Google Colab](https://colab.research.google.com) or the [NGC NeMo Framework Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo) to get started. Explore [example scripts](https://github.com/NVIDIA/NeMo/tree/main/examples) for advanced training.

## Requirements

*   Python 3.10 or above
*   Pytorch 2.5 or above
*   NVIDIA GPU (for training)

## Developer Documentation

See the tables below for details:

| Version | Status                                                                                                                                                              | Description                                                                                                                    |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Latest  | [![Documentation Status](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)     | [Documentation of the latest (i.e. main) branch.](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)          |
| Stable  | [![Documentation Status](https://readthedocs.com/projects/nvidia-nemo/badge/?version=stable)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/) | [Documentation of the stable (i.e. most recent release)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/) |

## Installation

Choose an installation method based on your needs:

*   **Conda / Pip:**  For exploring NeMo, recommended for ASR and TTS.
*   **NGC PyTorch container:** For source installations and optimization.
*   **NGC NeMo container:** Ready-to-go solution for maximum performance.

### Support Matrix

| OS / Platform              | Install from PyPi | Source into NGC container |
|----------------------------|-------------------|---------------------------|
| `linux` - `amd64/x84_64`   | Limited support   | Full support              |
| `linux` - `arm64`          | Limited support   | Limited support           |
| `darwin` - `amd64/x64_64`  | Deprecated        | Deprecated                |
| `darwin` - `arm64`         | Limited support   | Limited support           |
| `windows` - `amd64/x64_64` | No support yet    | No support yet            |
| `windows` - `arm64`        | No support yet    | No support yet            |

### Conda / Pip

1.  Create a Conda environment:

```bash
conda create --name nemo python==3.10.12
conda activate nemo
```

2.  Install the nemo_toolkit:

*   **Using pre-built wheels:**
    ```bash
    pip install "nemo_toolkit[all]"
    ```
*   **Using a specific Git reference:**
    ```bash
    git clone https://github.com/NVIDIA/NeMo
    cd NeMo
    git checkout @${REF:-'main'}
    pip install '.[all]'
    ```
3.  Install a specific domain:

```bash
pip install nemo_toolkit['all'] # or pip install "nemo_toolkit['all']@git+https://github.com/NVIDIA/NeMo@${REF:-'main'}"
pip install nemo_toolkit['asr'] # or pip install "nemo_toolkit['asr']@git+https://github.com/NVIDIA/NeMo@$REF:-'main'}"
pip install nemo_toolkit['nlp'] # or pip install "nemo_toolkit['nlp']@git+https://github.com/NVIDIA/NeMo@${REF:-'main'}"
pip install nemo_toolkit['tts'] # or pip install "nemo_toolkit['tts']@git+https://github.com/NVIDIA/NeMo@${REF:-'main'}"
pip install nemo_toolkit['vision'] # or pip install "nemo_toolkit['vision']@git+https://github.com/NVIDIA/NeMo@${REF:-'main'}"
pip install nemo_toolkit['multimodal'] # or pip install "nemo_toolkit['multimodal']@git+https://github.com/NVIDIA/NeMo@${REF:-'main'}"
```

### NGC PyTorch container (Supported from 24.04, NeMo-Toolkit 2.3.0)

1.  Launch the base PyTorch container:

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

## NGC NeMo container

Use pre-built containers:

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

## Discussions

Find answers and start discussions on the NeMo [Discussions board](https://github.com/NVIDIA/NeMo/discussions).

## Contribute

Contribute to NeMo! See [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md).

## Publications

Explore [publications](https://nvidia.github.io/NeMo/publications/) that use NeMo. Contribute articles by submitting a pull request to the `gh-pages-src` branch.

## Blogs

**(Details condensed, refer to original readme for full descriptions)**

*   **Large Language Models and Multimodal Models**
    *   Bria Builds Responsible Generative AI for Enterprises Using NVIDIA NeMo, Picasso (2024/03/06)
    *   New NVIDIA NeMo Framework Features and NVIDIA H200 (2023/12/06)
    *   NVIDIA now powers training for Amazon Titan Foundation models (2023/11/28)

## Licenses

Licensed under the [Apache License 2.0](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file).
```
Key improvements and optimizations:

*   **SEO Optimization:**  Uses keywords like "NVIDIA," "NeMo," "Generative AI," "LLMs," "ASR," "TTS," and "Computer Vision" throughout the headings and content.
*   **Clear Structure:**  Organizes the information with headings, subheadings, bullet points, and concise paragraphs for readability.
*   **Concise Language:**  Streamlines the text while retaining essential information.
*   **Focus on Benefits:** Highlights the advantages of using NeMo (ease of use, scalability, pre-trained models, etc.).
*   **Up-to-date Information:** Includes latest updates related to Hugging Face and other significant features.
*   **Actionable "Get Started" Sections:** Includes clear instructions and links to resources.
*   **Emphasis on Key Features:** The "Key Features" section concisely lists the core functionalities.
*   **Improved Installation Instructions:** The installation section is organized, and provides command examples.
*   **Comprehensive and Easy-to-Read Documentation:** Included detailed links, tables and text describing all the major features.