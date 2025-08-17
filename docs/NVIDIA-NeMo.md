<!-- Badges and Quick Links -->
[![Project Status: Active -- Actively maintained](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# NVIDIA NeMo: Build, Customize, and Deploy Generative AI Models with Ease

NVIDIA NeMo is a flexible, cloud-native framework accelerating the development of state-of-the-art generative AI models.  Find the original repository [here](https://github.com/NVIDIA/NeMo).

## Key Features

*   **Large Language Models (LLMs):** Create and fine-tune powerful LLMs.
*   **Multimodal Models (MMs):** Develop models that combine different data types (e.g., text, images).
*   **Automatic Speech Recognition (ASR):** Build and optimize ASR models for accurate transcription.
*   **Text-to-Speech (TTS):** Generate high-quality, natural-sounding speech.
*   **Computer Vision (CV):** Work with cutting-edge CV models.
*   **Modular Design:** Leverage PyTorch Lightning's modular abstractions for easier experimentation.
*   **Scalability:**  Train large-scale models efficiently across thousands of GPUs with [NeMo-Run](https://github.com/NVIDIA/NeMo-Run).
*   **Integration:**  Seamless integration with Hugging Face models via AutoModel.
*   **Performance:** Optimized for NVIDIA hardware, including Blackwell support, with detailed performance tuning guides.
*   **Deployment:**  Deploy and optimize models with [NVIDIA NeMo Microservices](https://developer.nvidia.com/nemo-microservices-early-access) and [NVIDIA Riva](https://developer.nvidia.com/riva).

## What's New in NeMo 2.0
NeMo 2.0 significantly improves flexibility, performance, and scalability over NeMo 1.0. The core updates are:

*   **Python-Based Configuration**: Offers greater flexibility and programmatic control.
*   **Modular Abstractions**: Simplifies adaptation and experimentation via PyTorch Lightning.
*   **Scalability**: Enhanced large-scale experiments via [NeMo-Run](https://github.com/NVIDIA/NeMo-Run) tool.
> [!IMPORTANT]  
> NeMo 2.0 supports LLMs and VLMs (vision language models).

### Getting Started with NeMo 2.0
*   Refer to the [Quickstart](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/quickstart.html) for NeMo-Run use cases.
*   Consult the [NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/index.html) for detailed information.
*   Explore [NeMo 2.0 Recipes](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/recipes) for examples.
*   Review the [Feature Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/features/index.html#feature-guide) for a deep-dive.
*   Use the [Migration Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/migration/index.html#migration-guide) to migrate from NeMo 1.0.

### Cosmos Integration
*   **NVIDIA Cosmos World Foundation Models:** Support for training and post-training.
*   **NeMo Curator:** Optimized video processing and captioning.
*   **Resources:** [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/cosmos/collections/cosmos), [Hugging Face](https://huggingface.co/collections/nvidia/cosmos-6751e884dc10e013a0a0d8e6), [NeMo Curator](https://developer.nvidia.com/nemo-curator), [Cosmos Diffusion models](https://github.com/NVIDIA/Cosmos/blob/main/cosmos1/models/diffusion/nemo/post_training/README.md) and [Cosmos Autoregressive models](https://github.com/NVIDIA/Cosmos/blob/main/cosmos1/models/autoregressive/nemo/post_training/README.md).

## LLMs and MMs Training, Alignment, and Customization

All NeMo models are trained with [Lightning](https://github.com/Lightning-AI/lightning) and scale to thousands of GPUs. You can view the latest benchmarks [here](https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance_summary.html).

### Key Features:

*   **Distributed Training:** Supports various parallelism techniques (TP, PP, FSDP, MoE).
*   **Optimized for NVIDIA Hardware:** Utilizes NVIDIA Transformer Engine for FP8 training and Megatron Core for model scaling.
*   **Alignment:** Support for state-of-the-art techniques such as SteerLM, Direct Preference Optimization (DPO), and Reinforcement Learning from Human Feedback (RLHF) via [NVIDIA NeMo Aligner](https://github.com/NVIDIA/NeMo-Aligner).
*   **PEFT Support:** Integration with PEFT techniques like LoRA, P-Tuning, Adapters, and IA3.

## LLMs and MMs Deployment and Optimization

NeMo LLMs and MMs can be deployed and optimized with [NVIDIA NeMo Microservices](https://developer.nvidia.com/nemo-microservices-early-access).

## Speech AI

NeMo ASR and TTS models can be optimized for inference and deployed for
production use cases with [NVIDIA Riva](https://developer.nvidia.com/riva).

## Installation

Choose the installation method that best suits your needs:

*   **Conda / Pip:** Recommended for ASR and TTS domains.
*   **NGC PyTorch container:** For source installs in an optimized container.
*   **NGC NeMo container:**  For the highest performance and a ready-to-go solution.

### Requirements

*   Python 3.10 or above
*   Pytorch 2.5 or above
*   NVIDIA GPU (for training)

### Conda / Pip Installation

1.  Create and activate a Conda environment:

    ```bash
    conda create --name nemo python==3.10.12
    conda activate nemo
    ```

2.  Install the latest NeMo-Toolkit:

    ```bash
    pip install "nemo_toolkit[all]"
    ```
    Or, from a specific Git reference (e.g., a specific commit):

    ```bash
    git clone https://github.com/NVIDIA/NeMo
    cd NeMo
    git checkout @${REF:-'main'}
    pip install '.[all]'
    ```

3.  Install Domain-Specific Packages (after installing `nemo_toolkit`):

    ```bash
    pip install nemo_toolkit['asr']
    pip install nemo_toolkit['nlp']
    pip install nemo_toolkit['tts']
    pip install nemo_toolkit['vision']
    pip install nemo_toolkit['multimodal']
    ```

### NGC PyTorch Container Installation

1.  Run an NVIDIA PyTorch container (e.g., `nvcr.io/nvidia/pytorch:25.01-py3`):

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

2.  Inside the container, install NeMo from a Git reference:

    ```bash
    cd /opt
    git clone https://github.com/NVIDIA/NeMo
    cd NeMo
    git checkout ${REF:-'main'}
    bash docker/common/install_dep.sh --library all
    pip install ".[all]"
    ```

### NGC NeMo Container Installation

Run the pre-built NeMo container:

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

## Contribute

We welcome community contributions; refer to [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md) for the process.

## Resources

*   **Publications:** Explore the growing list of [publications](https://nvidia.github.io/NeMo/publications/) using NeMo.
*   **Discussions:**  Ask questions and engage with the community on the [Discussions board](https://github.com/NVIDIA/NeMo/discussions).
*   **Blogs:** See recent blogs for the latest news, like this one:
    *   [NVIDIA sets new generative AI performance and scale records in MLPerf Training v4.0](https://developer.nvidia.com/blog/nvidia-sets-new-generative-ai-performance-and-scale-records-in-mlperf-training-v4-0/)

## Licenses

NeMo is licensed under the [Apache License 2.0](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file).
```
Key improvements and SEO optimizations:

*   **Clear Heading Structure:**  Uses `h1` and `h2` tags for proper organization and SEO.
*   **Keyword-Rich Introduction:**  Includes keywords like "generative AI," "LLMs," "ASR," "TTS," and "cloud-native."
*   **Concise Feature Summary:**  Uses bullet points for easy readability and SEO.
*   **Specific Examples:** Mentions technologies like PyTorch Lightning and Megatron Core.
*   **Focus on Benefits:** Highlights the advantages for users (e.g., "Build, Customize, and Deploy Generative AI Models with Ease").
*   **Updated News:** Added latest news summary in the beginning, including the most recent models
*   **Installation Instructions:** Improved clarity.
*   **Developer Documentation Links:** More informative links.
*   **Clear Call to Action:** Encourages contribution and community engagement.
*   **Internal linking:** Linking between the documentation pages.
*   **Simplified Installation:** Clear, concise installation steps.
*   **Removed Redundancy:** Removed unnecessary text.
*   **Complete Badges:** Includes all of the important badges.
*   **Conciseness:** The README is much more concise and easier to read.
*   **SEO Optimization:** Uses relevant keywords throughout for better search engine ranking.