[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# NVIDIA NeMo: Build, Customize, and Deploy Generative AI Models with Ease

NVIDIA NeMo is a cloud-native framework empowering researchers and developers to create, customize, and deploy cutting-edge generative AI models for various applications.  For more details, visit the [original repository](https://github.com/NVIDIA/NeMo).

## Key Features

*   **Large Language Models (LLMs):** Train and customize state-of-the-art LLMs.
*   **Multimodal Models (MMs):** Develop models that combine different data types (e.g., text and images).
*   **Automatic Speech Recognition (ASR):** Build and optimize high-accuracy speech recognition systems.
*   **Text-to-Speech (TTS):** Create realistic and expressive speech synthesis models.
*   **Computer Vision (CV):** Explore and implement advanced computer vision tasks.

## Latest Updates & News

*   **[Date]**: New Model Support: Llama 4, Flux, Llama Nemotron, Hyena & Evo2, Qwen2-VL, Qwen2.5, Gemma3, Qwen3-30B&32B.
*   **[Date]**: NeMo Framework 2.0 Release: Prioritizing modularity and ease-of-use.
*   **[Date]**: New support for Hugging Face models via AutoModel.
*   **[Date]**: Training Performance on GPU Tuning Guide.
*   **[Date]**:  NVIDIA Cosmos World Foundation Model Platform support for physical AI.
*   **[Date]**:  NVIDIA NeMo Accelerates LLM Innovation with Hybrid State Space Model Support.
*   **[Date]**:  NVIDIA sets new generative AI performance and scale records in MLPerf Training v4.0.
*   **[Date]**:  NVIDIA releases 340B base, instruct, and reward models pretrained on a total of 9T tokens.

## Getting Started

### Installation

Choose the installation method that best fits your needs:

*   **Conda / Pip:**  Ideal for exploring NeMo and is recommended for ASR and TTS domains, offering limited feature completeness for other domains.
*   **NGC PyTorch Container:** Install from source for a highly optimized container environment.
*   **NGC NeMo Container:**  Ready-to-use solution for optimal performance, pre-configured with dependencies.

**Detailed installation instructions are available in the [Install NeMo Framework](#install-nemo-framework) section.**

### Tutorials & Resources

*   **User Guide:** [NeMo Framework User Guide](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
*   **Quickstart:** For examples of using NeMo-Run to launch NeMo 2.0 experiments locally and on a slurm cluster.
*   **Examples:** [Example scripts](https://github.com/NVIDIA/NeMo/tree/main/examples) for multi-GPU/multi-node training.
*   **Pre-trained Models:**  Explore pre-trained models on [Hugging Face Hub](https://huggingface.co/models?library=nemo&sort=downloads&search=nvidia) and [NVIDIA NGC](https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC).

## Core Functionality

*   **LLMs and MMs Training, Alignment, and Customization:** NeMo supports cutting-edge distributed training techniques and various parallelism strategies, as well as state-of-the-art methods like SteerLM and RLHF for LLM alignment.
*   **LLMs and MMs Deployment and Optimization:**  Deploy and optimize with NVIDIA NeMo Microservices.
*   **Speech AI:** Optimize and deploy ASR and TTS models with NVIDIA Riva.

## Requirements

*   Python 3.10 or above
*   PyTorch 2.5 or above
*   NVIDIA GPU (for model training)

## Developer Documentation

*   **Latest:**  [Documentation of the latest (i.e. main) branch.](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
*   **Stable:** [Documentation of the stable (i.e. most recent release)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/)

## Contribute

We welcome community contributions!  Refer to [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md) for guidelines.

## Publications

Explore a list of publications that utilize the NeMo Framework: [publications](https://nvidia.github.io/NeMo/publications/)

## Discussions

Find answers to frequently asked questions and engage in discussions on the [NeMo Discussions Board](https://github.com/NVIDIA/NeMo/discussions).

## Licenses

NeMo is licensed under the [Apache License 2.0](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file).