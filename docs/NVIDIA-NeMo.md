[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# NVIDIA NeMo: The Open-Source Framework for Cutting-Edge Generative AI

NVIDIA NeMo is a comprehensive and versatile framework designed to streamline the development, customization, and deployment of Large Language Models (LLMs), Multimodal Models (MMs), and other generative AI applications. [Explore the original repository](https://github.com/NVIDIA/NeMo).

## Key Features

*   **LLMs and MMs:** Train, fine-tune, align, and customize LLMs and MMs with cutting-edge techniques like Tensor Parallelism, Pipeline Parallelism, FSDP, MoE, and PEFT methods.
*   **Speech AI:** Optimize ASR and TTS models for inference and deploy them using NVIDIA Riva.
*   **Multimodal Capabilities:** Support for vision and language models, including integration with Cosmos and NeMo Curator for video processing.
*   **Modular Design:** Built with PyTorch Lightning for flexibility and ease of use.
*   **Scalability:** Seamlessly scale experiments across thousands of GPUs with NeMo-Run.
*   **Deployment & Optimization:** Deploy and optimize LLMs and MMs with NVIDIA NeMo Microservices.

## What's New

*   **Hugging Face Integration:** Broad support for Hugging Face models via AutoModel, enabling pretraining and finetuning for text generation and image-to-text tasks.
*   **Blackwell Support:** Added support for Blackwell, with performance benchmarks.
*   **Performance Tuning Guide:** Comprehensive guide available for performance tuning.
*   **New Model Support:** Support for Llama 4, Flux, Llama Nemotron, Hyena & Evo2, Qwen2-VL, Qwen2.5, Gemma3, Qwen3-30B&32B.
*   **NeMo 2.0:** Significant updates including Python-based configuration, modular abstractions, and enhanced scalability.
*   **Cosmos Support:** Support for training and customizing NVIDIA Cosmos world foundation models.

## Getting Started

*   **Quickstart:** Explore examples of using NeMo-Run.
*   **User Guide:** Access comprehensive documentation and guides.
*   **Recipes:** Find examples of large-scale runs.
*   **Feature Guide:** Explore the main features of NeMo 2.0.
*   **Migration Guide:** Transition from NeMo 1.0 to 2.0.

## Installation

Choose the best installation method depending on your needs:

*   **Conda / Pip:**  Install with native Pip into a virtual environment.
*   **NGC PyTorch container:** Install from source into a highly optimized container.
*   **NGC NeMo container:** Ready-to-go solution with all dependencies installed.

### Conda / Pip

```bash
conda create --name nemo python==3.10.12
conda activate nemo
pip install "nemo_toolkit[all]" # or specific versions via Git
```

### NGC PyTorch container

```bash
docker run --gpus all -it --rm --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/pytorch:${NV_PYTORCH_TAG:-'nvcr.io/nvidia/pytorch:25.01-py3'}
```

Install NeMo from the container:
```bash
cd /opt
git clone https://github.com/NVIDIA/NeMo
cd NeMo
git checkout ${REF:-'main'}
bash docker/common/install_dep.sh --library all
pip install ".[all]"
```

### NGC NeMo container

```bash
docker run --gpus all -it --rm --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/pytorch:${NV_PYTORCH_TAG:-'nvcr.io/nvidia/nemo:25.02'}
```

## Requirements

*   Python 3.10 or above
*   Pytorch 2.5 or above
*   NVIDIA GPU (for model training)

## Resources

*   **Developer Documentation:** Comprehensive documentation available.
*   **Discussions Board:** Ask questions and start discussions.
*   **Contribute:** Contribute to the NeMo project.
*   **Publications:** Access a list of publications using NeMo.
*   **Blogs:** Read blog posts for updates and insights.

## License

NVIDIA NeMo is licensed under the [Apache License 2.0](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file).