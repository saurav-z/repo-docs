[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# NVIDIA NeMo: Build, Customize, and Deploy Generative AI Models

**NVIDIA NeMo is a cloud-native framework empowering researchers and developers to create cutting-edge Large Language Models (LLMs), Multimodal Models (MMs), and more.**  ([View the original repository](https://github.com/NVIDIA/NeMo))

## Key Features

*   **LLMs and MMs:** Train, customize, and align large language models using cutting-edge techniques like SteerLM, DPO, and RLHF.
*   **Automatic Speech Recognition (ASR):**  Optimize and deploy ASR models for production with NVIDIA Riva.
*   **Text-to-Speech (TTS):** Utilize pre-trained models or build your own with NeMo's TTS capabilities.
*   **Multimodal AI:** Explore and build multimodal models by leveraging NeMo's support for vision and language tasks.
*   **Computer Vision (CV):** Build and experiment with computer vision models using NeMo's available tools.
*   **Modular Design:**  Benefit from PyTorch Lightning's modular abstractions for easier adaptation and experimentation.
*   **Scalability:** Seamlessly scale experiments across thousands of GPUs with tools like NeMo-Run.
*   **Parameter-Efficient Fine-tuning (PEFT):** Utilize techniques like LoRA, P-Tuning, and Adapters.
*   **Deployment and Optimization:** Deploy and optimize LLMs and MMs with NVIDIA NeMo Microservices.

## What's New

*   **Hugging Face Integration:** NeMo 2.0 and later includes support for Hugging Face models via AutoModel, with specific focus on models like AutoModelForCausalLM and AutoModelForImageTextToText.
*   **Blackwell Support:** Includes performance benchmarks on GB200 & B200.
*   **Performance Tuning Guide:**  A comprehensive guide for performance tuning to achieve optimal throughput.
*   **New Model Support:** Expanded support for community models including Llama 4, Flux, Llama Nemotron, Hyena & Evo2, Qwen2-VL, Qwen2.5, Gemma3, and Qwen3-30B&32B.
*   **NVIDIA Cosmos World Foundation Model Support:**  Train and customize NVIDIA Cosmos video models, and accelerate your video processing.
*   **NeMo 2.0 Release:**  Focuses on modularity, ease-of-use, and Python-based configuration.

## Getting Started

*   **Quickstart:**  Explore examples of using NeMo-Run to launch NeMo 2.0 experiments.
*   **User Guide:** Access comprehensive documentation for NeMo Framework.
*   **Recipes:**  Find examples of launching large-scale runs.
*   **Feature Guide:**  Explore the main features of NeMo 2.0.
*   **Migration Guide:**  Transition from NeMo 1.0 to 2.0.
*   **Pre-trained Models:** Utilize state-of-the-art pre-trained models available on [Hugging Face Hub](https://huggingface.co/models?library=nemo&sort=downloads&search=nvidia) and [NVIDIA NGC](https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC).

## Installation

Choose your installation method based on your needs:

*   **Conda / Pip:** Install NeMo with native Pip into a virtual environment. Ideal for exploring and the ASR and TTS domains.
*   **NGC PyTorch Container:** Install NeMo from source into a highly optimized container.
*   **NGC NeMo Container:** Ready-to-go solution with all dependencies and optimized for performance.

### Installation Instructions

```bash
# Conda / Pip
conda create --name nemo python==3.10.12
conda activate nemo
pip install "nemo_toolkit[all]" # Or install specific domain: "nemo_toolkit['asr']" etc.

# NGC PyTorch Container
docker run --gpus all -it --rm --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/pytorch:${NV_PYTORCH_TAG:-'nvcr.io/nvidia/pytorch:25.01-py3'}
cd /opt
git clone https://github.com/NVIDIA/NeMo
cd NeMo
git checkout ${REF:-'main'}
bash docker/common/install_dep.sh --library all
pip install ".[all]"

# NGC NeMo Container
docker run --gpus all -it --rm --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/pytorch:${NV_PYTORCH_TAG:-'nvcr.io/nvidia/nemo:25.02'}
```

## Requirements

*   Python 3.10 or above
*   PyTorch 2.5 or above
*   NVIDIA GPU (for training)

## Documentation

*   [NeMo Framework User Guide](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)

## Resources

*   **Discussions:** [NeMo Discussions Board](https://github.com/NVIDIA/NeMo/discussions)
*   **Contribute:** [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md)
*   **Publications:** [Publications](https://nvidia.github.io/NeMo/publications/)
*   **Blogs:** (See expanded blog section in original README)

## License

NVIDIA NeMo is licensed under the [Apache License 2.0](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file).