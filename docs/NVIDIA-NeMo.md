[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# NVIDIA NeMo: Your Toolkit for Cutting-Edge Generative AI

NVIDIA NeMo is a cloud-native framework, providing researchers and developers with the tools to efficiently build, customize, and deploy state-of-the-art generative AI models for LLMs, multimodal applications, speech recognition, text-to-speech, and computer vision.  For more details, visit the original [NVIDIA NeMo repository](https://github.com/NVIDIA/NeMo).

**Key Features:**

*   **Comprehensive Domain Support:**  Develop models for Large Language Models (LLMs), Multimodal Models (MMs), Automatic Speech Recognition (ASR), Text-to-Speech (TTS), and Computer Vision (CV).
*   **Scalable Training:** Train models efficiently across thousands of GPUs using advanced parallelism techniques.
*   **Modular and Flexible:** Benefit from a Python-based configuration system and modular abstractions for easier customization.
*   **Pre-trained Models:** Access a wide variety of pre-trained models on [Hugging Face Hub](https://huggingface.co/models?library=nemo&sort=downloads&search=nvidia) and [NVIDIA NGC](https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC) to accelerate your projects.
*   **Deployment and Optimization:** Deploy and optimize LLMs and MMs with [NVIDIA NeMo Microservices](https://developer.nvidia.com/nemo-microservices-early-access) and utilize NVIDIA Riva for speech AI.
*   **Extensive Documentation & Support:** Access detailed documentation and tutorials to help you get started quickly.

## What's New

*   **[Pretrain and finetune :hugs:Hugging Face models via AutoModel](https://developer.nvidia.com/blog/run-hugging-face-models-instantly-with-day-0-support-from-nvidia-nemo-framework)**: Nemo Framework's latest feature AutoModel enables broad support for :hugs:Hugging Face models.
*   **Training on Blackwell using Nemo:**  NeMo Framework has added Blackwell support with performance benchmarks on GB200 & B200.
*   **Training Performance on GPU Tuning Guide**: NeMo Framework has published [a comprehensive guide for performance tuning to achieve optimal throughput](https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance-guide.html)!
*   **New Models Support**: NeMo Framework has added support for latest community models - [Llama 4](https://docs.nvidia.com/nemo-framework/user-guide/latest/vlms/llama4.html), [Flux](https://docs.nvidia.com/nemo-framework/user-guide/latest/vision/diffusionmodels/flux.html), [Llama Nemotron](https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/llama_nemotron.html), [Hyena & Evo2](https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/hyena.html#), [Qwen2-VL](https://docs.nvidia.com/nemo-framework/user-guide/latest/vlms/qwen2vl.html), [Qwen2.5](https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/qwen2.html), Gemma3, Qwen3-30B&32B.

*   **NVIDIA NeMo 2.0:** Upgrades include a Python-based configuration, modular abstractions, and improved scalability. Check out the [NeMo 2.0 Quickstart](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/quickstart.html) and [Feature Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/features/index.html#feature-guide).

*   **Cosmos Integration:** Training and customization support for [NVIDIA Cosmos](https://github.com/NVIDIA/Cosmos) world foundation models for physical AI.  See [NeMo Curator](https://developer.nvidia.com/nemo-curator) for video processing.

## Getting Started

*   **Install:** Choose from various installation methods, including Conda/Pip and NGC containers.
*   **Tutorials & Examples:** Utilize extensive tutorials, example scripts, and [playbooks](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html) to quickly begin training and deploying models.

## Requirements

*   Python 3.10 or above
*   PyTorch 2.5 or above
*   NVIDIA GPU (for model training)

## Resources

*   **Developer Documentation:** Find the latest documentation at  [NeMo Framework User Guide](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/).
*   **Discussions Board:** Get your questions answered and join the community on the [NeMo Discussions board](https://github.com/NVIDIA/NeMo/discussions).
*   **Contribute:** Help improve NeMo by reviewing the [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md) for the contribution process.
*   **Publications:** Explore [publications](https://nvidia.github.io/NeMo/publications/) that utilize the NeMo Framework.

## License

Licensed under the [Apache License 2.0](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file).