[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# NVIDIA NeMo: Build and Deploy Generative AI Models at Scale

NVIDIA NeMo is a comprehensive and cloud-native framework that empowers researchers and developers to build, customize, and deploy state-of-the-art generative AI models for various applications.  For more details, visit the original [NVIDIA NeMo repository](https://github.com/NVIDIA/NeMo).

**Key Features:**

*   **Large Language Models (LLMs):** Train, fine-tune, and deploy powerful LLMs with cutting-edge techniques like Transformer Engine and Megatron Core.
*   **Multimodal Models (MMs):**  Develop models that process and generate multiple data types like text and images.
*   **Automatic Speech Recognition (ASR):** Build high-accuracy ASR models optimized for inference with NVIDIA Riva.
*   **Text-to-Speech (TTS):** Create realistic and expressive speech synthesis models deployable with NVIDIA Riva.
*   **Computer Vision (CV):** Utilize NeMo for diverse computer vision tasks.
*   **Modular Architecture:** NeMo 2.0 provides a Python-based configuration, modular abstractions, and seamless scalability.
*   **Model Alignment:**  Leverage methods like SteerLM, DPO, and RLHF to align LLMs.
*   **Parameter-Efficient Fine-tuning (PEFT):**  Utilize techniques like LoRA and adapters for efficient model customization.
*   **Cosmos World Foundation Models Support**: NeMo supports training and customizing NVIDIA Cosmos, the end-to-end platform for physical AI systems.

**Latest News and Updates:**

*   **[Pretrain and finetune :hugs:Hugging Face models via AutoModel](https://developer.nvidia.com/blog/run-hugging-face-models-instantly-with-day-0-support-from-nvidia-nemo-framework)**: Added support for Hugging Face models.
*   **[Training on Blackwell using Nemo](https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance_summary.html)**: Added Blackwell support.
*   **[Training Performance on GPU Tuning Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance-guide.html)**: Published a guide for performance tuning.
*   **[New Models Support](https://docs.nvidia.com/nemo-framework/user-guide/latest/vlms/llama4.html)**: Added support for community models.
*   **[NeMo Framework 2.0](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/index.html)**: Released NeMo 2.0 with improved modularity.
*   **[New Cosmos World Foundation Models Support](https://developer.nvidia.com/blog/advancing-physical-ai-with-nvidia-cosmos-world-foundation-model-platform)**: Accelerate world model development for physical AI systems.
*   **[Large Language Models and Multimodal Models](https://developer.nvidia.com/blog/state-of-the-art-multimodal-generative-ai-model-development-with-nvidia-nemo/)**: Enhanced NeMo with multimodal generative AI models.
*   **[Speech Recognition](https://developer.nvidia.com/blog/accelerating-leaderboard-topping-asr-models-10x-with-nvidia-nemo/)**: Multiple optimizations and improvements for ASR.

**Getting Started:**

*   **User Guide:** [NeMo Framework User Guide](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
*   **Tutorials:** Comprehensive [tutorials](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/starthere/tutorials.html) that can be run on [Google Colab](https://colab.research.google.com) or with our [NGC NeMo Framework Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo).
*   **Pre-trained Models:** Explore pre-trained models on [Hugging Face Hub](https://huggingface.co/models?library=nemo&sort=downloads&search=nvidia) and [NVIDIA NGC](https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC).

**Installation:**

Choose the installation method that best suits your needs:

*   [Conda / Pip](#conda--pip): Ideal for exploring NeMo and recommended for ASR/TTS.
*   [NGC PyTorch container](#ngc-pytorch-container):  Install from source in a highly optimized container.
*   [NGC NeMo container](#ngc-nemo-container): Ready-to-go solution for the best performance.

**Requirements:**

*   Python 3.10 or above
*   PyTorch 2.5 or above
*   NVIDIA GPU (for training)

**Contribute:**

We welcome community contributions! Please refer to [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md) for details.

**License:**

NVIDIA NeMo is licensed under the [Apache License 2.0](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file).