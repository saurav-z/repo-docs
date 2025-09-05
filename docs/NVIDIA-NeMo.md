<!-- Project Badges -->
[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# NVIDIA NeMo: Unleash the Power of Generative AI for LLMs, ASR, and More

[Explore the NVIDIA NeMo Framework on GitHub](https://github.com/NVIDIA/NeMo) to build, customize, and deploy state-of-the-art generative AI models with ease.

## **Key Features**

*   **Large Language Models (LLMs):** Develop and fine-tune powerful LLMs.
*   **Multimodal Models (MMs):** Explore models that combine various data types (text, images, etc.).
*   **Automatic Speech Recognition (ASR):** Build and deploy high-accuracy speech recognition systems.
*   **Text-to-Speech (TTS):** Create realistic and expressive synthetic speech.
*   **Computer Vision (CV):** Develop and customize computer vision models.
*   **Modular and Scalable:** Built with PyTorch Lightning for modularity and seamless scaling across thousands of GPUs.
*   **Optimized Training:** Leverages techniques like Tensor Parallelism, Pipeline Parallelism, and FP8 for efficient training.
*   **Hugging Face Integration:** Supports pre-training and fine-tuning of Hugging Face models.
*   **Comprehensive Documentation:** Extensive documentation and tutorials for getting started.

## **What's New**

*   **NVIDIA NeMo 2.0:** Features include a Python-based configuration, modular abstractions, and streamlined scalability, optimized to run on the latest NVIDIA hardware.
*   **Hugging Face Integration:** Enhanced support for pretraining and finetuning Hugging Face models, including AutoModelForCausalLM and AutoModelForImageTextToText.
*   **Cosmos Support:** Training and customization for NVIDIA Cosmos world foundation models for creating realistic synthetic videos of environments.
*   **Model Updates:** Support for new models like Llama 4, Flux, Llama Nemotron, Hyena, Qwen2-VL, Qwen2.5, Gemma3, Qwen3-30B&32B, and more.
*   **Performance Benchmarks:** Performance benchmarks and optimizations for latest NVIDIA hardware.

## **Introduction**

The NVIDIA NeMo Framework is a cloud-native, flexible framework for researchers and PyTorch developers. It simplifies the creation, customization, and deployment of generative AI models across various domains. Leverage existing code and pre-trained model checkpoints to accelerate your AI development.  The framework is well-suited for Large Language Models (LLMs), Multimodal Models (MMs), Automatic Speech Recognition (ASR), Text-to-Speech (TTS), and Computer Vision (CV).

## **Getting Started**

1.  **Install:** Follow the instructions in the [Install NeMo Framework](#install-nemo-framework) section.
2.  **Explore Pre-trained Models:** Find pre-trained models on [Hugging Face Hub](https://huggingface.co/models?library=nemo&sort=downloads&search=nvidia) and [NVIDIA NGC](https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC).
3.  **Tutorials and Examples:** Run tutorials on [Google Colab](https://colab.research.google.com) or with the [NGC NeMo Framework Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo). Explore [example scripts](https://github.com/NVIDIA/NeMo/tree/main/examples) for advanced training.

## **Installation**

Choose the installation method that best suits your needs:

*   **Conda / Pip:** Recommended for ASR and TTS, for exploring NeMo on any supported platform. See the [Conda / Pip](#conda--pip) section.
*   **NGC PyTorch Container:** Install from source within a highly optimized container. See the [NGC PyTorch container](#ngc-pytorch-container) section.
*   **NGC NeMo Container:** Pre-built and optimized for performance. See the [NGC NeMo container](#ngc-nemo-container) section.

### **Requirements**

*   Python 3.10 or above
*   PyTorch 2.5 or above
*   NVIDIA GPU (for training)

## **Developer Documentation**

For detailed technical information, see the [NeMo Framework User Guide](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/).

## **Contribute**

Contributions are welcome! See [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md) for guidelines.

## **License**

NVIDIA NeMo is licensed under the [Apache License 2.0](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file).

---

_This README provides a high-level overview. For detailed information, tutorials, and examples, please refer to the official documentation and the NVIDIA NeMo repository._