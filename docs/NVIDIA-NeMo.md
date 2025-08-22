<!--
  This README is optimized for SEO and clarity.
  Key improvements:
    - Clear headings and subheadings for easy navigation.
    - Bulleted lists for key features and benefits.
    - Up-to-date information from the original README, focusing on the latest releases.
    -  Added relevant keywords for SEO optimization.
    - Links back to the original repo and documentation.
-->

[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# NVIDIA NeMo: Build, Customize, and Deploy Generative AI Models

**NVIDIA NeMo is a cloud-native framework empowering researchers and developers to create, customize, and deploy state-of-the-art generative AI models with ease.** [Visit the original repository for more details.](https://github.com/NVIDIA/NeMo)

## Key Features & Benefits

*   **Comprehensive AI Domain Support:**  NeMo supports Large Language Models (LLMs), Multimodal Models (MMs), Automatic Speech Recognition (ASR), Text-to-Speech (TTS), and Computer Vision (CV).
*   **Modular and Flexible:** Designed for modularity and ease-of-use for PyTorch developers.
*   **Scalable Training:**  Easily scale experiments across thousands of GPUs.
*   **Pre-trained Models:** Access and utilize state-of-the-art pre-trained models on Hugging Face and NVIDIA NGC.
*   **Cutting-Edge Techniques:** Leverages advanced training techniques like Tensor Parallelism, Pipeline Parallelism, FSDP, MoE, and Mixed Precision Training (BFloat16 and FP8).
*   **Optimized for NVIDIA Hardware:** Utilizes NVIDIA Transformer Engine for FP8 training and NVIDIA Megatron Core for scaling Transformer model training, ensuring optimal performance on NVIDIA Hopper GPUs and beyond.
*   **Model Alignment:** Supports state-of-the-art methods such as SteerLM, Direct Preference Optimization (DPO), and Reinforcement Learning from Human Feedback (RLHF).
*   **Parameter Efficient Fine-tuning (PEFT):** Supports popular PEFT techniques, including LoRA, P-Tuning, Adapters, and IA3.
*   **Deployment and Optimization:** Integrate NeMo models with NVIDIA Riva for Speech AI applications and NeMo Microservices for LLMs and MMs.
*   **Flexible Installation:** Supports Conda/Pip, NGC PyTorch Container, and NGC NeMo Container installation methods.

## What's New: Key Highlights & Recent Updates

*   **[Pretrain and finetune :hugs:Hugging Face models via AutoModel](https://developer.nvidia.com/blog/run-hugging-face-models-instantly-with-day-0-support-from-nvidia-nemo-framework)**: AutoModel enables broad support for :hugs:Hugging Face models.
*   **[Training on Blackwell using Nemo](https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance_summary.html)**: NeMo Framework has added Blackwell support, with performance benchmarks on GB200 & B200.
*   **[Training Performance on GPU Tuning Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance-guide.html)**: A comprehensive guide for performance tuning to achieve optimal throughput!
*   **[New Models Support](https://docs.nvidia.com/nemo-framework/user-guide/latest/vlms/llama4.html)**: Added support for latest community models - Llama 4, Flux, Llama Nemotron, Hyena & Evo2, Qwen2-VL, Qwen2.5, Gemma3, Qwen3-30B&32B.
*   **[NeMo Framework 2.0](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/index.html)**: Focuses on modularity and ease-of-use.

## Getting Started

### Installation

Choose the installation method that best suits your needs:

*   **Conda / Pip:** Recommended for ASR and TTS. Provides limited feature completeness for other domains.
*   **NGC PyTorch Container:** For feature-complete installations from source.
*   **NGC NeMo Container:** Pre-built container for optimal performance.

Detailed installation instructions are available in the [original README](https://github.com/NVIDIA/NeMo).

### Resources

*   **Documentation:** [NeMo Framework User Guide](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
*   **Tutorials:** Comprehensive tutorials run on [Google Colab](https://colab.research.google.com) or with the [NGC NeMo Framework Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo).
*   **Playbooks:** Training NeMo models with the NeMo Framework Launcher.
*   **Example Scripts:** Support multi-GPU/multi-node training for advanced users.
*   **NGC:** [NGC](https://catalog.ngc.nvidia.com/models?query=nemo&orderBy=weightPopularDESC)
*   **Hugging Face Hub:** [Hugging Face Hub](https://huggingface.co/models?library=nemo&sort=downloads&search=nvidia)

## Contribute

We encourage community contributions! Please review our [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md) for guidelines.

## Additional Information

*   **Publications:** [List of publications](https://nvidia.github.io/NeMo/publications/) that utilize NeMo.
*   **Discussions:** [NeMo Discussions board](https://github.com/NVIDIA/NeMo/discussions) for questions and discussions.
*   **Licenses:** Apache License 2.0