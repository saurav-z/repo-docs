[![Project Status: Active -- The project has reached a stable, usable state and is being actively developed.](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Documentation](https://readthedocs.com/projects/nvidia-nemo/badge/?version=main)](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
[![CodeQL](https://github.com/nvidia/nemo/actions/workflows/codeql.yml/badge.svg?branch=main&event=push)](https://github.com/nvidia/nemo/actions/workflows/codeql.yml)
[![NeMo core license and license for collections in this repo](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://github.com/NVIDIA/NeMo/blob/master/LICENSE)
[![Release version](https://badge.fury.io/py/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![Python version](https://img.shields.io/pypi/pyversions/nemo-toolkit.svg)](https://badge.fury.io/py/nemo-toolkit)
[![PyPi total downloads](https://static.pepy.tech/personalized-badge/nemo-toolkit?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tech/project/nemo-toolkit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# NVIDIA NeMo: Build, Customize, and Deploy Generative AI Models

**NVIDIA NeMo is a powerful, cloud-native framework enabling researchers and developers to efficiently create, customize, and deploy state-of-the-art generative AI models.**  (See the original repo [here](https://github.com/NVIDIA/NeMo).)

## Key Features

*   **Large Language Models (LLMs):** Train, fine-tune, and deploy LLMs with cutting-edge techniques like Transformer Engine and Megatron Core.
*   **Multimodal Models (MMs):** Develop models that combine different data modalities, like text and images.
*   **Automatic Speech Recognition (ASR):** Build and optimize ASR models for various applications.
*   **Text-to-Speech (TTS):** Create high-quality speech synthesis models.
*   **Computer Vision (CV):** Develop CV models.
*   **Modular Design:** Benefit from modular abstractions and PyTorch Lightning integrations.
*   **Scalability:** Train models on thousands of GPUs.
*   **Pre-trained Models:** Leverage a wide range of pre-trained models available on Hugging Face Hub and NGC.
*   **Model Alignment:** Utilize methods like SteerLM, DPO, and RLHF for LLM alignment.
*   **PEFT Support:** Use techniques like LoRA, P-Tuning, and Adapters for parameter-efficient fine-tuning.
*   **Deployment and Optimization:** Deploy and optimize LLMs and MMs with NVIDIA NeMo Microservices and Riva.
*   **Cosmos Support:** Supports training and customizing the NVIDIA Cosmos collection of world foundation models.

## What's New:

*   **[Pretrain and finetune :hugs:Hugging Face models via AutoModel](https://developer.nvidia.com/blog/run-hugging-face-models-instantly-with-day-0-support-from-nvidia-nemo-framework)**
*   **[Training on Blackwell using Nemo](https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance_summary.html)**
*   **[Training Performance on GPU Tuning Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance-guide.html)**
*   **[New Models Support](https://docs.nvidia.com/nemo-framework/user-guide/latest/vlms/llama4.html)**
*   **[NeMo Framework 2.0](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/index.html)**
*   **[New Cosmos World Foundation Models Support](https://developer.nvidia.com/blog/advancing-physical-ai-with-nvidia-cosmos-world-foundation-model-platform)**
*   **[Accelerate Custom Video Foundation Model Pipelines with New NVIDIA NeMo Framework Capabilities](https://developer.nvidia.com/blog/accelerate-custom-video-foundation-model-pipelines-with-new-nvidia-nemo-framework-capabilities/)**
*   **[State-of-the-Art Multimodal Generative AI Model Development with NVIDIA NeMo](https://developer.nvidia.com/blog/state-of-the-art-multimodal-generative-ai-model-development-with-nvidia-nemo/)**
*   **[New Llama 3.1 Support](https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/llama/index.html#new-llama-3-1-support for more information/)**
*   **[Accelerate your Generative AI Distributed Training Workloads with the NVIDIA NeMo Framework on Amazon EKS](https://aws.amazon.com/blogs/machine-learning/accelerate-your-generative-ai-distributed-training-workloads-with-the-nvidia-nemo-framework-on-amazon-eks/)**
*   **[NVIDIA NeMo Accelerates LLM Innovation with Hybrid State Space Model Support](https://developer.nvidia.com/blog/nvidia-nemo-accelerates-llm-innovation-with-hybrid-state-space-model-support/)**
*   **[NVIDIA releases 340B base, instruct, and reward models pretrained on a total of 9T tokens](https://huggingface.co/models?sort=trending&search=nvidia%2Fnemotron-4-340B)**
*   **[NVIDIA sets new generative AI performance and scale records in MLPerf Training v4.0](https://developer.nvidia.com/blog/nvidia-sets-new-generative-ai-performance-and-scale-records-in-mlperf-training-v4-0/)**
*   **[Accelerate your generative AI journey with NVIDIA NeMo Framework on GKE](https://cloud.google.com/blog/products/compute/gke-and-nvidia-nemo-framework-to-train-generative-ai-models)**
*   **[Accelerating Leaderboard-Topping ASR Models 10x with NVIDIA NeMo](https://developer.nvidia.com/blog/accelerating-leaderboard-topping-asr-models-10x-with-nvidia-nemo/)**
*   **[New Standard for Speech Recognition and Translation from the NVIDIA NeMo Canary Model](https://developer.nvidia.com/blog/new-standard-for-speech-recognition-and-translation-from-the-nvidia-nemo-canary-model/)**
*   **[Pushing the Boundaries of Speech Recognition with NVIDIA NeMo Parakeet ASR Models](https://developer.nvidia.com/blog/pushing-the-boundaries-of-speech-recognition-with-nemo-parakeet-asr-models/)**
*   **[Turbocharge ASR Accuracy and Speed with NVIDIA NeMo Parakeet-TDT](https://developer.nvidia.com/blog/turbocharge-asr-accuracy-and-speed-with-nvidia-nemo-parakeet-tdt/)**

## Get Started

*   **Quickstart:** Get started with examples of using NeMo-Run to launch NeMo 2.0 experiments locally and on a slurm cluster ([Quickstart](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/quickstart.html))
*   **NeMo Framework User Guide:** For more information about NeMo 2.0 ([NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/index.html)).
*   **NeMo 2.0 Recipes:** Additional examples of launching large-scale runs using NeMo 2.0 and NeMo-Run ([NeMo 2.0 Recipes](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/recipes)).
*   **Feature Guide:** For an in-depth exploration of the main features of NeMo 2.0 ([Feature Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/features/index.html#feature-guide)).
*   **Migration Guide:** To transition from NeMo 1.0 to 2.0, see the [Migration Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/migration/index.html#migration-guide) for step-by-step instructions.

## LLMs and MMs Training, Alignment, and Customization

All NeMo models are trained with
[Lightning](https://github.com/Lightning-AI/lightning). Training is
automatically scalable to 1000s of GPUs. You can check the performance benchmarks using the
latest NeMo Framework container [here](https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance_summary.html).

When applicable, NeMo models leverage cutting-edge distributed training
techniques, incorporating [parallelism
strategies](https://docs.nvidia.com/nemo-framework/user-guide/latest/modeloverview.html)
to enable efficient training of very large models. These techniques
include Tensor Parallelism (TP), Pipeline Parallelism (PP), Fully
Sharded Data Parallelism (FSDP), Mixture-of-Experts (MoE), and Mixed
Precision Training with BFloat16 and FP8, as well as others.

NeMo Transformer-based LLMs and MMs utilize [NVIDIA Transformer
Engine](https://github.com/NVIDIA/TransformerEngine) for FP8 training on
NVIDIA Hopper GPUs, while leveraging [NVIDIA Megatron
Core](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core) for
scaling Transformer model training.

NeMo LLMs can be aligned with state-of-the-art methods such as SteerLM,
Direct Preference Optimization (DPO), and Reinforcement Learning from
Human Feedback (RLHF). See [NVIDIA NeMo
Aligner](https://github.com/NVIDIA/NeMo-Aligner) for more information.

In addition to supervised fine-tuning (SFT), NeMo also supports the
latest parameter efficient fine-tuning (PEFT) techniques such as LoRA,
P-Tuning, Adapters, and IA3. Refer to the [NeMo Framework User
Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/sft_peft/index.html)
for the full list of supported models and techniques.

## LLMs and MMs Deployment and Optimization

NeMo LLMs and MMs can be deployed and optimized with [NVIDIA NeMo
Microservices](https://developer.nvidia.com/nemo-microservices-early-access).

## Speech AI

NeMo ASR and TTS models can be optimized for inference and deployed for
production use cases with [NVIDIA Riva](https://developer.nvidia.com/riva).

## Installation

### Requirements

*   Python 3.10 or above
*   Pytorch 2.5 or above
*   NVIDIA GPU (if training)

### Methods

*   **Conda / Pip:** Install using pip into a virtual environment.
    *   Recommended for ASR and TTS domains.
    *   `pip install "nemo_toolkit[all]"`
*   **NGC PyTorch container:** Install from source into a highly optimized container.
    *   Supported starting with 24.04 (NeMo-Toolkit 2.3.0).
*   **NGC NeMo container:** Use a pre-built, ready-to-go solution.

### Install

See [Install](#install-nemo-framework) section for detailed instructions.

## Developer Documentation

*   [Latest Documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/)
*   [Stable Documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/)

## Contribute

We welcome community contributions; see [CONTRIBUTING.md](https://github.com/NVIDIA/NeMo/blob/stable/CONTRIBUTING.md).

## Publications

Explore the [list of publications](https://nvidia.github.io/NeMo/publications/) leveraging NeMo. Contribute articles via a pull request to the `gh-pages-src` branch.

## Blogs

*   **[Bria Builds Responsible Generative AI for Enterprises Using NVIDIA NeMo, Picasso](https://blogs.nvidia.com/blog/bria-builds-responsible-generative-ai-using-nemo-picasso/)**
*   **[New NVIDIA NeMo Framework Features and NVIDIA H200](https://developer.nvidia.com/blog/new-nvidia-nemo-framework-features-and-nvidia-h200-supercharge-llm-training-performance-and-versatility/)**
*   **[NVIDIA now powers training for Amazon Titan Foundation models](https://blogs.nvidia.com/blog/nemo-amazon-titan/)**

## License

This project is licensed under the [Apache License 2.0](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file).