<div align="center">
  <img src="assets/logo.png" alt="LLaMA Factory Logo" width="200">
  <h1>LLaMA Factory: Fine-tune Any Large Language Model with Ease</h1>
  <p>
    <i>Easily fine-tune 100+ LLMs with zero-code CLI and Web UI.</i>
    <br>
    <a href="https://github.com/hiyouga/LLaMA-Factory">
      <img src="https://img.shields.io/github/stars/hiyouga/LLaMA-Factory?style=social" alt="GitHub Stars">
    </a>
    <a href="https://github.com/hiyouga/LLaMA-Factory">
      <img src="https://img.shields.io/github/last-commit/hiyouga/LLaMA-Factory" alt="Last Commit">
    </a>
    <a href="https://github.com/hiyouga/LLaMA-Factory">
      <img src="https://img.shields.io/github/contributors/hiyouga/LLaMA-Factory?color=orange" alt="Contributors">
    </a>
    <a href="https://github.com/hiyouga/LLaMA-Factory/actions/workflows/tests.yml">
      <img src="https://github.com/hiyouga/LLaMA-Factory/actions/workflows/tests.yml/badge.svg" alt="Build Status">
    </a>
    <a href="https://pypi.org/project/llamafactory/">
      <img src="https://img.shields.io/pypi/v/llamafactory" alt="PyPI">
    </a>
    <a href="https://scholar.google.com/scholar?cites=12620864006390196564">
      <img src="https://img.shields.io/badge/citation-730-green" alt="Citations">
    </a>
    <a href="https://hub.docker.com/r/hiyouga/llamafactory/tags">
      <img src="https://img.shields.io/docker/pulls/hiyouga/llamafactory" alt="Docker Pulls">
    </a>
  </p>
</div>

<p>
  LLaMA Factory empowers you to fine-tune a wide array of large language models (LLMs) without requiring extensive coding knowledge. It simplifies the process of customizing models for specific tasks, research, or applications.
</p>

**Key Features:**

*   ✅ **Broad Model Support:** Fine-tune LLaMA, LLaVA, Mistral, Mixtral, Qwen, DeepSeek, Yi, Gemma, ChatGLM, Phi, and many more!
*   ✅ **Versatile Training Methods:** Supports (Continuous) pre-training, (multimodal) supervised fine-tuning, reward modeling, PPO, DPO, KTO, ORPO, and more.
*   ✅ **Efficient Resource Utilization:** Leverage 16-bit full-tuning, freeze-tuning, LoRA, and 2/3/4/5/6/8-bit QLoRA for optimized memory and speed.
*   ✅ **Advanced Algorithms:** Incorporates cutting-edge techniques such as GaLore, BAdam, APOLLO, Adam-mini, Muon, DoRA, LongLoRA, LLaMA Pro, Mixture-of-Depths, LoRA+, LoftQ and PiSSA.
*   ✅ **Performance Enhancements:** Includes FlashAttention-2, Unsloth, Liger Kernel, RoPE scaling, NEFTune and rsLoRA for improved training and inference.
*   ✅ **Task Diversity:** Ready for multi-turn dialogue, tool using, image understanding, visual grounding, video recognition, audio understanding, and a variety of applications.
*   ✅ **Comprehensive Monitoring:** Integrated with LlamaBoard, TensorBoard, Wandb, MLflow, SwanLab for in-depth experiment tracking.
*   ✅ **Faster Inference:** Offers OpenAI-style API, Gradio UI, and CLI support with vLLM or SGLang backends.

**Used By:** [Amazon](https://aws.amazon.com/cn/blogs/machine-learning/how-apoidea-group-enhances-visual-information-extraction-from-banking-documents-with-multimodal-models-using-llama-factory-on-amazon-sagemaker-hyperpod/), [NVIDIA](https://developer.nvidia.com/rtx/ai-toolkit), [Aliyun](https://help.aliyun.com/zh/pai/use-cases/fine-tune-a-llama-3-model-with-llama-factory), and more.

<div align="center">
  <a href="https://warp.dev/llama-factory">
    <img alt="Warp sponsorship" width="400" src="https://github.com/user-attachments/assets/ab8dd143-b0fd-4904-bdc5-dd7ecac94eae">
  </a>
  <p>
    <a href="https://warp.dev/llama-factory">Warp, the agentic terminal for developers</a>
    <br>
    Available for MacOS, Linux, & Windows
  </p>
  <img src="https://trendshift.io/api/badge/repositories/4535" alt="GitHub Trend">
</div>

<p>
  Join our <a href="assets/wechat.jpg">WeChat group</a>, <a href="assets/wechat_npu.jpg">NPU user group</a> or <a href="assets/wechat_alaya.png">Alaya NeW user group</a>.
</p>

\[ <a href="README_zh.md">中文</a> | English ]

**Fine-tuning a large language model can be easy as...**
<p align="center">
    <img src="https://github.com/user-attachments/assets/3991a3a8-4276-4d30-9cab-4cb0c4b9b99e" alt="Quick Start">
</p>

**Choose Your Path:**

*   [Documentation (WIP)](https://llamafactory.readthedocs.io/en/latest/)
*   [Documentation (AMD GPU)](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/fine_tune/llama_factory_llama3.html)
*   [Colab (free)](https://colab.research.google.com/drive/1eRTPn37ltBbYsISy9Aw2NuI2Aq5CQrD9?usp=sharing)
*   [Local Machine](#getting-started)
*   [PAI-DSW (free trial)](https://gallery.pai-ml.com/#/preview/deepLearning/nlp/llama_factory)
*   [Alaya NeW (cloud GPU deal)](https://docs.alayanew.com/docs/documents/useGuide/LLaMAFactory/mutiple/?utm_source=LLaMA-Factory)

> [!NOTE]
> Except for the above links, all other websites are unauthorized third-party websites. Please carefully use them.

## Table of Contents

*   [Features](#features)
*   [Supported Models](#supported-models)
*   [Supported Training Approaches](#supported-training-approaches)
*   [Provided Datasets](#provided-datasets)
*   [Requirements](#requirement)
*   [Getting Started](#getting-started)
    *   [Installation](#installation)
    *   [Data Preparation](#data-preparation)
    *   [Quickstart](#quickstart)
    *   [Fine-Tuning with LLaMA Board GUI](#fine-tuning-with-llama-board-gui-powered-by-gradio)
    *   [Build Docker](#build-docker)
    *   [Deploy with OpenAI-style API and vLLM](#deploy-with-openai-style-api-and-vllm)
    *   [Download from ModelScope Hub](#download-from-modelscope-hub)
    *   [Download from Modelers Hub](#download-from-modelers-hub)
    *   [Use W&B Logger](#use-wb-logger)
    *   [Use SwanLab Logger](#use-swanlab-logger)
*   [Projects Using LLaMA Factory](#projects-using-llama-factory)
*   [License](#license)
*   [Citation](#citation)
*   [Acknowledgement](#acknowledgement)

## Supported Models

A comprehensive list of supported models, including Baichuan, BLOOM, ChatGLM, DeepSeek, Falcon, Gemma, GLM, GPT-2, InternLM, Llama, Mistral, Phi, Qwen, Yi, and more, along with their respective templates. (See full list above)

## Supported Training Approaches

*   Pre-Training
*   Supervised Fine-Tuning
*   Reward Modeling
*   PPO Training
*   DPO Training
*   KTO Training
*   ORPO Training
*   SimPO Training

## Provided Datasets

Access a range of pre-training, supervised fine-tuning, and preference datasets, including examples and links to popular datasets. (See full list above)

## Requirements

Lists essential and optional dependencies with their minimum and recommended versions.

## Getting Started

### Installation

Detailed instructions on installing LLaMA Factory from source or using a Docker image, with specifics for different environments (Windows, Ascend NPU, etc.).

### Data Preparation

Guidance on preparing your dataset, including file format details and links to helpful resources.

### Quickstart

Simplified steps to quickly fine-tune, infer, and merge a model using the command-line interface (CLI).

### Fine-Tuning with LLaMA Board GUI (powered by Gradio)

Instructions on running the user-friendly web interface.

### Build Docker

Step-by-step guidance for building and running Docker containers for CUDA, Ascend NPU, and AMD ROCm users.

### Deploy with OpenAI-style API and vLLM

How to deploy a fine-tuned model with an OpenAI-compatible API using vLLM.

### Download from ModelScope Hub

Instructions for downloading models and datasets from the ModelScope Hub if you experience issues with Hugging Face.

### Download from Modelers Hub

Instructions for downloading models and datasets from the Modelers Hub.

### Use W&B Logger

Learn how to integrate Weights & Biases for experiment tracking.

### Use SwanLab Logger

Learn how to integrate SwanLab for experiment tracking.

## Projects Using LLaMA Factory

A list of projects that have successfully utilized LLaMA Factory, including links to related papers and resources.

## License

The project is available under the Apache-2.0 License.

## Citation

Instructions for citing the project if you find it helpful.

## Acknowledgement

Credits and thanks to the projects and individuals that contributed to LLaMA Factory.