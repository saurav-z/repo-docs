# LLaMA Factory: Fine-tune Any Large Language Model with Ease

Fine-tune over 100+ LLMs with zero-code via a powerful CLI and intuitive web interface, unlocking the potential of open-source language models. 🚀 [Original Repo](https://github.com/hiyouga/LLaMA-Factory)

[![GitHub Repo stars](https://img.shields.io/github/stars/hiyouga/LLaMA-Factory?style=social)](https://github.com/hiyouga/LLaMA-Factory/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/hiyouga/LLaMA-Factory)](https://github.com/hiyouga/LLaMA-Factory/commits/main)
[![GitHub contributors](https://img.shields.io/github/contributors/hiyouga/LLaMA-Factory?color=orange)](https://github.com/hiyouga/LLaMA-Factory/graphs/contributors)
[![GitHub workflow](https://github.com/hiyouga/LLaMA-Factory/actions/workflows/tests.yml/badge.svg)](https://github.com/hiyouga/LLaMA-Factory/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/llamafactory)](https://pypi.org/project/llamafactory/)
[![Citation](https://img.shields.io/badge/citation-651-green)](https://scholar.google.com/scholar?cites=12620864006390196564)
[![Docker Pulls](https://img.shields.io/docker/pulls/hiyouga/llamafactory)](https://hub.docker.com/r/hiyouga/llamafactory/tags)

[![Twitter](https://img.shields.io/twitter/follow/llamafactory_ai)](https://twitter.com/llamafactory_ai)
[![Discord](https://dcbadge.vercel.app/api/server/rKfvV9r9FK?compact=true&style=flat)](https://discord.gg/rKfvV9r9FK)
[![GitCode](https://gitcode.com/zhengyaowei/LLaMA-Factory/star/badge.svg)](https://gitcode.com/zhengyaowei/LLaMA-Factory)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1eRTPn37ltBbYsISy9Aw2NuI2Aq5CQrD9?usp=sharing)
[![Open in DSW](https://gallery.pai-ml.com/assets/open-in-dsw.svg)](https://gallery.pai-ml.com/#/preview/deepLearning/nlp/llama_factory)
[![Open in Alaya](assets/alaya_new.svg)](https://docs.alayanew.com/docs/documents/newActivities/llamafactory/?utm_source=LLaMA-Factory)
[![Open in Spaces](https://img.shields.io/badge/🤗-Open%20in%20Spaces-blue)](https://huggingface.co/spaces/hiyouga/LLaMA-Board)
[![Open in Studios](https://img.shields.io/badge/ModelScope-Open%20in%20Studios-blue)](https://modelscope.cn/studios/hiyouga/LLaMA-Board)
[![Open in Novita](https://img.shields.io/badge/Novita-Deploy%20Template-blue)](https://novita.ai/templates-library/105981?sharer=88115474-394e-4bda-968e-b88e123d0c47)

### Used by [Amazon](https://aws.amazon.com/cn/blogs/machine-learning/how-apoidea-group-enhances-visual-information-extraction-from-banking-documents-with-multimodal-models-using-llama-factory-on-amazon-sagemaker-hyperpod/), [NVIDIA](https://developer.nvidia.com/rtx/ai-toolkit), [Aliyun](https://help.aliyun.com/zh/pai/use-cases/fine-tune-a-llama-3-model-with-llama-factory), etc.

<div align="center" markdown="1">

### Supporters ❤️

<a href="https://warp.dev/llama-factory">
    <img alt="Warp sponsorship" width="400" src="https://github.com/user-attachments/assets/ab8dd143-b0fd-4904-bdc5-dd7ecac94eae">
</a>

#### [Warp, the agentic terminal for developers](https://warp.dev/llama-factory)

[Available for MacOS, Linux, & Windows](https://warp.dev/llama-factory)

----

### Easily fine-tune 100+ large language models with zero-code [CLI](#quickstart) and [Web UI](#fine-tuning-with-llama-board-gui-powered-by-gradio)

![GitHub Trend](https://trendshift.io/api/badge/repositories/4535)

</div>

👋 Join our [WeChat group](assets/wechat.jpg), [NPU user group](assets/wechat_npu.jpg) or [Alaya NeW user group](assets/wechat_alaya.png).

\[ English | [中文](README_zh.md) \]

**Fine-tuning a large language model can be easy as...**

https://github.com/user-attachments/assets/3991a3a8-4276-4d30-9cab-4cb0c4b9b99e

Choose your path:

- **Documentation (WIP)**: https://llamafactory.readthedocs.io/en/latest/
- **Documentation (AMD GPU)**: https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/fine_tune/llama_factory_llama3.html
- **Colab (free)**: https://colab.research.google.com/drive/1eRTPn37ltBbYsISy9Aw2NuI2Aq5CQrD9?usp=sharing
- **Local machine**: Please refer to [usage](#getting-started)
- **PAI-DSW (free trial)**: https://gallery.pai-ml.com/#/preview/deepLearning/nlp/llama_factory
- **Alaya NeW (cloud GPU deal)**: https://docs.alayanew.com/docs/documents/useGuide/LLaMAFactory/mutiple/?utm_source=LLaMA-Factory

> [!NOTE]
> Except for the above links, all other websites are unauthorized third-party websites. Please carefully use them.

## Table of Contents

- [Key Features](#key-features)
- [Supported Models](#supported-models)
- [Getting Started](#getting-started)
    - [Installation](#installation)
    - [Quickstart](#quickstart)
    - [Fine-Tuning with LLaMA Board GUI](#fine-tuning-with-llama-board-gui-powered-by-gradio)
- [More Information](#more-information)
    - [Supported Training Approaches](#supported-training-approaches)
    - [Provided Datasets](#provided-datasets)
    - [Requirement](#requirement)
    - [Getting Started Details](#getting-started)
    - [Projects using LLaMA Factory](#projects-using-llama-factory)
    - [License](#license)
    - [Citation](#citation)
    - [Acknowledgement](#acknowledgement)

## Key Features

*   **Extensive Model Support:** Fine-tune LLaMA, Mistral, Mixtral, Qwen, DeepSeek, Gemma, and many more.
*   **Flexible Training:** Utilize various methods like (Continuous) pre-training, (multimodal) supervised fine-tuning, PPO, DPO, and more.
*   **Efficiency & Optimization:** Leverage 16-bit, freeze-tuning, LoRA, and QLoRA (2/3/4/5/6/8-bit) for efficient training.
*   **Advanced Algorithms:** Integrate cutting-edge algorithms like GaLore, BAdam, APOLLO, and Muon for improved performance.
*   **Practical Enhancements:** Includes FlashAttention-2, Unsloth, RoPE scaling, and NEFTune for faster training and inference.
*   **Versatile Tasks:** Adapt to multi-turn dialogue, tool usage, image understanding, and other diverse applications.
*   **Comprehensive Monitoring:** Utilize experiment monitors like LlamaBoard, TensorBoard, and Wandb for tracking progress.
*   **Faster Inference:** Offers OpenAI-style API, Gradio UI, and CLI with vLLM or SGLang for optimized inference.

## Supported Models

See the table below or the [Supported Models](#supported-models) section in the original README.

## Getting Started

### Installation

Detailed installation instructions can be found [here](#installation). Briefly:

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
```

Or, install from Docker image:
```bash
docker run -it --rm --gpus=all --ipc=host hiyouga/llamafactory:latest
```

### Quickstart

Fine-tune, infer, and merge a Llama3-8B-Instruct model with these simple commands:

```bash
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
llamafactory-cli chat examples/inference/llama3_lora_sft.yaml
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
```

### Fine-Tuning with LLaMA Board GUI

Launch the user-friendly web interface for easy training:

```bash
llamafactory-cli webui
```

## More Information

### Supported Training Approaches

A detailed list is available [here](#supported-training-approaches).

### Provided Datasets

See the [Provided Datasets](#provided-datasets) section.

### Requirement

See the [Requirement](#requirement) section.

### Getting Started Details

Find detailed steps on:

*   [Data Preparation](#data-preparation)
*   [Build Docker](#build-docker)
*   [Deploy with OpenAI-style API and vLLM](#deploy-with-openai-style-api-and-vllm)
*   [Download from ModelScope Hub](#download-from-modelscope-hub)
*   [Download from Modelers Hub](#download-from-modelers-hub)
*   [Use W&B Logger](#use-wb-logger)
*   [Use SwanLab Logger](#use-swanlab-logger)

in the original README.

### Projects using LLaMA Factory

Check out projects using LLaMA Factory [here](#projects-using-llama-factory).

### License

This project is licensed under the [Apache-2.0 License](LICENSE).

### Citation

If you find this project useful, please cite the paper:

```bibtex
@inproceedings{zheng2024llamafactory,
  title={LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models},
  author={Yaowei Zheng and Richong Zhang and Junhao Zhang and Yanhan Ye and Zheyan Luo and Zhangchi Feng and Yongqiang Ma},
  booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations)},
  address={Bangkok, Thailand},
  publisher={Association for Computational Linguistics},
  year={2024},
  url={http://arxiv.org/abs/2403.13372}
}
```

### Acknowledgement

See the [Acknowledgement](#acknowledgement) section.