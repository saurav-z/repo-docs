[![LLaMA Factory](assets/logo.png)](https://github.com/hiyouga/LLaMA-Factory)

## LLaMA Factory: Fine-tune any Large Language Model with Ease

**Unleash the power of fine-tuning!** LLaMA Factory provides a streamlined, zero-code solution for fine-tuning over 100+ large language models (LLMs), empowering you to customize them for specific tasks and domains.

[![GitHub Repo stars](https://img.shields.io/github/stars/hiyouga/LLaMA-Factory?style=social)](https://github.com/hiyouga/LLaMA-Factory/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/hiyouga/LLaMA-Factory)](https://github.com/hiyouga/LLaMA-Factory/commits/main)
[![GitHub contributors](https://img.shields.io/github/contributors/hiyouga/LLaMA-Factory?color=orange)](https://github.com/hiyouga/LLaMA-Factory/graphs/contributors)
[![GitHub workflow](https://github.com/hiyouga/LLaMA-Factory/actions/workflows/tests.yml/badge.svg)](https://github.com/hiyouga/LLaMA-Factory/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/llamafactory)](https://pypi.org/project/llamafactory/)
[![Citation](https://img.shields.io/badge/citation-760-green)](https://scholar.google.com/scholar?cites=12620864006390196564)
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

| <div style="text-align: center;"><a href="https://warp.dev/llama-factory"><img alt="Warp sponsorship" width="400" src="assets/warp.jpg"></a><br><a href="https://warp.dev/llama-factory" style="font-size:larger;">Warp, the agentic terminal for developers</a><br><a href="https://warp.dev/llama-factory">Available for MacOS, Linux, & Windows</a> | <a href="https://serpapi.com"><img alt="SerpAPI sponsorship" width="250" src="assets/serpapi.svg"> </a> |
| ---- | ---- |

----

### Fine-tune 100+ LLMs with zero-code [CLI](#quickstart) and [Web UI](#fine-tuning-with-llama-board-gui-powered-by-gradio)

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
  - [Deployment](#deployment)
- [Further Resources](#further-resources)
  - [Blogs](#blogs)
  - [Changelog](#changelog)
  - [Projects Using LLaMA Factory](#projects-using-llama-factory)
  - [License](#license)
  - [Citation](#citation)
  - [Acknowledgement](#acknowledgement)

## Key Features

*   **Extensive Model Support:** Fine-tune a wide range of LLMs, including LLaMA, Mistral, Qwen, Gemma, and many more (over 100+ models).
*   **Versatile Training Methods:** Utilize various methods like (Continuous) pre-training, (multimodal) supervised fine-tuning, reward modeling, PPO, DPO, KTO, ORPO, and SimPO.
*   **Efficient Training Techniques:** Leverage techniques such as full-tuning, freeze-tuning, LoRA, and QLoRA (2/3/4/5/6/8-bit) for efficient training.
*   **Advanced Algorithms & Optimization:** Integrate advanced algorithms like GaLore, BAdam, APOLLO, and Muon, along with practical tricks like FlashAttention-2, Unsloth, RoPE scaling, and NEFTune.
*   **Comprehensive Task Support:** Address diverse tasks including multi-turn dialogue, tool using, image understanding, and audio understanding.
*   **Monitoring and Experimentation:** Track progress with tools such as LlamaBoard, TensorBoard, Wandb, and SwanLab.
*   **Optimized Inference:** Benefit from faster inference with an OpenAI-style API, Gradio UI, and CLI integration with vLLM or SGLang.
*   **Day-N Support:** Get support for fine-tuning cutting-edge models.

## Supported Models

*   [Model List](#supported-models) (See above)

## Getting Started

### Installation

*   [Installation Instructions](#installation) (See above)

### Quickstart

Fine-tune, infer, and merge a model using the following commands:

```bash
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
llamafactory-cli chat examples/inference/llama3_lora_sft.yaml
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
```

See [examples/README.md](examples/README.md) for advanced usage.

### Fine-Tuning with LLaMA Board GUI

Launch the user-friendly web interface:

```bash
llamafactory-cli webui
```

### Deployment

*   [Deploy with OpenAI-style API and vLLM](#deploy-with-openai-style-api-and-vllm)
*   [Build Docker](#build-docker)

## Further Resources

### Blogs

*   [Fine-tune GPT-OSS for Role-Playing using LLaMA-Factory](https://docs.llamafactory.com.cn/docs/documents/best-practice/gptoss/?utm_source=LLaMA-Factory) (Chinese)
*   [Fine-tune Llama3.1-70B for Medical Diagnosis using LLaMA-Factory](https://docs.alayanew.com/docs/documents/bestPractice/bigModel/llama70B/?utm_source=LLaMA-Factory) (Chinese)
*   [How Apoidea Group enhances visual information extraction from banking documents with multimodal models using LLaMA-Factory on Amazon SageMaker HyperPod](https://aws.amazon.com/cn/blogs/machine-learning/how-apoidea-group-enhances-visual-information-extraction-from-banking-documents-with-multimodal-models-using-llama-factory-on-amazon-sagemaker-hyperpod/) (English)
*   [Easy Dataset × LLaMA Factory: Enabling LLMs to Efficiently Learn Domain Knowledge](https://buaa-act.feishu.cn/wiki/GVzlwYcRFiR8OLkHbL6cQpYin7g) (English)

### Changelog

*   [Changelog](#changelog) (See above)

### Projects Using LLaMA Factory

*   [Projects using LLaMA Factory](#projects-using-llama-factory) (See above)

### License

*   [License](#license) (See above)

### Citation

*   [Citation](#citation) (See above)

### Acknowledgement

*   [Acknowledgement](#acknowledgement) (See above)

[**Original Repository:**](https://github.com/hiyouga/LLaMA-Factory)