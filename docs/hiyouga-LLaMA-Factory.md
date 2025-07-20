# LLaMA Factory: Fine-tune Any LLM with Ease 🚀

**Fine-tune and deploy over 100+ large language models with zero-code options using this versatile and powerful framework.**

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

### Fine-tune 100+ LLMs with ease, using zero-code CLI and Web UI.

![GitHub Trend](https://trendshift.io/api/badge/repositories/4535)

</div>

👋 Join our [WeChat group](assets/wechat.jpg), [NPU user group](assets/wechat_npu.jpg) or [Alaya NeW user group](assets/wechat_alaya.png).

\[ English | [中文](README_zh.md) \]

**Transform your LLM with a few clicks: fine-tuning is now easier than ever!**

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

## Key Features 🌟

*   **Extensive Model Support:** Train LLaMA, Mistral, Mixtral, Qwen, DeepSeek, Gemma, and more.
*   **Versatile Training Methods:**  Including (continuous) pre-training, (multimodal) supervised fine-tuning, reward modeling, PPO, DPO, and others.
*   **Flexible Training Strategies:** Utilize 16-bit full-tuning, freeze-tuning, LoRA, QLoRA, and other quantization techniques for efficient use of resources.
*   **Advanced Optimization Algorithms:** Leverages cutting-edge algorithms like GaLore, BAdam, APOLLO, and Muon for improved performance.
*   **Practical Enhancements:** Integrates techniques such as FlashAttention-2, Unsloth, and Liger Kernel to enhance training speed and efficiency.
*   **Broad Application:** Supports diverse tasks, including multi-turn dialogue, tool usage, image understanding, and more.
*   **Comprehensive Monitoring:** Offers experiment tracking with tools such as LlamaBoard, TensorBoard, Wandb, and SwanLab.
*   **Fast Inference:** Provides an OpenAI-style API, Gradio UI, and CLI with vLLM or SGLang backend for faster inference.

### Day-N Support for Fine-Tuning Cutting-Edge Models

| Support Date | Model Name                                                           |
| ------------ | -------------------------------------------------------------------- |
| Day 0        | Qwen3 / Qwen2.5-VL / Gemma 3 / GLM-4.1V / InternLM 3 / MiniCPM-o-2.6 |
| Day 1        | Llama 3 / GLM-4 / Mistral Small / PaliGemma2 / Llama 4               |

## Blogs & Resources 📚

*   A curated list of blogs and articles demonstrating the power and versatility of LLaMA Factory, including those by Amazon, NVIDIA, and Aliyun, and more.

## Changelog

Provides up-to-date information on the latest features and supported models.

[See the Full Changelog](#changelog)

## Supported Models 🤖

Comprehensive support for various Large Language Models.

[See Supported Models](#supported-models)

## Supported Training Approaches 🚀

A table detailing the training approaches.

[See Supported Training Approaches](#supported-training-approaches)

## Provided Datasets 💾

Access a wide range of pre-built datasets.

[See Provided Datasets](#provided-datasets)

## Requirements 🛠️

Lists the necessary software and hardware requirements for running LLaMA Factory.

[See Requirement](#requirement)

## Getting Started 🚀

A comprehensive guide to help you start using LLaMA Factory:

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

## Projects using LLaMA Factory 💡

A list of projects that use LLaMA Factory.

[See Projects using LLaMA Factory](#projects-using-llama-factory)

## License 📜

Details regarding the licensing of LLaMA Factory.

[See License](#license)

## Citation ✍️

Instructions on how to cite LLaMA Factory.

[See Citation](#citation)

## Acknowledgement 🙏

Acknowledgements for the contributions and resources that have made LLaMA Factory possible.

[See Acknowledgement](#acknowledgement)

---

**[Back to Top](#)** |  **[GitHub Repository](https://github.com/hiyouga/LLaMA-Factory)**