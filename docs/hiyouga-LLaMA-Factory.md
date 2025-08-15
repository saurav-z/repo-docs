<!--  LLaMA Factory: Fine-tune any large language model with ease! -->
# LLaMA Factory: Your All-in-One Solution for LLM Fine-tuning

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

### Easily fine-tune 100+ large language models with zero-code [CLI](#quickstart) and [Web UI](#fine-tuning-with-llama-board-gui-powered-by-gradio)

![GitHub Trend](https://trendshift.io/api/badge/repositories/4535)

</div>

👋 Join our [WeChat group](assets/wechat.jpg), [NPU user group](assets/wechat_npu.jpg) or [Alaya NeW user group](assets/wechat_alaya.png).

\[ [English](README.md) | [中文](README_zh.md) ]

**Fine-tuning a large language model can be easy as...**

[![Fine-tuning LLM in LLaMA-Factory](https://github.com/user-attachments/assets/3991a3a8-4276-4d30-9cab-4cb0c4b9b99e)](https://github.com/hiyouga/LLaMA-Factory)

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
- [Supported Training Approaches](#supported-training-approaches)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Quickstart](#quickstart)
  - [Fine-tuning with LLaMA Board GUI](#fine-tuning-with-llama-board-gui-powered-by-gradio)
  - [Deployment](#deploy-with-openai-style-api-and-vllm)
- [More Information](#more-information)
  - [Blogs](#blogs)
  - [Changelog](#changelog)
  - [Provided Datasets](#provided-datasets)
  - [Projects using LLaMA Factory](#projects-using-llama-factory)
  - [License](#license)
  - [Citation](#citation)
  - [Acknowledgement](#acknowledgement)

## Key Features

*   **Versatile Model Support:** Fine-tune a wide range of LLMs, including LLaMA, Mistral, Qwen, Gemma, and many more.
*   **Flexible Training Approaches:** Leverage various methods such as (Continuous) pre-training, (multimodal) supervised fine-tuning, reward modeling, PPO, DPO, and more.
*   **Efficient Training Techniques:** Utilize 16-bit full-tuning, freeze-tuning, LoRA, and QLoRA for efficient resource utilization.
*   **Advanced Algorithms:** Access cutting-edge optimization algorithms like GaLore, BAdam, APOLLO, and Muon.
*   **Practical Enhancements:** Incorporate FlashAttention-2, Unsloth, and other tricks for improved performance.
*   **Wide Range of Tasks:** Tackle diverse tasks like multi-turn dialogue, tool usage, and image understanding.
*   **Comprehensive Monitoring:** Monitor experiments with LlamaBoard, TensorBoard, Wandb, MLflow, and SwanLab.
*   **Accelerated Inference:** Deploy with OpenAI-style API, Gradio UI, and CLI using vLLM or SGLang.

## Supported Models

Llama Factory supports a wide array of models. For a comprehensive list, please refer to the [Supported Models](#supported-models) section.

## Supported Training Approaches

Llama Factory offers a variety of training methods.  See the [Supported Training Approaches](#supported-training-approaches) section for more details.

## Getting Started

### Installation

Detailed installation instructions, including instructions for specific environments (Windows, Ascend NPU, and others) are available in the [Installation](#installation) section.

### Quickstart

Get up and running quickly with these three commands for LoRA fine-tuning, inference, and merging:

```bash
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
llamafactory-cli chat examples/inference/llama3_lora_sft.yaml
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
```

### Fine-tuning with LLaMA Board GUI

Launch the user-friendly web interface for training, evaluation, and inference:

```bash
llamafactory-cli webui
```

### Deployment

Deploy your fine-tuned models with the OpenAI-style API and vLLM for fast inference:

```bash
API_PORT=8000 llamafactory-cli api examples/inference/llama3.yaml infer_backend=vllm vllm_enforce_eager=true
```

## More Information

### Blogs

*   Fine-tune GPT-OSS for Role-Playing using LLaMA-Factory (Chinese)
*   Fine-tune Llama3.1-70B for Medical Diagnosis using LLaMA-Factory (Chinese)
*   A One-Stop Code-Free Model Reinforcement Learning and Deployment Platform based on LLaMA-Factory and EasyR1 (Chinese)
*   How Apoidea Group enhances visual information extraction from banking documents with multimodal models using LLaMA-Factory on Amazon SageMaker HyperPod (English)
*   Easy Dataset × LLaMA Factory: Enabling LLMs to Efficiently Learn Domain Knowledge (English)

### Changelog

Stay up-to-date with the latest features and updates in the [Changelog](#changelog) section.

### Provided Datasets

Explore the pre-training, supervised fine-tuning, and preference datasets available in the [Provided Datasets](#provided-datasets) section.

### Projects using LLaMA Factory

Discover how others are utilizing LLaMA Factory in the [Projects using LLaMA Factory](#projects-using-llama-factory) section.

### License

The project is licensed under the [Apache-2.0 License](LICENSE).

### Citation

If you use this project, please cite it using the information in the [Citation](#citation) section.

### Acknowledgement

This project benefits from the contributions of [PEFT](https://github.com/huggingface/peft), [TRL](https://github.com/huggingface/trl), [QLoRA](https://github.com/artidoro/qlora) and [FastChat](https://github.com/lm-sys/FastChat). Thanks for their wonderful works.