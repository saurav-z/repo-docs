![LLaMA Factory](assets/logo.png)

[![GitHub Repo stars](https://img.shields.io/github/stars/hiyouga/LLaMA-Factory?style=social)](https://github.com/hiyouga/LLaMA-Factory/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/hiyouga/LLaMA-Factory)](https://github.com/hiyouga/LLaMA-Factory/commits/main)
[![GitHub contributors](https://img.shields.io/github/contributors/hiyouga/LLaMA-Factory?color=orange)](https://github.com/hiyouga/LLaMA-Factory/graphs/contributors)
[![GitHub workflow](https://github.com/hiyouga/LLaMA-Factory/actions/workflows/tests.yml/badge.svg)](https://github.com/hiyouga/LLaMA-Factory/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/llamafactory)](https://pypi.org/project/llamafactory/)
[![Citation](https://img.shields.io/badge/citation-730-green)](https://scholar.google.com/scholar?cites=12620864006390196564)
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

## LLaMA Factory: Fine-tune LLMs with Ease and Efficiency

LLaMA Factory is a powerful and user-friendly toolkit designed for **fine-tuning over 100 large language models (LLMs)** with various training approaches, providing a seamless experience for both beginners and experts.  Find the original repo [here](https://github.com/hiyouga/LLaMA-Factory).

### Key Features

*   **Extensive Model Support:** Fine-tune a wide variety of models, including LLaMA, Mistral, Mixtral, Qwen, Gemma, and many more.
*   **Versatile Training Approaches:** Utilize various methods such as pre-training, supervised fine-tuning, reward modeling, PPO, DPO, and more.
*   **Efficient Training Techniques:** Leverage 16-bit full-tuning, freeze-tuning, LoRA, and QLoRA to optimize resource utilization.
*   **Advanced Algorithms:** Explore cutting-edge algorithms like GaLore, BAdam, APOLLO, and PiSSA.
*   **Practical Enhancements:** Integrate with FlashAttention-2, Unsloth, and Liger Kernel for improved performance.
*   **Broad Task Applicability:** Address multi-turn dialogue, tool using, image understanding, and other complex tasks.
*   **Comprehensive Monitoring:** Monitor experiments using LlamaBoard, TensorBoard, Wandb, and SwanLab.
*   **Faster Inference:** Deploy fine-tuned models with OpenAI-style APIs and Gradio UI using vLLM or SGLang.

### Table of Contents

-   [Key Features](#key-features)
-   [Supported Models](#supported-models)
-   [Supported Training Approaches](#supported-training-approaches)
-   [Provided Datasets](#provided-datasets)
-   [Requirements](#requirement)
-   [Getting Started](#getting-started)
    -   [Installation](#installation)
    -   [Data Preparation](#data-preparation)
    -   [Quickstart](#quickstart)
    -   [Fine-Tuning with LLaMA Board GUI](#fine-tuning-with-llama-board-gui-powered-by-gradio)
    -   [Build Docker](#build-docker)
    -   [Deploy with OpenAI-style API and vLLM](#deploy-with-openai-style-api-and-vllm)
    -   [Download from ModelScope Hub](#download-from-modelscope-hub)
    -   [Download from Modelers Hub](#download-from-modelers-hub)
    -   [Use W&B Logger](#use-wb-logger)
    -   [Use SwanLab Logger](#use-swanlab-logger)
-   [Projects using LLaMA Factory](#projects-using-llama-factory)
-   [License](#license)
-   [Citation](#citation)
-   [Acknowledgement](#acknowledgement)
-   [Star History](#star-history)

### Supported Models

*   A comprehensive list of supported models can be found [here](#supported-models).  The list includes Baichuan, BLOOM, ChatGLM, DeepSeek, Falcon, Gemma, Llama, Mixtral, Qwen, and more.

### Supported Training Approaches

*   LLaMA Factory supports a variety of training approaches. See a breakdown [here](#supported-training-approaches).

### Provided Datasets

*   The toolkit provides a selection of pre-training, supervised fine-tuning, and preference datasets.  Explore the available datasets [here](#provided-datasets).

### Requirements

*   Review the system requirements to ensure compatibility.  See the [requirements](#requirement) section for details on required Python, PyTorch, transformers, and other dependencies.

### Getting Started

#### Installation

*   Detailed installation instructions, including source and Docker options, can be found [here](#installation).

#### Data Preparation

*   Learn how to prepare your data for fine-tuning. Format details are available [here](#data-preparation).

#### Quickstart

*   Get up and running quickly with the 3-command LoRA fine-tuning, inference, and merging example:

```bash
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
llamafactory-cli chat examples/inference/llama3_lora_sft.yaml
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
```

#### Fine-Tuning with LLaMA Board GUI

*   Use the Gradio-powered Web UI:

```bash
llamafactory-cli webui
```

#### Build Docker

*   Instructions for building Docker images are available [here](#build-docker).

#### Deploy with OpenAI-style API and vLLM

*   Deploy fine-tuned models:

```bash
API_PORT=8000 llamafactory-cli api examples/inference/llama3.yaml infer_backend=vllm vllm_enforce_eager=true
```

#### Download from ModelScope Hub

*   Utilize ModelScope for model and dataset downloads.  Instructions can be found [here](#download-from-modelscope-hub).

#### Download from Modelers Hub

*   Use Modelers Hub for model and dataset downloads.  Instructions can be found [here](#download-from-modelers-hub).

#### Use W&B Logger

*   Integrate with Weights & Biases for experiment tracking.  See the setup instructions [here](#use-wb-logger).

#### Use SwanLab Logger

*   Use SwanLab for experiment tracking.  See the setup instructions [here](#use-swanlab-logger).

### Projects using LLaMA Factory

*   A list of projects using LLaMA Factory is provided [here](#projects-using-llama-factory), with links to relevant research.

### License

*   This project is licensed under the [Apache-2.0 License](LICENSE).

### Citation

*   If you find this work helpful, please cite the paper as shown [here](#citation).

### Acknowledgement

*   This project leverages the work of PEFT, TRL, QLoRA, and FastChat. See [here](#acknowledgement) for details.

### Star History
* See the growth of the project [here](#star-history).