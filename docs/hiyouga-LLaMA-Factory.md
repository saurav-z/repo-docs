<div align="center">
  <img src="assets/logo.png" alt="LLaMA Factory Logo" width="200">
  <h1>LLaMA Factory: The Ultimate Toolkit for Fine-Tuning Large Language Models</h1>

  [![GitHub Repo stars](https://img.shields.io/github/stars/hiyouga/LLaMA-Factory?style=social)](https://github.com/hiyouga/LLaMA-Factory/stargazers)
  [![GitHub last commit](https://img.shields.io/github/last-commit/hiyouga/LLaMA-Factory)](https://github.com/hiyouga/LLaMA-Factory/commits/main)
  [![GitHub contributors](https://img.shields.io/github/contributors/hiyouga/LLaMA-Factory?color=orange)](https://github.com/hiyouga/LLaMA-Factory/graphs/contributors)
  [![GitHub workflow](https://github.com/hiyouga/LLaMA-Factory/actions/workflows/tests.yml/badge.svg)](https://github.com/hiyouga/LLaMA-Factory/actions/workflows/tests.yml)
  [![PyPI](https://img.shields.io/pypi/v/llamafactory)](https://pypi.org/project/llamafactory/)
  [![Citation](https://img.shields.io/badge/citation-818-green)](https://scholar.google.com/scholar?cites=12620864006390196564)
  [![Docker Pulls](https://img.shields.io/docker/pulls/hiyouga/llamafactory)](https://hub.docker.com/r/hiyouga/llamafactory/tags)

  [![Twitter](https://img.shields.io/twitter/follow/llamafactory_ai)](https://twitter.com/llamafactory_ai)
  [![Discord](https://dcbadge.vercel.app/api/server/rKfvV9r9FK?compact=true&style=flat)](https://discord.gg/rKfvV9r9FK)

  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1eRTPn37ltBbYsISy9Aw2NuI2Aq5CQrD9?usp=sharing)
  [![Open in DSW](https://gallery.pai-ml.com/assets/open-in-dsw.svg)](https://gallery.pai-ml.com/#/preview/deepLearning/nlp/llama_factory)
  [![Open in Alaya](assets/alaya_new.svg)](https://docs.alayanew.com/docs/documents/newActivities/llamafactory/?utm_source=LLaMA-Factory)
  [![Open in Spaces](https://img.shields.io/badge/🤗-Open%20in%20Spaces-blue)](https://huggingface.co/spaces/hiyouga/LLaMA-Board)
  [![Open in Studios](https://img.shields.io/badge/ModelScope-Open%20in%20Studios-blue)](https://modelscope.cn/studios/hiyouga/LLaMA-Board)
  [![Open in Novita](https://img.shields.io/badge/Novita-Deploy%20Template-blue)](https://novita.ai/templates-library/105981?sharer=88115474-394e-4bda-968e-b88e123d0c47)
</div>

**LLaMA Factory empowers anyone to easily fine-tune 100+ state-of-the-art large language models with minimal coding, using a zero-code [CLI](#quickstart) and [Web UI](#fine-tuning-with-llama-board-gui-powered-by-gradio).**

Used by [Amazon](https://aws.amazon.com/cn/blogs/machine-learning/how-apoidea-group-enhances-visual-information-extraction-from-banking-documents-with-multimodal-models-using-llama-factory-on-amazon-sagemaker-hyperpod/), [NVIDIA](https://developer.nvidia.com/rtx/ai-toolkit), [Aliyun](https://help.aliyun.com/zh/pai/use-cases/fine-tune-a-llama-3-model-with-llama-factory), and many more.

<div align="center" markdown="1">

### Supporters ❤️

| <div style="text-align: center;"><a href="https://warp.dev/llama-factory"><img alt="Warp sponsorship" width="400" src="assets/warp.jpg"></a><br><a href="https://warp.dev/llama-factory" style="font-size:larger;">Warp, the agentic terminal for developers</a><br><a href="https://warp.dev/llama-factory">Available for MacOS, Linux, & Windows</a> | <a href="https://serpapi.com"><img alt="SerpAPI sponsorship" width="250" src="assets/serpapi.svg"> </a> |
| ---- | ---- |

----

![GitHub Trend](https://trendshift.io/api/badge/repositories/4535)

</div>

👋 Join our [WeChat group](assets/wechat.jpg), [NPU user group](assets/wechat_npu.jpg) or [Alaya NeW user group](assets/wechat_alaya.png).

\[ English | [中文](README_zh.md) ]

> **Fine-tuning your LLM has never been easier.**

<img src="https://github.com/user-attachments/assets/3991a3a8-4276-4d30-9cab-4cb0c4b9b99e" alt="Example Usage" />

Choose your path:

- **Documentation (WIP)**: https://llamafactory.readthedocs.io/en/latest/
- **Documentation (AMD GPU)**: https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/fine_tune/llama_factory_llama3.html
- **Colab (free)**: https://colab.research.google.com/drive/1eRTPn37ltBbYsISy9Aw2NuI2Aq5CQrD9?usp=sharing
- **Local machine**: Please refer to [usage](#getting-started)
- **PAI-DSW (free trial)**: https://gallery.pai-ml.com/#/preview/deepLearning/nlp/llama_factory
- **Alaya NeW (cloud GPU deal)**: https://docs.alayanew.com/docs/documents/useGuide/LLaMAFactory/mutiple/?utm_source=LLaMA-Factory

> [!NOTE]
> Except for the above links, all other websites are unauthorized third-party websites. Please carefully use them.

## Key Features

*   **Extensive Model Support:** Fine-tune a wide range of models, including LLaMA, Mistral, Qwen, Gemma, and more (see [Supported Models](#supported-models)).
*   **Versatile Training Methods:** Implement various training approaches, including (continuous) pre-training, supervised fine-tuning, reward modeling, PPO, DPO, and more (see [Supported Training Approaches](#supported-training-approaches)).
*   **Efficient Training Techniques:** Leverage advanced optimization methods like LoRA, QLoRA, and other efficient parameter tuning strategies.
*   **Advanced Algorithms:** Includes cutting-edge algorithms like GaLore, BAdam, APOLLO, Adam-mini, Muon, OFT, DoRA, LongLoRA, LLaMA Pro, Mixture-of-Depths, LoRA+, LoftQ, and PiSSA.
*   **Practical Enhancements:** Integrates FlashAttention-2, Unsloth, Liger Kernel, RoPE scaling, NEFTune, and rsLoRA for improved performance.
*   **Broad Task Compatibility:** Suitable for diverse tasks, including multi-turn dialogue, tool usage, image understanding, visual grounding, and more.
*   **Comprehensive Monitoring:** Utilize various experiment monitoring tools such as LlamaBoard, TensorBoard, Wandb, MLflow, and SwanLab.
*   **Optimized Inference:** Offers OpenAI-style API, Gradio UI, and CLI with vLLM or SGLang for faster inference.

## Table of Contents

- [Key Features](#key-features)
- [Blogs](#blogs)
- [Changelog](#changelog)
- [Supported Models](#supported-models)
- [Supported Training Approaches](#supported-training-approaches)
- [Provided Datasets](#provided-datasets)
- [Requirement](#requirement)
- [Getting Started](#getting-started)
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

## Blogs

*   [Fine-tune GPT-OSS for Role-Playing using LLaMA-Factory](https://docs.llamafactory.com.cn/docs/documents/best-practice/gptroleplay/?utm_source=LLaMA-Factory) (Chinese)
*   [Fine-tune Llama3.1-70B for Medical Diagnosis using LLaMA-Factory](https://docs.alayanew.com/docs/documents/bestPractice/bigModel/llama70B/?utm_source=LLaMA-Factory) (Chinese)
*   [A One-Stop Code-Free Model Reinforcement Learning and Deployment Platform based on LLaMA-Factory and EasyR1](https://aws.amazon.com/cn/blogs/china/building-llm-model-hub-based-on-llamafactory-and-easyr1/) (Chinese)
*   [How Apoidea Group enhances visual information extraction from banking documents with multimodal models using LLaMA-Factory on Amazon SageMaker HyperPod](https://aws.amazon.com/cn/blogs/machine-learning/how-apoidea-group-enhances-visual-information-extraction-from-banking-documents-with-multimodal-models-using-llama-factory-on-amazon-sagemaker-hyperpod/) (English)
*   [Easy Dataset × LLaMA Factory: Enabling LLMs to Efficiently Learn Domain Knowledge](https://buaa-act.feishu.cn/wiki/GVzlwYcRFiR8OLkHbL6cQpYin7g) (English)

<details><summary>All Blogs</summary>

*   [Fine-tune Qwen2.5-VL for Autonomous Driving using LLaMA-Factory](https://docs.alayanew.com/docs/documents/useGuide/LLaMAFactory/mutiple/?utm_source=LLaMA-Factory) (Chinese)
*   [LLaMA Factory: Fine-tuning the DeepSeek-R1-Distill-Qwen-7B Model for News Classifier](https://gallery.pai-ml.com/#/preview/deepLearning/nlp/llama_factory_deepseek_r1_distill_7b) (Chinese)
*   [A One-Stop Code-Free Model Fine-Tuning & Deployment Platform based on SageMaker and LLaMA-Factory](https://aws.amazon.com/cn/blogs/china/a-one-stop-code-free-model-fine-tuning-deployment-platform-based-on-sagemaker-and-llama-factory/) (Chinese)
*   [LLaMA Factory Multi-Modal Fine-Tuning Practice: Fine-Tuning Qwen2-VL for Personal Tourist Guide](https://gallery.pai-ml.com/#/preview/deepLearning/nlp/llama_factory_qwen2vl) (Chinese)
*   [LLaMA Factory: Fine-tuning Llama3 for Role-Playing](https://gallery.pai-ml.com/#/preview/deepLearning/nlp/llama_factory) (Chinese)

</details>

## Changelog

[See the original README for an extensive changelog](https://github.com/hiyouga/LLaMA-Factory#changelog).

## Supported Models

[See the original README for a detailed list of supported models](https://github.com/hiyouga/LLaMA-Factory#supported-models).

## Supported Training Approaches

[See the original README for a detailed list of supported training approaches](https://github.com/hiyouga/LLaMA-Factory#supported-training-approaches).

## Provided Datasets

[See the original README for a detailed list of provided datasets](https://github.com/hiyouga/LLaMA-Factory#provided-datasets).

## Requirement

[See the original README for the hardware and software requirements](https://github.com/hiyouga/LLaMA-Factory#requirement).

## Getting Started

### Installation

[See the original README for installation instructions](https://github.com/hiyouga/LLaMA-Factory#installation).

### Data Preparation

[See the original README for data preparation instructions](https://github.com/hiyouga/LLaMA-Factory#data-preparation).

### Quickstart

[See the original README for quickstart instructions](https://github.com/hiyouga/LLaMA-Factory#quickstart).

### Fine-Tuning with LLaMA Board GUI (powered by [Gradio](https://github.com/gradio-app/gradio))

```bash
llamafactory-cli webui
```

### Build Docker

[See the original README for Docker build instructions](https://github.com/hiyouga/LLaMA-Factory#build-docker).

### Deploy with OpenAI-style API and vLLM

```bash
API_PORT=8000 llamafactory-cli api examples/inference/llama3.yaml infer_backend=vllm vllm_enforce_eager=true
```

### Download from ModelScope Hub

```bash
export USE_MODELSCOPE_HUB=1 # `set USE_MODELSCOPE_HUB=1` for Windows
```

Train the model by specifying a model ID of the ModelScope Hub as the `model_name_or_path`. You can find a full list of model IDs at [ModelScope Hub](https://modelscope.cn/models), e.g., `LLM-Research/Meta-Llama-3-8B-Instruct`.

### Download from Modelers Hub

```bash
export USE_OPENMIND_HUB=1 # `set USE_OPENMIND_HUB=1` for Windows
```

Train the model by specifying a model ID of the Modelers Hub as the `model_name_or_path`. You can find a full list of model IDs at [Modelers Hub](https://modelers.cn/models), e.g., `TeleAI/TeleChat-7B-pt`.

### Use W&B Logger

[See the original README for using the W&B logger](https://github.com/hiyouga/LLaMA-Factory#use-wb-logger).

### Use SwanLab Logger

[See the original README for using the SwanLab logger](https://github.com/hiyouga/LLaMA-Factory#use-swanlab-logger).

## Projects using LLaMA Factory

[See the original README for a list of projects using LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory#projects-using-llama-factory).

## License

This project is licensed under the [Apache-2.0 License](LICENSE).

## Citation

[See the original README for citation information](https://github.com/hiyouga/LLaMA-Factory#citation).

## Acknowledgement

[See the original README for acknowledgement](https://github.com/hiyouga/LLaMA-Factory#acknowledgement).

**[View the original repo for more information.](https://github.com/hiyouga/LLaMA-Factory)**