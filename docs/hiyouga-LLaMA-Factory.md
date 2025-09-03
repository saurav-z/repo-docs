<div align="center">
  <img src="assets/logo.png" alt="# LLaMA Factory" width="300">
  <h1>LLaMA Factory: Unleash the Power of Fine-Tuning Your LLMs</h1>
  <p><b>Fine-tune 100+ large language models (LLMs) with ease using zero-code CLI and a user-friendly Web UI.</b></p>

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

<br>

LLaMA Factory empowers you to fine-tune state-of-the-art large language models (LLMs) with remarkable efficiency.  This versatile toolkit offers a user-friendly experience with both a command-line interface (CLI) and a graphical user interface (GUI), enabling users of all skill levels to customize powerful LLMs for their specific needs.

**Key Features:**

*   <b>Extensive Model Support:</b> Works with a vast array of models including LLaMA, Mistral, Qwen, DeepSeek, Gemma, and many more.
*   <b>Versatile Training Methods:</b> Supports (Continuous) pre-training, supervised fine-tuning, reward modeling, PPO, DPO, KTO, and ORPO.
*   <b>Optimized for Efficiency:</b> Offers 16-bit full-tuning, freeze-tuning, LoRA, and low-bit quantization (QLoRA via AQLM/AWQ/GPTQ/LLM.int8/HQQ/EETQ) for resource-conscious training.
*   <b>Advanced Algorithms:</b> Integrates cutting-edge techniques like GaLore, BAdam, APOLLO, Adam-mini, Muon, OFT, DoRA, LongLoRA, and more.
*   <b>Practical Enhancements:</b> Includes FlashAttention-2, Unsloth, Liger Kernel, RoPE scaling, and NEFTune for improved performance.
*   <b>Wide Application:</b> Addresses diverse tasks, including multi-turn dialogue, image understanding, and more.
*   <b>Experiment Tracking:</b> Integrates with LlamaBoard, TensorBoard, Wandb, MLflow, and SwanLab for monitoring.
*   <b>Faster Inference:</b> Offers OpenAI-style API, Gradio UI and CLI with vLLM worker or SGLang worker for efficient deployment.
*   <b>Day-N Support:</b> Supports cutting edge models with Day 0 / Day 1 support.

<div align="center" markdown="1">
  <br>
  Used by [Amazon](https://aws.amazon.com/cn/blogs/machine-learning/how-apoidea-group-enhances-visual-information-extraction-from-banking-documents-with-multimodal-models-using-llama-factory-on-amazon-sagemaker-hyperpod/), [NVIDIA](https://developer.nvidia.com/rtx/ai-toolkit), [Aliyun](https://help.aliyun.com/zh/pai/use-cases/fine-tune-a-llama-3-model-with-llama-factory), etc.

  <br>

  ### Supporters ❤️

  | <div style="text-align: center;"><a href="https://warp.dev/llama-factory"><img alt="Warp sponsorship" width="400" src="assets/warp.jpg"></a><br><a href="https://warp.dev/llama-factory" style="font-size:larger;">Warp, the agentic terminal for developers</a><br><a href="https://warp.dev/llama-factory">Available for MacOS, Linux, & Windows</a> | <a href="https://serpapi.com"><img alt="SerpAPI sponsorship" width="250" src="assets/serpapi.svg"> </a> |
  | ---- | ---- |
</div>

----

<div align="center" markdown="1">
  <img src="https://trendshift.io/api/badge/repositories/4535">
</div>

👋 Join our [WeChat group](assets/wechat.jpg), [NPU user group](assets/wechat_npu.jpg) or [Alaya NeW user group](assets/wechat_alaya.png).

\[ English | [中文](README_zh.md) \]

**Quick Links:**

*   [Original Repository](https://github.com/hiyouga/LLaMA-Factory)
*   **Documentation (WIP)**: https://llamafactory.readthedocs.io/en/latest/
*   **Documentation (AMD GPU)**: https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/fine_tune/llama_factory_llama3.html
*   **Colab (free)**: https://colab.research.google.com/drive/1eRTPn37ltBbYsISy9Aw2NuI2Aq5CQrD9?usp=sharing
*   **PAI-DSW (free trial)**: https://gallery.pai-ml.com/#/preview/deepLearning/nlp/llama_factory
*   **Alaya NeW (cloud GPU deal)**: https://docs.alayanew.com/docs/documents/useGuide/LLaMAFactory/mutiple/?utm_source=LLaMA-Factory

> [!NOTE]
> Except for the above links, all other websites are unauthorized third-party websites. Please carefully use them.

## Table of Contents

*   [Features](#features)
*   [Supported Models](#supported-models)
*   [Getting Started](#getting-started)
    *   [Installation](#installation)
    *   [Quickstart](#quickstart)
    *   [Fine-Tuning with LLaMA Board GUI](#fine-tuning-with-llama-board-gui-powered-by-gradio)
    *   [Deployment](#deploy-with-openai-style-api-and-vllm)
*   [Projects using LLaMA Factory](#projects-using-llama-factory)
*   [Citation](#citation)
*   [License](#license)

## Supported Models

A comprehensive list of supported models and their respective templates can be found in the [Supported Models](#supported-models) section.

## Getting Started

Detailed instructions on installation, data preparation, and usage are available to get you up and running quickly.

*   [Installation](#installation)
*   [Quickstart](#quickstart)
*   [Fine-Tuning with LLaMA Board GUI](#fine-tuning-with-llama-board-gui-powered-by-gradio)
*   [Deployment](#deploy-with-openai-style-api-and-vllm)

## Projects using LLaMA Factory

Explore real-world applications of LLaMA Factory, including projects that showcase its versatility and impact.  See the [Projects using LLaMA Factory](#projects-using-llama-factory) section for a curated list.

## Citation

If you find LLaMA Factory useful in your research, please cite the project using the information provided in the [Citation](#citation) section.

## License

LLaMA Factory is licensed under the [Apache-2.0 License](LICENSE).