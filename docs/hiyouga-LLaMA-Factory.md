<!--
  Title: LLaMA Factory - Fine-tune Any LLM with Ease
  Description: Fine-tune 100+ large language models with zero-code, offering LoRA, QLoRA, and advanced optimization techniques. Supports models like LLaMA 3, Mistral, and Qwen.  Quickstart, Web UI, and API available.
  Keywords: LLaMA Factory, LLM fine-tuning, large language models, LoRA, QLoRA,  Llama 3, Mistral, Qwen, zero-code fine-tuning, AI, machine learning, NLP, natural language processing, deep learning, PyTorch, Transformers
-->

<div align="center" markdown="1">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo.png">
    <img alt="LLaMA Factory Logo" src="assets/logo.png" width="300">
  </picture>
</div>

## LLaMA Factory: Fine-tune LLMs Effortlessly 🚀

**Unlock the power of large language models with LLaMA Factory, a versatile and user-friendly toolkit for fine-tuning over 100 LLMs.** Dive into the world of AI with our zero-code CLI and Web UI, enabling you to customize models like LLaMA 3, Mistral, and Qwen with ease.

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
[![Open in DSW](https://gallery.pai-ml.com/#/preview/deepLearning/nlp/llama_factory)
[![Open in Alaya](assets/alaya_new.svg)](https://docs.alayanew.com/docs/documents/newActivities/llamafactory/?utm_source=LLaMA-Factory)
[![Open in Spaces](https://img.shields.io/badge/🤗-Open%20in%20Spaces-blue)](https://huggingface.co/spaces/hiyouga/LLaMA-Board)
[![Open in Studios](https://img.shields.io/badge/ModelScope-Open%20in%20Studios-blue)](https://modelscope.cn/studios/hiyouga/LLaMA-Board)
[![Open in Novita](https://img.shields.io/badge/Novita-Deploy%20Template-blue)](https://novita.ai/templates-library/105981?sharer=88115474-394e-4bda-968e-b88e123d0c47)

### Trusted by Industry Leaders

Used by [Amazon](https://aws.amazon.com/cn/blogs/machine-learning/how-apoidea-group-enhances-visual-information-extraction-from-banking-documents-with-multimodal-models-using-llama-factory-on-amazon-sagemaker-hyperpod/), [NVIDIA](https://developer.nvidia.com/rtx/ai-toolkit), [Aliyun](https://help.aliyun.com/zh/pai/use-cases/fine-tune-a-llama-3-model-with-llama-factory), and many more.

<div align="center" markdown="1">
  <!-- Warp Sponsorship -->
  <a href="https://warp.dev/llama-factory">
      <img alt="Warp sponsorship" width="400" src="https://github.com/user-attachments/assets/ab8dd143-b0fd-4904-bdc5-dd7ecac94eae">
  </a>

  <!-- Warp the agentic terminal for developers -->
  #### [Warp, the agentic terminal for developers](https://warp.dev/llama-factory)

  [Available for MacOS, Linux, & Windows](https://warp.dev/llama-factory)

  ----

  ### Fine-tune LLMs Easily with Zero-Code

  **Easily fine-tune 100+ large language models with zero-code [CLI](#quickstart) and [Web UI](#fine-tuning-with-llama-board-gui-powered-by-gradio)**.

  ![GitHub Trend](https://trendshift.io/api/badge/repositories/4535)
</div>

👋 Connect with us: [WeChat group](assets/wechat.jpg), [NPU user group](assets/wechat_npu.jpg) or [Alaya NeW user group](assets/wechat_alaya.png).

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
  - [Deploy with OpenAI-style API and vLLM](#deploy-with-openai-style-api-and-vllm)
- [Projects Using LLaMA Factory](#projects-using-llama-factory)
- [License](#license)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

## Key Features

*   **Extensive Model Support**: Compatible with a vast range of models, including LLaMA, Mistral, Qwen, DeepSeek, Yi, and more.
*   **Flexible Training Methods**: Supports pre-training, supervised fine-tuning, reward modeling, PPO, DPO, KTO, and ORPO.
*   **Efficient Optimization**: Includes 16-bit full-tuning, freeze-tuning, LoRA, and 2/3/4/5/6/8-bit QLoRA via AQLM/AWQ/GPTQ/LLM.int8/HQQ/EETQ.
*   **Advanced Techniques**: Integrates cutting-edge algorithms like GaLore, BAdam, APOLLO, Muon, DoRA, LongLoRA, Mixture-of-Depths, LoRA+, LoftQ, and PiSSA.
*   **Practical Enhancements**: Offers FlashAttention-2, Unsloth, Liger Kernel, RoPE scaling, NEFTune, and rsLoRA for improved performance.
*   **Versatile Task Support**: Suitable for multi-turn dialogue, tool using, image understanding, visual grounding, video recognition, and audio understanding.
*   **Comprehensive Monitoring**: Supports LlamaBoard, TensorBoard, Wandb, MLflow, SwanLab, and others for experiment tracking.
*   **Faster Inference**: Provides an OpenAI-style API, Gradio UI, and CLI with vLLM or SGLang for rapid inference.

## Supported Models

A wide range of models are supported, including: Baichuan 2, BLOOM/BLOOMZ, ChatGLM3, Command R, DeepSeek, Falcon, Gemma, GLM-4, GPT-2, Granite, Hunyuan, Index, InternLM, InternVL, Kimi-VL, Llama, Llama 2, Llama 3, Llama 4, LLaVA, MiMo, MiniCPM, Mistral, OLMo, PaliGemma, Phi, Pixtral, Qwen, Seed Coder, Skywork, StarCoder 2, TeleChat2, XVERSE, Yi, and Yuan 2.

> \[!NOTE]
> For the "base" models, the `template` argument can be chosen from `default`, `alpaca`, `vicuna` etc. But make sure to use the **corresponding template** for the "instruct/chat" models.
>
> Remember to use the **SAME** template in training and inference.
>
> \*: You should install the `transformers` from main branch and use `DISABLE_VERSION_CHECK=1` to skip version check.
>
> \*\*: You need to install a specific version of `transformers` to use the corresponding model.

## Getting Started

### Installation

Install LLaMA Factory with the following command:

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
```

### Quickstart

Fine-tune, infer, and merge a Llama3-8B-Instruct model with these commands:

```bash
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
llamafactory-cli chat examples/inference/llama3_lora_sft.yaml
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
```

### Fine-Tuning with LLaMA Board GUI

Launch the web UI:

```bash
llamafactory-cli webui
```

### Deploy with OpenAI-style API and vLLM

```bash
API_PORT=8000 llamafactory-cli api examples/inference/llama3.yaml infer_backend=vllm vllm_enforce_eager=true
```

## Projects Using LLaMA Factory

Explore a list of projects using LLaMA Factory to see how it's being utilized in the AI community.  See [Projects using LLaMA Factory](#projects-using-llama-factory) for a detailed list.

## License

This project is licensed under the [Apache-2.0 License](LICENSE).

## Citation

If this project has been useful, please cite it as:

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

## Acknowledgement

We'd like to thank the developers of PEFT, TRL, QLoRA, and FastChat for their valuable contributions.