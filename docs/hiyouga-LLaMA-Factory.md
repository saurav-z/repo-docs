[![LLaMA Factory](assets/logo.png)](https://github.com/hiyouga/LLaMA-Factory)

**Fine-tune and deploy over 100+ large language models effortlessly with LLaMA Factory – your all-in-one solution for efficient LLM customization!**

[Get Started with LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory)

[![GitHub Repo stars](https://img.shields.io/github/stars/hiyouga/LLaMA-Factory?style=social)](https://github.com/hiyouga/LLaMA-Factory/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/hiyouga/LLaMA-Factory)](https://github.com/hiyouga/LLaMA-Factory/commits/main)
[![GitHub contributors](https://img.shields.io/github/contributors/hiyouga/LLaMA-Factory?color=orange)](https://github.com/hiyouga/LLaMA-Factory/graphs/contributors)
[![GitHub workflow](https://github.com/hiyouga/LLaMA-Factory/actions/workflows/tests.yml/badge.svg)](https://github.com/hiyouga/LLaMA-Factory/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/llamafactory)](https://pypi.org/project/llamafactory/)
[![Citation](https://img.shields.io/badge/citation-840-green)](https://scholar.google.com/scholar?cites=12620864006390196564)
[![Docker Pulls](https://img.shields.io/docker/pulls/hiyouga/llamafactory)](https://hub.docker.com/r/hiyouga/llamafactory/tags)

[![Twitter](https://img.shields.io/twitter/follow/llamafactory_ai)](https://twitter.com/llamafactory_ai)
[![Discord](https://dcbadge.vercel.app/api/server/rKfvV9r9FK?compact=true&style=flat)](https://discord.gg/rKfvV9r9FK)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1eRTPn37ltBbYsISy9Aw2NuI2Aq5CQrD9?usp=sharing)
[![Open in DSW](https://gallery.pai-ml.com/assets/open-in-dsw.svg)](https://gallery.pai-ml.com/#/preview/deepLearning/nlp/llama_factory)
[![Open in Lab4ai](assets/lab4ai.svg)](https://www.lab4ai.cn/course/detail?id=7c13e60f6137474eb40f6fd3983c0f46?utm_source=LLaMA-Factory)
[![Open in Online](assets/online.svg)](https://www.llamafactory.com.cn/?utm_source=LLaMA-Factory)
[![Open in Spaces](https://img.shields.io/badge/🤗-Open%20in%20Spaces-blue)](https://huggingface.co/spaces/hiyouga/LLaMA-Board)
[![Open in Studios](https://img.shields.io/badge/ModelScope-Open%20in%20Studios-blue)](https://modelscope.cn/studios/hiyouga/LLaMA-Board)
[![Open in Novita](https://img.shields.io/badge/Novita-Deploy%20Template-blue)](https://novita.ai/templates-library/105981?sharer=88115474-394e-4bda-968e-b88e123d0c47)

### Used by [Amazon](https://aws.amazon.com/cn/blogs/machine-learning/how-apoidea-group-enhances-visual-information-extraction-from-banking-documents-with-multimodal-models-using-llama-factory-on-amazon-sagemaker-hyperpod/), [NVIDIA](https://developer.nvidia.com/rtx/ai-toolkit), [Aliyun](https://help.aliyun.com/zh/pai/use-cases/fine-tune-a-llama-3-model-with-llama-factory), and more.

<div align="center" markdown="1">

### Supporters ❤️

| <div style="text-align: center;"><a href="https://warp.dev/llama-factory"><img alt="Warp sponsorship" width="400" src="assets/warp.jpg"></a><br><a href="https://warp.dev/llama-factory" style="font-size:larger;">Warp, the agentic terminal for developers</a><br><a href="https://warp.dev/llama-factory">Available for MacOS, Linux, & Windows</a> | <a href="https://serpapi.com"><img alt="SerpAPI sponsorship" width="250" src="assets/serpapi.svg"> </a> |
| ---- | ---- |

----

### Easily fine-tune 100+ large language models with zero-code [CLI](#quickstart) and [Web UI](#fine-tuning-with-llama-board-gui-powered-by-gradio)

![GitHub Trend](https://trendshift.io/api/badge/repositories/4535)

</div>

👋 Join our [WeChat](assets/wechat.jpg), [NPU](assets/wechat_npu.jpg), [Lab4AI](assets/wechat_lab4ai.jpg), [LLaMA Factory Online](assets/wechat_online.jpg) user group.

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
- **Official Course**: https://www.lab4ai.cn/course/detail?id=7c13e60f6137474eb40f6fd3983c0f46?utm_source=LLaMA-Factory
- **LLaMA Factory Online**: https://www.llamafactory.com.cn/?utm_source=LLaMA-Factory

> [!NOTE]
> Except for the above links, all other websites are unauthorized third-party websites. Please carefully use them.

## Table of Contents

- [Key Features](#key-features)
- [Supported Models](#supported-models)
- [Supported Training Approaches](#supported-training-approaches)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Quickstart](#quickstart)
  - [Fine-Tuning with LLaMA Board GUI](#fine-tuning-with-llama-board-gui-powered-by-gradio)
  - [Deployment](#deployment)
- [Additional Resources](#additional-resources)
- [Projects using LLaMA Factory](#projects-using-llama-factory)
- [License](#license)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

## Key Features

*   **Extensive Model Support:** Fine-tune 100+ LLMs, including LLaMA, Mistral, Mixtral-MoE, Qwen, Gemma, and more.
*   **Versatile Training Methods:** Utilize continuous pre-training, supervised fine-tuning, reward modeling, PPO, DPO, and more.
*   **Efficient Training:** Leverage 16-bit full-tuning, freeze-tuning, LoRA, and QLoRA for optimal resource utilization.
*   **Advanced Algorithms:** Access state-of-the-art methods like GaLore, BAdam, APOLLO, and PiSSA.
*   **Practical Optimizations:** Benefit from FlashAttention-2, Unsloth, RoPE scaling, and other performance enhancements.
*   **Comprehensive Task Coverage:** Address various tasks, including dialogue, tool usage, image understanding, and more.
*   **Monitoring & Logging:** Track experiments with LlamaBoard, TensorBoard, Wandb, and SwanLab.
*   **Fast Inference:** Deploy with OpenAI-style API, Gradio UI, and CLI using vLLM or SGLang.

## Supported Models

(See original for detailed list. Highlights include LLaMA 3, Qwen, Mixtral, Gemma, etc.)

## Supported Training Approaches

(See original for the table. Includes Pre-training, SFT, Reward Modeling, PPO, DPO, KTO, ORPO, SimPO).

## Getting Started

### Installation

(See original for installation instructions, including source, Docker, and platform-specific instructions)

### Quickstart

Use these commands for LoRA fine-tuning, inference, and merging:

```bash
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
llamafactory-cli chat examples/inference/llama3_lora_sft.yaml
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
```

For more advanced usage, see [examples/README.md](examples/README.md).

### Fine-Tuning with LLaMA Board GUI (powered by [Gradio](https://github.com/gradio-app/gradio))

```bash
llamafactory-cli webui
```

### Deployment

(See original for Docker, API, and ModelScope/Modelers Hub download instructions)

## Additional Resources

*   [Documentation](https://llamafactory.readthedocs.io/en/latest/)
*   [Colab Notebook](https://colab.research.google.com/drive/1eRTPn37ltBbYsISy9Aw2NuI2Aq5CQrD9?usp=sharing)
*   [LLaMA Factory Online](https://www.llamafactory.com.cn/?utm_source=LLaMA-Factory)
*   [FAQs](https://github.com/hiyouga/LLaMA-Factory/issues/4614)

## Projects using LLaMA Factory

(See original for a list of projects)

## License

This repository is licensed under the [Apache-2.0 License](LICENSE).

## Citation

(See original for BibTeX)

## Acknowledgement

This project leverages [PEFT](https://github.com/huggingface/peft), [TRL](https://github.com/huggingface/trl), [QLoRA](https://github.com/artidoro/qlora), and [FastChat](https://github.com/lm-sys/FastChat).