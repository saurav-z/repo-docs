[![LLaMA Factory](assets/logo.png)](https://github.com/hiyouga/LLaMA-Factory)

**Fine-tune any large language model with ease using LLaMA Factory, a comprehensive and efficient framework.**

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

### Fine-tune 100+ Large Language Models with Zero-Code CLI and GUI

![GitHub Trend](https://trendshift.io/api/badge/repositories/4535)

</div>

👋 Join our [WeChat group](assets/wechat.jpg), [NPU user group](assets/wechat_npu.jpg) or [Alaya NeW user group](assets/wechat_alaya.png).

\[ English | [中文](README_zh.md) \]

**Get started fine-tuning your models in just a few steps...**

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
  - [Fine-tuning with LLaMA Board GUI](#fine-tuning-with-llama-board-gui-powered-by-gradio)
  - [Deployment](#deployment)
- [More Information](#more-information)
  - [Supported Training Approaches](#supported-training-approaches)
  - [Provided Datasets](#provided-datasets)
  - [Projects Using LLaMA Factory](#projects-using-llama-factory)
  - [Hardware Requirements](#hardware-requirement)
- [License](#license)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

## Key Features

*   **Wide Model Support:** Fine-tune 100+ models including LLaMA, Mistral, Qwen, and more.
*   **Versatile Training Methods:** Supports pre-training, supervised fine-tuning, reward modeling, and reinforcement learning.
*   **Efficient Training:** Offers LoRA, QLoRA, and other optimizations for resource-efficient fine-tuning.
*   **User-Friendly Interface:** Includes a zero-code CLI and a Gradio-powered web UI for easy use.
*   **Experiment Tracking:** Integrates with tools like Wandb and SwanLab for monitoring.
*   **Faster Inference:** Provides OpenAI-style API and vLLM integration for quick deployment.

## Supported Models

*   [Baichuan 2](https://huggingface.co/baichuan-inc)
*   [BLOOM/BLOOMZ](https://huggingface.co/bigscience)
*   [ChatGLM3](https://huggingface.co/THUDM)
*   [Command R](https://huggingface.co/CohereForAI)
*   [DeepSeek (Code/MoE)](https://huggingface.co/deepseek-ai)
*   [Falcon](https://huggingface.co/tiiuae)
*   [Gemma/Gemma 2/CodeGemma](https://huggingface.co/google)
*   [GLM-4/GLM-4.1V/GLM-Z1](https://huggingface.co/zai-org)
*   [GPT-2](https://huggingface.co/openai-community)
*   [Hunyuan](https://huggingface.co/tencent/)
*   [InternLM 2-3](https://huggingface.co/internlm)
*   [Llama](https://github.com/facebookresearch/llama)
*   [Llama 2/3/4](https://huggingface.co/meta-llama)
*   [LLaVA-1.5](https://huggingface.co/llava-hf)
*   [Mixtral/Mistral/Mistral Small](https://huggingface.co/mistralai)
*   [Phi-1.5/Phi-2/Phi-3/Phi-4](https://huggingface.co/microsoft)
*   [Qwen (1-2.5) (Code/Math/MoE/QwQ)](https://huggingface.co/Qwen)
*   [Qwen3 (MoE)](https://huggingface.co/Qwen)
*   [Yi/Yi-1.5 (Code)](https://huggingface.co/01-ai)
*   More models are supported, see the complete list [here](src/llamafactory/extras/constants.py).

## Getting Started

### Installation

Install LLaMA Factory using pip:

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
```

See the [installation](#installation) section for more details.

### Quickstart

Fine-tune, infer, and merge a model with these commands:

```bash
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
llamafactory-cli chat examples/inference/llama3_lora_sft.yaml
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
```

See [examples/README.md](examples/README.md) for more advanced usage.

### Fine-tuning with LLaMA Board GUI (powered by [Gradio](https://github.com/gradio-app/gradio))

```bash
llamafactory-cli webui
```

### Deployment

Deploy with an OpenAI-style API and vLLM:

```bash
API_PORT=8000 llamafactory-cli api examples/inference/llama3.yaml infer_backend=vllm vllm_enforce_eager=true
```

## More Information

### Supported Training Approaches

*   Pre-Training
*   Supervised Fine-Tuning
*   Reward Modeling
*   PPO Training
*   DPO Training
*   KTO Training
*   ORPO Training
*   SimPO Training

### Provided Datasets

*   [Wiki Demo (en)](data/wiki_demo.txt)
*   [Stanford Alpaca (en)](https://github.com/tatsu-lab/stanford_alpaca)
*   [BELLE 2M (zh)](https://huggingface.co/datasets/BelleGroup/train_2M_CN)
*   See [Provided Datasets](#provided-datasets) for a full list.

### Projects Using LLaMA Factory

*   See the [Projects using LLaMA Factory](#projects-using-llama-factory) section for a list of related projects.

### Hardware Requirement

| Method                          | Bits |   7B  |  14B  |  30B  |   70B  |   `x`B  |
| ------------------------------- | ---- | ----- | ----- | ----- | ------ | ------- |
| Full (`bf16` or `fp16`)         |  32  | 120GB | 240GB | 600GB | 1200GB | `18x`GB |
| Freeze/LoRA/GaLore/APOLLO/BAdam |  16  |  16GB |  32GB |  64GB |  160GB |  `2x`GB |
| QLoRA                           |   8  |  10GB |  20GB |  40GB |   80GB |   `x`GB |
| QLoRA                           |   4  |   6GB |  12GB |  24GB |   48GB | `x/2`GB |
| QLoRA                           |   2  |   4GB |   8GB |  16GB |   24GB | `x/4`GB |

## License

This project is licensed under the [Apache-2.0 License](LICENSE).

## Citation

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

This repo benefits from [PEFT](https://github.com/huggingface/peft), [TRL](https://github.com/huggingface/trl), [QLoRA](https://github.com/artidoro/qlora) and [FastChat](https://github.com/lm-sys/FastChat). Thanks for their wonderful works.

[**Original Repository**](https://github.com/hiyouga/LLaMA-Factory)