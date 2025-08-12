[![LLaMA Factory Logo](assets/logo.png)](https://github.com/hiyouga/LLaMA-Factory)

<!-- Badges for quick access to project info -->
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

LLaMA Factory simplifies fine-tuning over 100+ large language models, making it accessible to everyone.  ([Original Repository](https://github.com/hiyouga/LLaMA-Factory))

**Used by [Amazon](https://aws.amazon.com/cn/blogs/machine-learning/how-apoidea-group-enhances-visual-information-extraction-from-banking-documents-with-multimodal-models-using-llama-factory-on-amazon-sagemaker-hyperpod/), [NVIDIA](https://developer.nvidia.com/rtx/ai-toolkit), [Aliyun](https://help.aliyun.com/zh/pai/use-cases/fine-tune-a-llama-3-model-with-llama-factory), etc.**

<div align="center" markdown="1">

### Supporters ❤️

| <div style="text-align: center;"><a href="https://warp.dev/llama-factory"><img alt="Warp sponsorship" width="400" src="assets/warp.jpg"></a><br><a href="https://warp.dev/llama-factory" style="font-size:larger;">Warp, the agentic terminal for developers</a><br><a href="https://warp.dev/llama-factory">Available for MacOS, Linux, & Windows</a> | <a href="https://serpapi.com"><img alt="SerpAPI sponsorship" width="250" src="assets/serpapi.svg"> </a> |
| ---- | ---- |

----

### Effortlessly Fine-tune LLMs with Zero-Code Solutions

![GitHub Trend](https://trendshift.io/api/badge/repositories/4535)

</div>

👋 Join our [WeChat group](assets/wechat.jpg), [NPU user group](assets/wechat_npu.jpg) or [Alaya NeW user group](assets/wechat_alaya.png).

\[ English | [中文](README_zh.md) \]

**Unlock the power of custom LLMs with ease!**

https://github.com/user-attachments/assets/3991a3a8-4276-4d30-9cab-4cb0c4b9b99e

Explore LLaMA Factory with these resources:

- **Documentation (WIP)**: https://llamafactory.readthedocs.io/en/latest/
- **Documentation (AMD GPU)**: https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/fine_tune/llama_factory_llama3.html
- **Colab (free)**: https://colab.research.google.com/drive/1eRTPn37ltBbYsISy9Aw2NuI2Aq5CQrD9?usp=sharing
- **Local machine**: Please refer to [usage](#getting-started)
- **PAI-DSW (free trial)**: https://gallery.pai-ml.com/#/preview/deepLearning/nlp/llama_factory
- **Alaya NeW (cloud GPU deal)**: https://docs.alayanew.com/docs/documents/useGuide/LLaMAFactory/mutiple/?utm_source=LLaMA-Factory

> [!NOTE]
> Except for the above links, all other websites are unauthorized third-party websites. Please carefully use them.

## Table of Contents

- [**Key Features**](#key-features)
- [**Supported Models**](#supported-models)
- [**Supported Training Approaches**](#supported-training-approaches)
- [**Getting Started**](#getting-started)
    - [Installation](#installation)
    - [Quickstart](#quickstart)
    - [Fine-Tuning with LLaMA Board GUI](#fine-tuning-with-llama-board-gui-powered-by-gradio)
    - [Deployment](#deploy-with-openai-style-api-and-vllm)
- [**Use Cases & Community**](#projects-using-llama-factory)
- [**Learn More**](#blogs)
- [**Citation**](#citation)
- [**License**](#license)

## **Key Features**

*   **Extensive Model Support**:  Fine-tune a vast array of models including LLaMA, Mistral, Qwen, Gemma, and many more.
*   **Versatile Training Methods**: Supports a variety of techniques like (continuous) pre-training, supervised fine-tuning, reward modeling, and advanced methods such as LoRA and QLoRA.
*   **Advanced Algorithms**: Leverage cutting-edge optimizers and techniques, including GaLore, BAdam, APOLLO, and FlashAttention-2, for improved performance and efficiency.
*   **User-Friendly Interface**: Offers a zero-code CLI and a Gradio-powered Web UI (LLaMA Board) for easy fine-tuning.
*   **Flexible Deployment**:  Deploy your fine-tuned models using an OpenAI-style API and vLLM for faster inference.

## **Supported Models**

LLaMA Factory supports a wide range of LLMs.  (See details in the [Supported Models](#supported-models) section.)

## **Supported Training Approaches**

Choose from a comprehensive list of training approaches to meet your project requirements. (See the [Supported Training Approaches](#supported-training-approaches) section.)

## **Getting Started**

### Installation

Install LLaMA Factory with a single command.

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
```

### Quickstart

Get up and running with these commands to fine-tune, infer, and merge a Llama3-8B-Instruct model.

```bash
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
llamafactory-cli chat examples/inference/llama3_lora_sft.yaml
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
```

### Fine-Tuning with LLaMA Board GUI (powered by Gradio)

```bash
llamafactory-cli webui
```

### Deployment

Deploy your fine-tuned models with an OpenAI-style API and vLLM for faster inference.

```bash
API_PORT=8000 llamafactory-cli api examples/inference/llama3.yaml infer_backend=vllm vllm_enforce_eager=true
```

## **Use Cases & Community**

Discover how others are using LLaMA Factory and find inspiration for your projects. (Explore the [Projects using LLaMA Factory](#projects-using-llama-factory) section.)

## **Learn More**

*   **Blogs**:  Read our latest blog posts and keep up-to-date with the latest features. (See the [Blogs](#blogs) section.)

## **Citation**

Cite this work as:

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

## **License**

LLaMA Factory is released under the [Apache-2.0 License](LICENSE).