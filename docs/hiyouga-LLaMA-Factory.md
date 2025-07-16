# LLaMA Factory: Unleash the Power of Fine-Tuning for Large Language Models

**Fine-tune any of 100+ Large Language Models (LLMs) with ease, leveraging zero-code CLI and a user-friendly Web UI, and unlock the full potential of your AI projects!**  [Explore the original repository](https://github.com/hiyouga/LLaMA-Factory)

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

## Table of Contents

- [Key Features](#key-features)
- [Supported Models](#supported-models)
- [Supported Training Approaches](#supported-training-approaches)
- [Getting Started](#getting-started)
    - [Installation](#installation)
    - [Quickstart](#quickstart)
    - [Fine-Tuning with LLaMA Board GUI](#fine-tuning-with-llama-board-gui-powered-by-gradio)
- [Additional Resources](#additional-resources)
    - [Provided Datasets](#provided-datasets)
    - [Blogs](#blogs)
    - [Changelog](#changelog)
    - [Projects using LLaMA Factory](#projects-using-llama-factory)
    - [Deployment Options](#deployment-options)
- [Technical Details](#technical-details)
    - [Requirement](#requirement)
    - [Hardware Requirement](#hardware-requirement)
    - [Citation](#citation)
    - [License](#license)
    - [Acknowledgement](#acknowledgement)

## Key Features

*   **Wide Model Support:** Fine-tune a vast array of models, including LLaMA, Mistral, Qwen, and many more.
*   **Diverse Training Methods:** Supports various methods, including (continuous) pre-training, (multimodal) supervised fine-tuning, reward modeling, PPO, DPO, KTO, ORPO, etc.
*   **Efficient Training:** Utilize 16-bit full-tuning, freeze-tuning, LoRA, and efficient quantization techniques like QLoRA (2/3/4/5/6/8-bit) for resource optimization.
*   **Cutting-Edge Algorithms:** Integrates advanced algorithms such as GaLore, BAdam, APOLLO, Adam-mini, Muon, DoRA, LongLoRA, LLaMA Pro, Mixture-of-Depths, LoRA+, LoftQ and PiSSA.
*   **Practical Optimizations:** Incorporates tools like FlashAttention-2, Unsloth, Liger Kernel, RoPE scaling, NEFTune, and rsLoRA for improved performance.
*   **Versatile Applications:** Adaptable for various tasks, including multi-turn dialogue, tool usage, image understanding, and more.
*   **Experiment Tracking:** Supports various experiment monitors, including LlamaBoard, TensorBoard, Wandb, MLflow, [SwanLab](https://github.com/SwanHubX/SwanLab), etc.
*   **Fast Inference:** Offers OpenAI-style API, Gradio UI and CLI with [vLLM worker](https://github.com/vllm-project/vllm) or [SGLang worker](https://github.com/sgl-project/sglang).

## Supported Models

*   [Baichuan 2](https://huggingface.co/baichuan-inc)
*   [BLOOM/BLOOMZ](https://huggingface.co/bigscience)
*   [ChatGLM3](https://huggingface.co/THUDM)
*   [Command R](https://huggingface.co/CohereForAI)
*   [DeepSeek (Code/MoE)](https://huggingface.co/deepseek-ai)
*   [DeepSeek 2.5/3](https://huggingface.co/deepseek-ai)
*   [DeepSeek R1 (Distill)](https://huggingface.co/deepseek-ai)
*   [Falcon](https://huggingface.co/tiiuae)
*   [Falcon-H1](https://huggingface.co/tiiuae)
*   [Gemma/Gemma 2/CodeGemma](https://huggingface.co/google)
*   [Gemma 3/Gemma 3n](https://huggingface.co/google)
*   [GLM-4/GLM-4-0414/GLM-Z1](https://huggingface.co/THUDM)
*   [GLM-4.1V](https://huggingface.co/THUDM)
*   [GPT-2](https://huggingface.co/openai-community)
*   [Granite 3.0-3.3](https://huggingface.co/ibm-granite)
*   [Hunyuan](https://huggingface.co/tencent/)
*   [Index](https://huggingface.co/IndexTeam)
*   [InternLM 2-3](https://huggingface.co/internlm)
*   [InternVL 2.5-3](https://huggingface.co/OpenGVLab)
*   [Kimi-VL](https://huggingface.co/moonshotai)
*   [Llama](https://github.com/facebookresearch/llama)
*   [Llama 2](https://huggingface.co/meta-llama)
*   [Llama 3-3.3](https://huggingface.co/meta-llama)
*   [Llama 4](https://huggingface.co/meta-llama)
*   [Llama 3.2 Vision](https://huggingface.co/meta-llama)
*   [LLaVA-1.5](https://huggingface.co/llava-hf)
*   [LLaVA-NeXT](https://huggingface.co/llava-hf)
*   [LLaVA-NeXT-Video](https://huggingface.co/llava-hf)
*   [MiMo](https://huggingface.co/XiaomiMiMo)
*   [MiniCPM](https://huggingface.co/openbmb)
*   [MiniCPM-o-2.6/MiniCPM-V-2.6](https://huggingface.co/openbmb)
*   [Ministral/Mistral-Nemo](https://huggingface.co/mistralai)
*   [Mistral/Mixtral](https://huggingface.co/mistralai)
*   [Mistral Small](https://huggingface.co/mistralai)
*   [OLMo](https://huggingface.co/allenai)
*   [PaliGemma/PaliGemma2](https://huggingface.co/google)
*   [Phi-1.5/Phi-2](https://huggingface.co/microsoft)
*   [Phi-3/Phi-3.5](https://huggingface.co/microsoft)
*   [Phi-3-small](https://huggingface.co/microsoft)
*   [Phi-4](https://huggingface.co/microsoft)
*   [Pixtral](https://huggingface.co/mistralai)
*   [Qwen (1-2.5) (Code/Math/MoE/QwQ)](https://huggingface.co/Qwen)
*   [Qwen3 (MoE)](https://huggingface.co/Qwen)
*   [Qwen2-Audio](https://huggingface.co/Qwen)
*   [Qwen2.5-Omni](https://huggingface.co/Qwen)
*   [Qwen2-VL/Qwen2.5-VL/QVQ](https://huggingface.co/Qwen)
*   [Seed Coder](https://huggingface.co/ByteDance-Seed)
*   [Skywork o1](https://huggingface.co/Skywork)
*   [StarCoder 2](https://huggingface.co/bigcode)
*   [TeleChat2](https://huggingface.co/Tele-AI)
*   [XVERSE](https://huggingface.co/xverse)
*   [Yi/Yi-1.5 (Code)](https://huggingface.co/01-ai)
*   [Yi-VL](https://huggingface.co/01-ai)
*   [Yuan 2](https://huggingface.co/IEITYuan)

## Supported Training Approaches

*   Pre-Training
*   Supervised Fine-Tuning
*   Reward Modeling
*   PPO Training
*   DPO Training
*   KTO Training
*   ORPO Training
*   SimPO Training

## Getting Started

### Installation

To get started, follow these steps:

1.  **Clone the Repository:**

    ```bash
    git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
    cd LLaMA-Factory
    ```

2.  **Install Dependencies:**

    ```bash
    pip install -e ".[torch,metrics]" --no-build-isolation
    ```
    (Check [installation](#installation) section for more details)

### Quickstart

Fine-tuning, inference, and merging can be performed with these commands:

```bash
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
llamafactory-cli chat examples/inference/llama3_lora_sft.yaml
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
```

### Fine-Tuning with LLaMA Board GUI

Launch the web UI with:

```bash
llamafactory-cli webui
```

## Additional Resources

### Provided Datasets

LLaMA Factory provides a wide array of pre-training, supervised fine-tuning, and preference datasets, including:

*   Wiki Demo (en)
*   Stanford Alpaca (en)
*   BELLE (zh)
*   UltraChat (en)
*   OpenOrca (en)
*   ...and many more (See [Provided Datasets](#provided-datasets) for a full list)

### Blogs

*   [A One-Stop Code-Free Model Reinforcement Learning and Deployment Platform based on LLaMA-Factory and EasyR1](https://aws.amazon.com/cn/blogs/china/building-llm-model-hub-based-on-llamafactory-and-easyr1/) (Chinese)
*   [Fine-tune Qwen2.5-VL for Autonomous Driving using LLaMA-Factory](https://docs.alayanew.com/docs/documents/useGuide/LLaMAFactory/mutiple/?utm_source=LLaMA-Factory) (Chinese)
*   [How Apoidea Group enhances visual information extraction from banking documents with multimodal models using LLaMA-Factory on Amazon SageMaker HyperPod](https://aws.amazon.com/cn/blogs/machine-learning/how-apoidea-group-enhances-visual-information-extraction-from-banking-documents-with-multimodal-models-using-llama-factory-on-amazon-sagemaker-hyperpod/) (English)
*   [Easy Dataset × LLaMA Factory: Enabling LLMs to Efficiently Learn Domain Knowledge](https://buaa-act.feishu.cn/wiki/GVzlwYcRFiR8OLkHbL6cQpYin7g) (English)

### Changelog

The changelog is regularly updated, including recent additions like:

*   Support for fine-tuning [GLM-4.1V](https://github.com/THUDM/GLM-4.1V-Thinking) and [Qwen3](https://qwenlm.github.io/blog/qwen3/) model families.
*   Integration of the [Muon](https://github.com/KellerJordan/Muon) optimizer.
*   Support for [InternVL3](https://huggingface.co/OpenGVLab/InternVL3-8B), [GLM-Z1](https://huggingface.co/THUDM/GLM-Z1-9B-0414), and [Kimi-VL](https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct) models.
*   Support for [Llama 4](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) models
*   ... and much more (See [Changelog](#changelog) for a full list)

### Projects using LLaMA Factory

*   Wang et al. ESRL: Efficient Sampling-based Reinforcement Learning for Sequence Generation. 2023. [[arxiv]](https://arxiv.org/abs/2308.02223)
*   Yu et al. Open, Closed, or Small Language Models for Text Classification? 2023. [[arxiv]](https://arxiv.org/abs/2308.10092)
*   Wang et al. UbiPhysio: Support Daily Functioning, Fitness, and Rehabilitation with Action Understanding and Feedback in Natural Language. 2023. [[arxiv]](https://arxiv.org/abs/2308.10526)
*   Luceri et al. Leveraging Large Language Models to Detect Influence Campaigns in Social Media. 2023. [[arxiv]](https://arxiv.org/abs/2311.07816)
*   ...and many more (See [Projects using LLaMA Factory](#projects-using-llama-factory) for a full list)

### Deployment Options

*   Deploy with OpenAI-style API and vLLM
*   Docker support
*   Download models from ModelScope and Modelers Hub

## Technical Details

### Requirement

*   Python 3.9+
*   PyTorch 2.0.0+
*   Transformers 4.49.0+
*   Datasets 2.16.0+
*   Accelerate 0.34.0+
*   PEFT 0.14.0+
*   TRL 0.8.6+
    (See [Requirement](#requirement) for a full list)

### Hardware Requirement

*   Full-tuning (32-bit): 7B model requires ~120GB, 70B requires ~1200GB
*   Freeze/LoRA/GaLore/APOLLO/BAdam: 7B model requires ~16GB
*   QLoRA: 7B model requires ~10GB (4-bit)

### Citation

If you find this work useful, please cite:

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

### License

This project is licensed under the [Apache-2.0 License](LICENSE).

### Acknowledgement

This project builds upon the work of [PEFT](https://github.com/huggingface/peft), [TRL](https://github.com/huggingface/trl), [QLoRA](https://github.com/artidoro/qlora), and [FastChat](https://github.com/lm-sys/FastChat).