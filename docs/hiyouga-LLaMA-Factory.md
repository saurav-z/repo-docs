# LLaMA Factory: Fine-tune Any Large Language Model with Ease

**Fine-tune over 100+ large language models with zero code using LLaMA Factory, a versatile and efficient toolkit. Get started with a simple command, access cutting-edge algorithms, and deploy your models with ease.  ([Original Repository](https://github.com/hiyouga/LLaMA-Factory))**

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
  - [Deploy with OpenAI-style API and vLLM](#deploy-with-openai-style-api-and-vllm)
- [Projects Using LLaMA Factory](#projects-using-llama-factory)
- [License](#license)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

## Key Features

*   **Extensive Model Support:** Fine-tune a wide range of models, including LLaMA, Mistral, Qwen, and more.
*   **Versatile Training Methods:** Utilize (continuous) pre-training, supervised fine-tuning, reward modeling, PPO, DPO, KTO, ORPO, and SimPO training approaches.
*   **Efficient Training Options:** Leverage 16-bit full-tuning, freeze-tuning, LoRA, and QLoRA (2/3/4/5/6/8-bit) for optimized resource usage.
*   **Advanced Algorithms and Tricks:** Access state-of-the-art techniques such as GaLore, BAdam, APOLLO, FlashAttention-2, Unsloth, RoPE scaling, and NEFTune.
*   **Multi-Task Capabilities:** Tackle diverse tasks like multi-turn dialogue, tool use, image understanding, and more.
*   **Experiment Tracking & Monitoring:**  Integrate with LlamaBoard, TensorBoard, Wandb, MLflow, and SwanLab for experiment tracking.
*   **Fast Inference:**  Deploy models with an OpenAI-style API, Gradio UI, and CLI with vLLM or SGLang backends for faster inference.
*   **Day-N Support:** Benefit from day-one support for the latest models, including Qwen3 and Llama 3.

## Supported Models

*   Baichuan 2, BLOOM/BLOOMZ, ChatGLM3, Command R, DeepSeek (Code/MoE), Falcon, Gemma/Gemma 2, GLM-4, GPT-2, Granite, Hunyuan, Index, InternLM 2-3, Kimi-VL, Llama, Llama 2, Llama 3, LLaVA-1.5, MiMo, MiniCPM, Mistral/Mixtral, Phi-4, Qwen (1-2.5), Seed Coder, Skywork o1, StarCoder 2, TeleChat2, XVERSE, Yi, Yuan 2, and more.

    *   See [Supported Models](#supported-models) for a comprehensive list.

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

Follow these steps to install LLaMA Factory:

1.  **Install from Source:**
    ```bash
    git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
    cd LLaMA-Factory
    pip install -e ".[torch,metrics]" --no-build-isolation
    ```
    *   Refer to the original README for detailed installation instructions, including extra dependencies and Windows/NPU-specific setups.

2.  **Install from Docker Image:**
    ```bash
    docker run -it --rm --gpus=all --ipc=host hiyouga/llamafactory:latest
    ```

    *   Pre-built images are available at: [https://hub.docker.com/r/hiyouga/llamafactory/tags](https://hub.docker.com/r/hiyouga/llamafactory/tags)
    *   See the original README for instructions on building the Docker image.

### Quickstart

Fine-tune, infer, and merge the Llama3-8B-Instruct model with these commands:

```bash
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
llamafactory-cli chat examples/inference/llama3_lora_sft.yaml
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
```

*   Explore [examples/README.md](examples/README.md) for advanced usage.

### Fine-Tuning with LLaMA Board GUI

Launch the GUI with this command:

```bash
llamafactory-cli webui
```

### Deploy with OpenAI-style API and vLLM

```bash
API_PORT=8000 llamafactory-cli api examples/inference/llama3.yaml infer_backend=vllm vllm_enforce_eager=true
```

*   Refer to the original README for more deployment options, including ModelScope and Modelers Hub downloads.

## Projects Using LLaMA Factory

*   A list of projects that leverage LLaMA Factory can be found in the [Projects using LLaMA Factory](#projects-using-llama-factory) section of the original README.

## License

This repository is licensed under the [Apache-2.0 License](LICENSE).

## Citation

If you use this work, please cite the following paper:

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

This project is built upon the contributions of the PEFT, TRL, QLoRA, and FastChat projects.

---