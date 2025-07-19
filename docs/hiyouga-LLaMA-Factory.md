# LLaMA Factory: Fine-tune Any LLM with Ease

**Unlock the power of large language models (LLMs) with LLaMA Factory, the ultimate open-source toolkit for fine-tuning over 100+ models like Llama, Mistral, and Qwen!** ([Original Repository](https://github.com/hiyouga/LLaMA-Factory))

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
  - [Web UI](#fine-tuning-with-llama-board-gui-powered-by-gradio)
  - [API Deployment](#deploy-with-openai-style-api-and-vllm)
- [Resources](#resources)
- [Projects using LLaMA Factory](#projects-using-llama-factory)
- [License](#license)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

## Key Features

*   **Wide Model Support:** Fine-tune 100+ LLMs including LLaMA, Mistral, Qwen, and more.
*   **Flexible Training:** Supports various methods: pre-training, SFT, reward modeling, PPO, DPO, and efficient techniques like LoRA and QLoRA.
*   **Advanced Algorithms:**  Integrates cutting-edge optimizers and tricks to enhance training efficiency and performance.
*   **Experiment Tracking:** Monitor your experiments with tools like LlamaBoard, TensorBoard, Wandb, and SwanLab.
*   **Fast Inference:** Deploy your fine-tuned models with an OpenAI-style API and vLLM for high-speed performance.

## Supported Models

*   **Wide Range of Models**:  Baichuan, BLOOM, ChatGLM3, DeepSeek, Gemma, Llama, Mistral, Qwen, and more. (See the full list [here](#supported-models))
*   **Model Size and Template Compatibility**: The table provides a comprehensive overview of supported models with their sizes and template requirements.

## Supported Training Approaches

*   **Full flexibility**: Fine-tuning, Freeze-tuning, LoRA, QLoRA methods.
*   **Diverse Training Methods**: Pre-training, Supervised Fine-Tuning, Reward Modeling, PPO Training, DPO Training, KTO Training, ORPO Training, and SimPO Training. (Refer to the [list](#supported-training-approaches) for details.)

## Getting Started

### Installation

Follow these steps to get started with LLaMA Factory:

1.  **Install from Source**:

    ```bash
    git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
    cd LLaMA-Factory
    pip install -e ".[torch,metrics]" --no-build-isolation
    ```

    *   Extra dependencies available: torch, torch-npu, metrics, deepspeed, liger-kernel, bitsandbytes, hqq, eetq, gptq, aqlm, vllm, sglang, galore, apollo, badam, adam-mini, qwen, minicpm\_v, openmind, swanlab, dev
2.  **Docker Installation:**
    Run pre-built docker image or build your own. Instructions available in the [docs](#build-docker).
3.  **Virtual Environment:**
    Optionally, create a virtual environment for isolated package management using uv.

    ```bash
    uv sync --extra torch --extra metrics --prerelease=allow
    ```

### Quickstart

Quickly fine-tune, infer and merge.

1.  **Fine-tuning**:

    ```bash
    llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
    ```

2.  **Inference**:

    ```bash
    llamafactory-cli chat examples/inference/llama3_lora_sft.yaml
    ```

3.  **Merging**:

    ```bash
    llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
    ```

    *   See [examples/README.md](examples/README.md) for advanced usage (including distributed training).

### Fine-Tuning with LLaMA Board GUI (powered by [Gradio](https://github.com/gradio-app/gradio))

```bash
llamafactory-cli webui
```

### Deploy with OpenAI-style API and vLLM

```bash
API_PORT=8000 llamafactory-cli api examples/inference/llama3.yaml infer_backend=vllm vllm_enforce_eager=true
```

## Resources

*   **Documentation:**  Check the [documentation](https://llamafactory.readthedocs.io/en/latest/) for a deeper dive.
*   **Colab Notebook:** Try the free [Colab notebook](https://colab.research.google.com/drive/1eRTPn37ltBbYsISy9Aw2NuI2Aq5CQrD9?usp=sharing) for a quick start.
*   **Examples:** Explore the detailed [examples](examples/README.md) to get your hands dirty.

## Projects using LLaMA Factory

LLaMA Factory is being used in a variety of research and commercial applications.  Explore the projects [here](#projects-using-llama-factory).

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

This project is based on the works of [PEFT](https://github.com/huggingface/peft), [TRL](https://github.com/huggingface/trl), [QLoRA](https://github.com/artidoro/qlora) and [FastChat](https://github.com/lm-sys/FastChat).