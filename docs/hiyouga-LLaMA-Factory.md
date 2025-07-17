<div align="center">
  <img src="assets/logo.png" alt="LLaMA Factory Logo" width="300">
  <h1>LLaMA Factory: Unleash the Power of Fine-Tuning for LLMs</h1>
  <p><em>Fine-tune over 100 large language models with ease, from Llama 3 to Qwen and more, with no-code options, advanced algorithms, and comprehensive monitoring.</em></p>

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

  <p><b>Used by Amazon, NVIDIA, Aliyun, and more.</b></p>

    <a href="https://warp.dev/llama-factory">
        <img alt="Warp sponsorship" width="400" src="https://github.com/user-attachments/assets/ab8dd143-b0fd-4904-bdc5-dd7ecac94eae">
    </a>
    <p>[Warp, the agentic terminal for developers](https://warp.dev/llama-factory)</p>
    <p>[Available for MacOS, Linux, & Windows](https://warp.dev/llama-factory)</p>

  <p>Fine-tuning a large language model can be easy as...</p>

  <img src="https://github.com/user-attachments/assets/3991a3a8-4276-4d30-9cab-4cb0c4b9b99e" alt="Fine-tuning example" width="500">

</div>

> **LLaMA Factory** empowers you to fine-tune a vast array of LLMs, including Llama 3, Qwen, Mistral, and more, with a user-friendly approach.

**[Access the original repository on GitHub](https://github.com/hiyouga/LLaMA-Factory).**

## Key Features

*   **Extensive Model Support:** Fine-tune LLaMA, LLaVA, Mistral, Mixtral-MoE, Qwen, DeepSeek, Yi, Gemma, ChatGLM, Phi, and many more.
*   **Diverse Training Methods:** Utilize (Continuous) pre-training, (multimodal) supervised fine-tuning, reward modeling, PPO, DPO, KTO, ORPO, and SimPO.
*   **Flexible Training Options:** Choose from 16-bit full-tuning, freeze-tuning, LoRA, and 2/3/4/5/6/8-bit QLoRA.
*   **Cutting-Edge Algorithms:** Access advanced algorithms like GaLore, BAdam, APOLLO, Adam-mini, Muon, DoRA, LongLoRA, LLaMA Pro, Mixture-of-Depths, LoftQ, and PiSSA.
*   **Performance Enhancements:** Leverage FlashAttention-2, Unsloth, Liger Kernel, RoPE scaling, NEFTune, and rsLoRA.
*   **Broad Application Support:** Tackle multi-turn dialogue, tool use, image understanding, visual grounding, video recognition, and audio understanding.
*   **Comprehensive Monitoring:** Monitor experiments with LlamaBoard, TensorBoard, Wandb, MLflow, and SwanLab.
*   **Faster Inference:** Utilize OpenAI-style API, Gradio UI, and CLI with vLLM or SGLang for accelerated inference.

### Day-N Support for Fine-Tuning Cutting-Edge Models

| Support Date | Model Name                                                           |
| ------------ | -------------------------------------------------------------------- |
| Day 0        | Qwen3 / Qwen2.5-VL / Gemma 3 / GLM-4.1V / InternLM 3 / MiniCPM-o-2.6 |
| Day 1        | Llama 3 / GLM-4 / Mistral Small / PaliGemma2 / Llama 4               |

## Quickstart Guide
*   **Quickstart using YAML:**
    *   First, run fine-tuning: `llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml`
    *   Next, perform inference: `llamafactory-cli chat examples/inference/llama3_lora_sft.yaml`
    *   Finally, merge LoRA weights: `llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml`

## Table of Contents

*   [Key Features](#key-features)
*   [Supported Models](#supported-models)
*   [Supported Training Approaches](#supported-training-approaches)
*   [Provided Datasets](#provided-datasets)
*   [Requirement](#requirement)
*   [Getting Started](#getting-started)
    *   [Installation](#installation)
    *   [Data Preparation](#data-preparation)
    *   [Quickstart](#quickstart)
    *   [Fine-Tuning with LLaMA Board GUI (powered by Gradio)](#fine-tuning-with-llama-board-gui-powered-by-gradio)
    *   [Build Docker](#build-docker)
    *   [Deploy with OpenAI-style API and vLLM](#deploy-with-openai-style-api-and-vllm)
    *   [Download from ModelScope Hub](#download-from-modelscope-hub)
    *   [Download from Modelers Hub](#download-from-modelers-hub)
    *   [Use W&B Logger](#use-wb-logger)
    *   [Use SwanLab Logger](#use-swanlab-logger)
*   [Projects using LLaMA Factory](#projects-using-llama-factory)
*   [License](#license)
*   [Citation](#citation)
*   [Acknowledgement](#acknowledgement)

## Supported Models

See the table in the original README for the exhaustive list.

## Supported Training Approaches

*   Pre-Training
*   Supervised Fine-Tuning
*   Reward Modeling
*   PPO Training
*   DPO Training
*   KTO Training
*   ORPO Training
*   SimPO Training

## Provided Datasets

*   Wiki Demo (en)
*   Stanford Alpaca (en&zh)
*   Open Orca (en)
*   and many more (See original README for full list)

## Requirement

See the original README for the requirements.

## Getting Started

### Installation

Install using pip:

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
```
(See original README for detailed instructions, including instructions for Windows, Ascend NPU and Docker.)

### Data Preparation

Refer to [data/README.md](data/README.md) for dataset format details.

### Fine-Tuning with LLaMA Board GUI (powered by Gradio)

```bash
llamafactory-cli webui
```

### Build Docker

(See original README for detailed Docker build instructions for CUDA, Ascend NPU, and ROCm).

### Deploy with OpenAI-style API and vLLM

```bash
API_PORT=8000 llamafactory-cli api examples/inference/llama3.yaml infer_backend=vllm vllm_enforce_eager=true
```

### Download from ModelScope Hub

```bash
export USE_MODELSCOPE_HUB=1 # `set USE_MODELSCOPE_HUB=1` for Windows
```

### Download from Modelers Hub

```bash
export USE_OPENMIND_HUB=1 # `set USE_OPENMIND_HUB=1` for Windows
```

### Use W&B Logger

Add:

```yaml
report_to: wandb
run_name: test_run # optional
```

### Use SwanLab Logger

Add:

```yaml
use_swanlab: true
swanlab_run_name: test_run # optional
```

## Projects using LLaMA Factory

See the list in the original README.

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

This project benefits from [PEFT](https://github.com/huggingface/peft), [TRL](https://github.com/huggingface/trl), [QLoRA](https://github.com/artidoro/qlora) and [FastChat](https://github.com/lm-sys/FastChat).