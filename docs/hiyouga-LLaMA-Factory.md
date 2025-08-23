<div align="center">
  <img src="assets/logo.png" alt="LLaMA Factory Logo" width="200">
  <h1>LLaMA Factory: Effortlessly Fine-tune LLMs with Zero Code</h1>
  <p>Unlock the power of large language models (LLMs) with LLaMA Factory, a versatile and user-friendly toolkit designed for efficient fine-tuning. Fine-tune 100+ LLMs, including LLaMA, Mistral, Qwen, and more!  <a href="https://github.com/hiyouga/LLaMA-Factory">Explore the GitHub Repository</a></p>

  [![GitHub Repo stars](https://img.shields.io/github/stars/hiyouga/LLaMA-Factory?style=social)](https://github.com/hiyouga/LLaMA-Factory/stargazers)
  [![GitHub last commit](https://img.shields.io/github/last-commit/hiyouga/LLaMA-Factory)](https://github.com/hiyouga/LLaMA-Factory/commits/main)
  [![GitHub contributors](https://img.shields.io/github/contributors/hiyouga/LLaMA-Factory?color=orange)](https://github.com/hiyouga/LLaMA-Factory/graphs/contributors)
  [![GitHub workflow](https://github.com/hiyouga/LLaMA-Factory/actions/workflows/tests.yml/badge.svg)](https://github.com/hiyouga/LLaMA-Factory/actions/workflows/tests.yml)
  [![PyPI](https://img.shields.io/pypi/v/llamafactory)](https://pypi.org/project/llamafactory/)
  [![Citation](https://img.shields.io/badge/citation-760-green)](https://scholar.google.com/scholar?cites=12620864006390196564)
  [![Docker Pulls](https://img.shields.io/docker/pulls/hiyouga/llamafactory)](https://hub.docker.com/r/hiyouga/llamafactory/tags)

  [![Twitter](https://img.shields.io/twitter/follow/llamafactory_ai)](https://twitter.com/llamafactory_ai)
  [![Discord](https://dcbadge.vercel.app/api/server/rKfvV9r9FK?compact=true&style=flat)](https://discord.gg/rKfvV9r9FK)

  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1eRTPn37ltBbYsISy9Aw2NuI2Aq5CQrD9?usp=sharing)
  [![Open in DSW](https://gallery.pai-ml.com/assets/open-in-dsw.svg)](https://gallery.pai-ml.com/#/preview/deepLearning/nlp/llama_factory)
  [![Open in Alaya](assets/alaya_new.svg)](https://docs.alayanew.com/docs/documents/newActivities/llamafactory/?utm_source=LLaMA-Factory)
  [![Open in Spaces](https://img.shields.io/badge/🤗-Open%20in%20Spaces-blue)](https://huggingface.co/spaces/hiyouga/LLaMA-Board)
  [![Open in Studios](https://img.shields.io/badge/ModelScope-Open%20in%20Studios-blue)](https://modelscope.cn/studios/hiyouga/LLaMA-Board)
  [![Open in Novita](https://img.shields.io/badge/Novita-Deploy%20Template-blue)](https://novita.ai/templates-library/105981?sharer=88115474-394e-4bda-968e-b88e123d0c47)

  <p>
    Used by <a href="https://aws.amazon.com/cn/blogs/machine-learning/how-apoidea-group-enhances-visual-information-extraction-from-banking-documents-with-multimodal-models-using-llama-factory-on-amazon-sagemaker-hyperpod/" target="_blank">Amazon</a>, <a href="https://developer.nvidia.com/rtx/ai-toolkit" target="_blank">NVIDIA</a>, <a href="https://help.aliyun.com/zh/pai/use-cases/fine-tune-a-llama-3-model-with-llama-factory" target="_blank">Aliyun</a>, and more.
  </p>
</div>

## Key Features:

*   **Extensive Model Support:** Fine-tune a wide range of LLMs, including LLaMA, LLaVA, Mistral, Qwen, DeepSeek, Yi, Gemma, and more.
*   **Versatile Training Methods:** Supports pre-training, supervised fine-tuning (SFT), reward modeling, PPO, DPO, KTO, ORPO, and SimPO.
*   **Efficient Optimization:** Utilize a variety of techniques like full-tuning, freeze-tuning, LoRA, QLoRA (2/3/4/5/6/8-bit), GaLore, BAdam, APOLLO, and more.
*   **Cutting-Edge Algorithms:** Benefit from advanced features like FlashAttention-2, Unsloth, Liger Kernel, RoPE scaling, NEFTune, and rsLoRA.
*   **Diverse Task Capabilities:** Address various tasks, including multi-turn dialogue, tool use, image understanding, and video recognition.
*   **Comprehensive Monitoring:** Monitor experiments with LlamaBoard, TensorBoard, Wandb, MLflow, and SwanLab.
*   **Fast Inference:** Deploy with OpenAI-style API, Gradio UI, and CLI using vLLM or SGLang for accelerated inference.

<br>
<br>

## Table of Contents

*   [Supported Models](#supported-models)
*   [Getting Started](#getting-started)
    *   [Installation](#installation)
    *   [Quickstart](#quickstart)
    *   [Fine-tuning with LLaMA Board GUI](#fine-tuning-with-llama-board-gui-powered-by-gradio)
    *   [Deploy with OpenAI-style API and vLLM](#deploy-with-openai-style-api-and-vllm)
*   [Projects Using LLaMA Factory](#projects-using-llama-factory)
*   [License](#license)
*   [Citation](#citation)
*   [Acknowledgement](#acknowledgement)

<br>

## Supported Models
(See complete list in original README)

## Getting Started

### Installation

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
```

### Quickstart

```bash
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
llamafactory-cli chat examples/inference/llama3_lora_sft.yaml
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
```

### Fine-tuning with LLaMA Board GUI (powered by [Gradio](https://github.com/gradio-app/gradio))

```bash
llamafactory-cli webui
```

### Deploy with OpenAI-style API and vLLM

```bash
API_PORT=8000 llamafactory-cli api examples/inference/llama3.yaml infer_backend=vllm vllm_enforce_eager=true
```

<br>
<br>

## Projects Using LLaMA Factory

(See list in original README)

## License

This repository is licensed under the [Apache-2.0 License](LICENSE).

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