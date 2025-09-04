<!-- Logo and Social Badges -->
<div align="center">
  <img src="assets/logo.png" alt="LLaMA Factory Logo" width="300">
  <!-- GitHub Badges -->
  [![GitHub Repo stars](https://img.shields.io/github/stars/hiyouga/LLaMA-Factory?style=social)](https://github.com/hiyouga/LLaMA-Factory/stargazers)
  [![GitHub last commit](https://img.shields.io/github/last-commit/hiyouga/LLaMA-Factory)](https://github.com/hiyouga/LLaMA-Factory/commits/main)
  [![GitHub contributors](https://img.shields.io/github/contributors/hiyouga/LLaMA-Factory?color=orange)](https://github.com/hiyouga/LLaMA-Factory/graphs/contributors)
  [![GitHub workflow](https://github.com/hiyouga/LLaMA-Factory/actions/workflows/tests.yml/badge.svg)](https://github.com/hiyouga/LLaMA-Factory/actions/workflows/tests.yml)
  <!-- PyPI, Citation, Docker -->
  [![PyPI](https://img.shields.io/pypi/v/llamafactory)](https://pypi.org/project/llamafactory/)
  [![Citation](https://img.shields.io/badge/citation-840-green)](https://scholar.google.com/scholar?cites=12620864006390196564)
  [![Docker Pulls](https://img.shields.io/docker/pulls/hiyouga/llamafactory)](https://hub.docker.com/r/hiyouga/llamafactory/tags)
  <!-- Social Media Links -->
  [![Twitter](https://img.shields.io/twitter/follow/llamafactory_ai)](https://twitter.com/llamafactory_ai)
  [![Discord](https://dcbadge.vercel.app/api/server/rKfvV9r9FK?compact=true&style=flat)](https://discord.gg/rKfvV9r9FK)
  <!-- Run Online Links -->
  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1eRTPn37ltBbYsISy9Aw2NuI2Aq5CQrD9?usp=sharing)
  [![Open in DSW](https://gallery.pai-ml.com/assets/open-in-dsw.svg)](https://gallery.pai-ml.com/#/preview/deepLearning/nlp/llama_factory)
  [![Open in Lab4ai](assets/lab4ai.svg)](https://www.lab4ai.cn/course/detail?id=7c13e60f6137474eb40f6fd3983c0f46?utm_source=LLaMA-Factory)
  [![Open in Online](assets/online.svg)](https://www.llamafactory.com.cn/?utm_source=LLaMA-Factory)
  [![Open in Spaces](https://img.shields.io/badge/🤗-Open%20in%20Spaces-blue)](https://huggingface.co/spaces/hiyouga/LLaMA-Board)
  [![Open in Studios](https://img.shields.io/badge/ModelScope-Open%20in%20Studios-blue)](https://modelscope.cn/studios/hiyouga/LLaMA-Board)
  [![Open in Novita](https://img.shields.io/badge/Novita-Deploy%20Template-blue)](https://novita.ai/templates-library/105981?sharer=88115474-394e-4bda-968e-b88e123d0c47)
</div>

---

**LLaMA Factory: The open-source powerhouse for effortless fine-tuning of 100+ large language models.**  

Used by [Amazon](https://aws.amazon.com/cn/blogs/machine-learning/how-apoidea-group-enhances-visual-information-extraction-from-banking-documents-with-multimodal-models-using-llama-factory-on-amazon-sagemaker-hyperpod/), [NVIDIA](https://developer.nvidia.com/rtx/ai-toolkit), [Aliyun](https://help.aliyun.com/zh/pai/use-cases/fine-tune-a-llama-3-model-with-llama-factory), and more!

---
<div align="center" markdown="1">

### Supporters ❤️

| <div style="text-align: center;"><a href="https://warp.dev/llama-factory"><img alt="Warp sponsorship" width="400" src="assets/warp.jpg"></a><br><a href="https://warp.dev/llama-factory" style="font-size:larger;">Warp, the agentic terminal for developers</a><br><a href="https://warp.dev/llama-factory">Available for MacOS, Linux, & Windows</a> | <a href="https://serpapi.com"><img alt="SerpAPI sponsorship" width="250" src="assets/serpapi.svg"> </a> |
| ---- | ---- |

----

### Fine-tune LLMs with ease using zero-code [CLI](#quickstart) and [Web UI](#fine-tuning-with-llama-board-gui-powered-by-gradio)

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

*   [Key Features](#key-features)
*   [Supported Models](#supported-models)
*   [Getting Started](#getting-started)
    *   [Installation](#installation)
    *   [Quickstart](#quickstart)
    *   [Fine-tuning with LLaMA Board GUI](#fine-tuning-with-llama-board-gui-powered-by-gradio)
    *   [Deployment](#deploy-with-openai-style-api-and-vllm)
*   [Additional Resources](#additional-resources)
*   [License](#license)
*   [Citation](#citation)
*   [Acknowledgement](#acknowledgement)
---
## Key Features
*   **Extensive Model Support:** Fine-tune a vast array of models including LLaMA, Mistral, Qwen, Gemma, and more.
*   **Diverse Training Methods:** Utilize various methods: (Continuous) Pre-training, (Multimodal) Supervised Fine-tuning, Reward Modeling, and reinforcement learning methods like PPO/DPO/KTO/ORPO.
*   **Efficient Fine-tuning:** Leverage Full-tuning, Freeze-tuning, LoRA, QLoRA, and other methods for scalable resource usage.
*   **Advanced Techniques:** Benefit from cutting-edge algorithms, practical optimizations and tricks, including: GaLore, BAdam, APOLLO, FlashAttention-2, Unsloth, and many more.
*   **Comprehensive Tasks:** Tackle multi-turn dialogues, tool usage, image understanding, and more.
*   **Monitoring & Logging:** Integrate with LlamaBoard, TensorBoard, Weights & Biases (W&B), and SwanLab for experiment tracking.
*   **Fast Inference:** Utilize OpenAI-style API, Gradio UI, and CLI with vLLM or SGLang for accelerated inference.

## Supported Models

A comprehensive list of supported models is available in the [Supported Models](#supported-models) section.  We continuously add support for new models; see our [Changelog](#changelog) for the latest updates.

## Getting Started

### Installation

Follow these steps to install LLaMA Factory:

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
```

For detailed instructions including instructions for specific platforms (Windows, Ascend NPU, and more), refer to the [Installation](#installation) section of the documentation.

### Quickstart

Get up and running with LoRA fine-tuning, inference, and merging using the following commands:

```bash
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
llamafactory-cli chat examples/inference/llama3_lora_sft.yaml
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
```

Refer to [examples/README.md](examples/README.md) for advanced usage.

### Fine-tuning with LLaMA Board GUI

Launch the user-friendly web UI:

```bash
llamafactory-cli webui
```

### Deployment

Deploy your fine-tuned models with an OpenAI-style API and vLLM:

```bash
API_PORT=8000 llamafactory-cli api examples/inference/llama3.yaml infer_backend=vllm vllm_enforce_eager=true
```

## Additional Resources
*   **[Documentation](https://llamafactory.readthedocs.io/en/latest/):** Detailed documentation.
*   **[Examples](examples/README.md):** Code examples and usage.
*   **[Changelog](#changelog):** Stay up-to-date with the latest features.
*   **[Model and Dataset Hubs](https://modelscope.cn/models):** Download pre-trained models and datasets.

## License

This project is licensed under the [Apache-2.0 License](LICENSE).

## Citation

If you find this work helpful, please cite as:

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

We are grateful to the developers of PEFT, TRL, QLoRA, and FastChat for their contributions to this project.