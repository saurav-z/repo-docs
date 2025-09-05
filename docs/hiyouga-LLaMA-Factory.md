# LLaMA Factory: Fine-Tune Any LLM with Ease

**Fine-tune over 100+ large language models with zero-code simplicity, unlocking the potential of AI with LLaMA Factory!** ([Original Repo](https://github.com/hiyouga/LLaMA-Factory))

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

### Used by [Amazon](https://aws.amazon.com/cn/blogs/machine-learning/how-apoidea-group-enhances-visual-information-extraction-from-banking-documents-with-multimodal-models-using-llama-factory-on-amazon-sagemaker-hyperpod/), [NVIDIA](https://developer.nvidia.com/rtx/ai-toolkit), [Aliyun](https://help.aliyun.com/zh/pai/use-cases/fine-tune-a-llama-3-model-with-llama-factory), etc.

<div align="center" markdown="1">

### Supporters ❤️

| <div style="text-align: center;"><a href="https://warp.dev/llama-factory"><img alt="Warp sponsorship" width="400" src="assets/warp.jpg"></a><br><a href="https://warp.dev/llama-factory" style="font-size:larger;">Warp, the agentic terminal for developers</a><br><a href="https://warp.dev/llama-factory">Available for MacOS, Linux, & Windows</a> | <a href="https://serpapi.com"><img alt="SerpAPI sponsorship" width="250" src="assets/serpapi.svg"> </a> |
| ---- | ---- |

----

### Fine-tuning large language models has never been easier with zero-code [CLI](#quickstart) and [Web UI](#fine-tuning-with-llama-board-gui-powered-by-gradio)!

![GitHub Trend](https://trendshift.io/api/badge/repositories/4535)

</div>

👋 Join our [WeChat](assets/wechat.jpg), [NPU](assets/wechat_npu.jpg), [Lab4AI](assets/wechat_lab4ai.jpg), [LLaMA Factory Online](assets/wechat_online.jpg) user group.

\[ English | [中文](README_zh.md) \]

**Unlock the power of LLMs with LLaMA Factory, a user-friendly toolkit for fine-tuning and deploying a vast array of language models.**

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

## Key Features

*   **Extensive Model Support:** Fine-tune 100+ models including LLaMA, Mistral, Qwen, Gemma, and many more.
*   **Versatile Training Methods:** Utilize (Continuous) pre-training, (multimodal) supervised fine-tuning, reward modeling, PPO, DPO, KTO, ORPO and SimPO.
*   **Efficient Optimization:** Leverage 16-bit full-tuning, freeze-tuning, LoRA, and various quantization techniques (QLoRA, AWQ, GPTQ, etc.) for resource optimization.
*   **Cutting-Edge Algorithms:** Incorporate advanced techniques like GaLore, BAdam, APOLLO, Adam-mini, and many more.
*   **Practical Enhancements:** Integrate FlashAttention-2, Unsloth, and RoPE scaling for improved performance and efficiency.
*   **Broad Application:** Adapt to various tasks, from multi-turn dialogue to image understanding.
*   **Comprehensive Monitoring:** Utilize LlamaBoard, TensorBoard, Wandb, MLflow, and SwanLab for experiment tracking.
*   **Fast Inference:** Benefit from OpenAI-style API, Gradio UI, and CLI integration with vLLM or SGLang for faster inference.

## Blogs

*   [Fine-tune GPT-OSS for Role-Playing using LLaMA-Factory](https://docs.llamafactory.com.cn/docs/documents/best-practice/gptroleplay/?utm_source=LLaMA-Factory) (Chinese)
*   [A One-Stop Code-Free Model Reinforcement Learning and Deployment Platform based on LLaMA-Factory and EasyR1](https://aws.amazon.com/cn/blogs/china/building-llm-model-hub-based-on-llamafactory-and-easyr1/) (Chinese)
*   [How Apoidea Group enhances visual information extraction from banking documents with multimodal models using LLaMA-Factory on Amazon SageMaker HyperPod](https://aws.amazon.com/cn/blogs/machine-learning/how-apoidea-group-enhances-visual-information-extraction-from-banking-documents-with-multimodal-models-using-llama-factory-on-amazon-sagemaker-hyperpod/) (English)
*   [Easy Dataset × LLaMA Factory: Enabling LLMs to Efficiently Learn Domain Knowledge](https://buaa-act.feishu.cn/wiki/GVzlwYcRFiR8OLkHbL6cQpYin7g) (English)

<details><summary>All Blogs</summary>

- [Fine-tune Llama3.1-70B for Medical Diagnosis using LLaMA-Factory](https://docs.alayanew.com/docs/documents/bestPractice/bigModel/llama70B/?utm_source=LLaMA-Factory) (Chinese)
- [Fine-tune Qwen2.5-VL for Autonomous Driving using LLaMA-Factory](https://docs.alayanew.com/docs/documents/useGuide/LLaMAFactory/mutiple/?utm_source=LLaMA-Factory) (Chinese)
- [LLaMA Factory: Fine-tuning the DeepSeek-R1-Distill-Qwen-7B Model for News Classifier](https://gallery.pai-ml.com/#/preview/deepLearning/nlp/llama_factory_deepseek_r1_distill_7b) (Chinese)
- [A One-Stop Code-Free Model Fine-Tuning \& Deployment Platform based on SageMaker and LLaMA-Factory](https://aws.amazon.com/cn/blogs/china/a-one-stop-code-free-model-fine-tuning-deployment-platform-based-on-sagemaker-and-llama-factory/) (Chinese)
- [LLaMA Factory Multi-Modal Fine-Tuning Practice: Fine-Tuning Qwen2-VL for Personal Tourist Guide](https://gallery.pai-ml.com/#/preview/deepLearning/nlp/llama_factory_qwen2vl) (Chinese)
- [LLaMA Factory: Fine-tuning Llama3 for Role-Playing](https://gallery.pai-ml.com/#/preview/deepLearning/nlp/llama_factory) (Chinese)

</details>

## Changelog
  
- [Support the latest models](https://github.com/hiyouga/LLaMA-Factory/blob/main/README.md#day-n-support-for-fine-tuning-cutting-edge-models)
- [Supported models list](https://github.com/hiyouga/LLaMA-Factory/blob/main/README.md#supported-models)
- [Features list](https://github.com/hiyouga/LLaMA-Factory/blob/main/README.md#features)

<details><summary>Full Changelog</summary>
See details at https://github.com/hiyouga/LLaMA-Factory/blob/main/README.md#changelog
</details>

## Supported Models

Comprehensive list of supported models (See [constants.py](src/llamafactory/extras/constants.py) for a full list):

*   Baichuan 2
*   BLOOM/BLOOMZ
*   ChatGLM3
*   Command R
*   DeepSeek (Code/MoE)
*   DeepSeek 2.5/3
*   DeepSeek R1 (Distill)
*   Falcon
*   Falcon-H1
*   Gemma/Gemma 2/CodeGemma
*   Gemma 3/Gemma 3n
*   GLM-4/GLM-4-0414/GLM-Z1
*   GLM-4.1V
*   GLM-4.5/GLM-4.5V
*   GPT-2
*   GPT-OSS
*   Granite 3.0-3.3
*   Granite 4
*   Hunyuan
*   Index
*   InternLM 2-3
*   InternVL 2.5-3.5
*   InternLM/Intern-S1-mini
*   Kimi-VL
*   Llama
*   Llama 2
*   Llama 3-3.3
*   Llama 4
*   Llama 3.2 Vision
*   LLaVA-1.5
*   LLaVA-NeXT
*   LLaVA-NeXT-Video
*   MiMo
*   MiniCPM
*   MiniCPM-o-2.6/MiniCPM-V-2.6
*   Ministral/Mistral-Nemo
*   Mistral/Mixtral
*   Mistral Small
*   OLMo
*   PaliGemma/PaliGemma2
*   Phi-1.5/Phi-2
*   Phi-3/Phi-3.5
*   Phi-3-small
*   Phi-4
*   Pixtral
*   Qwen (1-2.5) (Code/Math/MoE/QwQ)
*   Qwen3 (MoE/Instruct/Thinking)
*   Qwen2-Audio
*   Qwen2.5-Omni
*   Qwen2-VL/Qwen2.5-VL/QVQ
*   Seed Coder
*   Skywork o1
*   StarCoder 2
*   TeleChat2
*   XVERSE
*   Yi/Yi-1.5 (Code)
*   Yi-VL
*   Yuan 2

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

Pre-built datasets for fine-tuning, including:

*   Wiki Demo
*   Stanford Alpaca
*   BELLE
*   UltraChat
*   OpenOrca
*   Web QA
*   ShareGPT
*   Various other datasets (See full list in the original README)

## Requirement

*   Python 3.9+
*   PyTorch 2.0.0+
*   Transformers 4.49.0+
*   Datasets 2.16.0+
*   Accelerate 0.34.0+
*   PEFT 0.14.0+
*   TRL 0.8.6+
*   CUDA 11.6+ (Recommended)
*   and other optional dependencies.
See full details in the original README.

### Hardware Requirement

\* *estimated*

| Method                              | Bits |   7B  |  14B  |  30B  |   70B  |   `x`B  |
| ----------------------------------- | ---- | ----- | ----- | ----- | ------ | ------- |
| Full (`bf16` or `fp16`)             |  32  | 120GB | 240GB | 600GB | 1200GB | `18x`GB |
| Full (`pure_bf16`)                  |  16  |  60GB | 120GB | 300GB |  600GB |  `8x`GB |
| Freeze/LoRA/GaLore/APOLLO/BAdam/OFT |  16  |  16GB |  32GB |  64GB |  160GB |  `2x`GB |
| QLoRA / QOFT                        |   8  |  10GB |  20GB |  40GB |   80GB |   `x`GB |
| QLoRA / QOFT                        |   4  |   6GB |  12GB |  24GB |   48GB | `x/2`GB |
| QLoRA / QOFT                        |   2  |   4GB |   8GB |  16GB |   24GB | `x/4`GB |

## Getting Started

### Installation

Install LLaMA Factory from source or using Docker. Follow the instructions in the original README.

### Data Preparation

Prepare your dataset by following the format requirements and update `data/dataset_info.json` to include your custom dataset.

### Quickstart

Fine-tune, infer, and merge a model using the following commands:

```bash
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
llamafactory-cli chat examples/inference/llama3_lora_sft.yaml
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
```

### Fine-Tuning with LLaMA Board GUI (powered by [Gradio](https://github.com/gradio-app/gradio))

```bash
llamafactory-cli webui
```

### LLaMA Factory Online

Read our [documentation](https://docs.llamafactory.com.cn/docs/documents/quickstart/getstarted/?utm_source=LLaMA-Factory).

### Build Docker
Detailed instructions can be found in the original README.

### Deploy with OpenAI-style API and vLLM

```bash
API_PORT=8000 llamafactory-cli api examples/inference/llama3.yaml infer_backend=vllm vllm_enforce_eager=true
```

## Projects using LLaMA Factory

(See details in the original README)

## License

This project is licensed under the [Apache-2.0 License](LICENSE).