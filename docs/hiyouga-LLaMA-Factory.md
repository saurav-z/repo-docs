<div align="center">
  <img src="assets/logo.png" alt="LLaMA Factory Logo" width="300">
  <h1>LLaMA Factory: Effortlessly Fine-Tune Large Language Models</h1>

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

  <p>
    <a href="https://warp.dev/llama-factory">
      <img alt="Warp sponsorship" width="400" src="https://github.com/user-attachments/assets/ab8dd143-b0fd-4904-bdc5-dd7ecac94eae">
    </a>
  </p>

  <p>
      <b>Fine-tune and deploy over 100 large language models with ease using the open-source LLaMA Factory.</b>
  </p>

</div>

**LLaMA Factory** empowers you to fine-tune a wide array of large language models (LLMs) with minimal coding, offering a user-friendly experience for both beginners and experts.  Check out the original repository: [https://github.com/hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)

**Key Features:**

*   **Extensive Model Support:** LLaMA Factory supports a vast selection of LLMs, including LLaMA, Mistral, Qwen, DeepSeek, and many more.
*   **Versatile Training Methods:**  Offers diverse training approaches such as (continuous) pre-training, (multimodal) supervised fine-tuning, reward modeling, PPO, DPO, KTO, and ORPO.
*   **Efficient Training Techniques:** Supports various techniques for efficient training, including 16-bit full-tuning, freeze-tuning, LoRA, and QLoRA (2/3/4/5/6/8-bit).
*   **Advanced Optimization Algorithms:** Includes advanced algorithms like GaLore, BAdam, APOLLO, and more.
*   **Practical Enhancements:** Integrates useful tricks like FlashAttention-2, Unsloth, and RoPE scaling.
*   **Wide Range of Applications:**  Suitable for multi-turn dialogue, tool usage, image understanding, and other tasks.
*   **Comprehensive Monitoring Tools:** Integrates with LlamaBoard, TensorBoard, Wandb, MLflow, and SwanLab for experiment tracking.
*   **Fast Inference Options:** Provides OpenAI-style API, Gradio UI, and CLI with vLLM or SGLang for faster inference.

**Used by:** [Amazon](https://aws.amazon.com/cn/blogs/machine-learning/how-apoidea-group-enhances-visual-information-extraction-from-banking-documents-with-multimodal-models-using-llama-factory-on-amazon-sagemaker-hyperpod/), [NVIDIA](https://developer.nvidia.com/rtx/ai-toolkit), [Aliyun](https://help.aliyun.com/zh/pai/use-cases/fine-tune-a-llama-3-model-with-llama-factory), etc.

Join our community:  Join our [WeChat group](assets/wechat.jpg), [NPU user group](assets/wechat_npu.jpg) or [Alaya NeW user group](assets/wechat_alaya.png).

**Choose Your Path:**

-   [Documentation (WIP)](https://llamafactory.readthedocs.io/en/latest/)
-   [Documentation (AMD GPU)](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/fine_tune/llama_factory_llama3.html)
-   [Colab (free)](https://colab.research.google.com/drive/1eRTPn37ltBbYsISy9Aw2NuI2Aq5CQrD9?usp=sharing)
-   [Local Machine](#getting-started)
-   [PAI-DSW (free trial)](https://gallery.pai-ml.com/#/preview/deepLearning/nlp/llama_factory)
-   [Alaya NeW (cloud GPU deal)](https://docs.alayanew.com/docs/documents/useGuide/LLaMAFactory/mutiple/?utm_source=LLaMA-Factory)

> [!NOTE]
> Except for the above links, all other websites are unauthorized third-party websites. Please carefully use them.

## Table of Contents

-   [Supported Models](#supported-models)
-   [Supported Training Approaches](#supported-training-approaches)
-   [Quickstart](#quickstart)
-   [Getting Started](#getting-started)
    -   [Installation](#installation)
    -   [Data Preparation](#data-preparation)
    -   [Fine-Tuning with LLaMA Board GUI](#fine-tuning-with-llama-board-gui-powered-by-gradio)
    -   [Build Docker](#build-docker)
    -   [Deploy with OpenAI-style API and vLLM](#deploy-with-openai-style-api-and-vllm)
    -   [Download from ModelScope Hub](#download-from-modelscope-hub)
    -   [Download from Modelers Hub](#download-from-modelers-hub)
    -   [Use W&B Logger](#use-wb-logger)
    -   [Use SwanLab Logger](#use-swanlab-logger)
-   [Projects using LLaMA Factory](#projects-using-llama-factory)
-   [License](#license)
-   [Citation](#citation)
-   [Acknowledgement](#acknowledgement)
-   [Star History](#star-history)

## Supported Models

A comprehensive list of supported models is available [here](#supported-models).

## Supported Training Approaches

The supported training approaches are available [here](#supported-training-approaches).

## Quickstart

Get started quickly with these 3 commands for fine-tuning, inference, and merging the Llama3-8B-Instruct model.

```bash
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
llamafactory-cli chat examples/inference/llama3_lora_sft.yaml
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
```

See [examples/README.md](examples/README.md) for advanced usage, including distributed training.

> [!TIP]
> Use `llamafactory-cli help` to show help information.
>
> Read [FAQs](https://github.com/hiyouga/LLaMA-Factory/issues/4614) first if you encounter any problems.

## Getting Started

### Installation

Comprehensive installation instructions can be found [here](#installation).

### Data Preparation

Details on data preparation, including dataset formats, can be found [here](#data-preparation).

### Fine-Tuning with LLaMA Board GUI (powered by [Gradio](https://github.com/gradio-app/gradio))

```bash
llamafactory-cli webui
```

### Build Docker

Instructions for building the Docker image can be found [here](#build-docker).

### Deploy with OpenAI-style API and vLLM

```bash
API_PORT=8000 llamafactory-cli api examples/inference/llama3.yaml infer_backend=vllm vllm_enforce_eager=true
```

> [!TIP]
> Visit [this page](https://platform.openai.com/docs/api-reference/chat/create) for API document.
>
> Examples: [Image understanding](scripts/api_example/test_image.py) | [Function calling](scripts/api_example/test_toolcall.py)

### Download from ModelScope Hub

Instructions for downloading models and datasets from ModelScope Hub are available [here](#download-from-modelscope-hub).

### Download from Modelers Hub

Instructions for downloading models and datasets from Modelers Hub are available [here](#download-from-modelers-hub).

### Use W&B Logger

Instructions on integrating Weights & Biases (W&B) logging are available [here](#use-wb-logger).

### Use SwanLab Logger

Instructions on integrating SwanLab logging are available [here](#use-swanlab-logger).

## Projects using LLaMA Factory

A list of projects utilizing LLaMA Factory can be found [here](#projects-using-llama-factory).

## License

This project is licensed under the [Apache-2.0 License](LICENSE).

## Citation

If this work is helpful, please kindly cite as:

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

This project benefits from the contributions of PEFT, TRL, QLoRA, and FastChat. We thank them for their work.

## Star History

The star history chart is included in the  [Star History](#star-history) section.