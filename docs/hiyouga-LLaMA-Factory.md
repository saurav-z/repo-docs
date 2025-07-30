<div align="center">
  <img src="assets/logo.png" alt="LLaMA Factory Logo" width="300">
  <h1>LLaMA Factory: Fine-tune Any LLM with Ease</h1>
  <p><b>Effortlessly fine-tune 100+ large language models with zero-code CLI and a user-friendly Web UI.</b></p>
  <a href="https://github.com/hiyouga/LLaMA-Factory">
    <img src="https://img.shields.io/github/stars/hiyouga/LLaMA-Factory?style=social" alt="GitHub Stars">
  </a>
  <a href="https://github.com/hiyouga/LLaMA-Factory">
    <img src="https://img.shields.io/github/last-commit/hiyouga/LLaMA-Factory" alt="Last Commit">
  </a>
  <a href="https://github.com/hiyouga/LLaMA-Factory">
    <img src="https://img.shields.io/github/contributors/hiyouga/LLaMA-Factory?color=orange" alt="Contributors">
  </a>
  <a href="https://github.com/hiyouga/LLaMA-Factory/actions/workflows/tests.yml">
    <img src="https://github.com/hiyouga/LLaMA-Factory/actions/workflows/tests.yml/badge.svg" alt="GitHub Workflow">
  </a>
  <a href="https://pypi.org/project/llamafactory/">
    <img src="https://img.shields.io/pypi/v/llamafactory" alt="PyPI">
  </a>
  <a href="https://scholar.google.com/scholar?cites=12620864006390196564">
    <img src="https://img.shields.io/badge/citation-730-green" alt="Citation">
  </a>
  <a href="https://hub.docker.com/r/hiyouga/llamafactory/tags">
    <img src="https://img.shields.io/docker/pulls/hiyouga/llamafactory" alt="Docker Pulls">
  </a>

  <p>
    Used by <a href="https://aws.amazon.com/cn/blogs/machine-learning/how-apoidea-group-enhances-visual-information-extraction-from-banking-documents-with-multimodal-models-using-llama-factory-on-amazon-sagemaker-hyperpod/">Amazon</a>, <a href="https://developer.nvidia.com/rtx/ai-toolkit">NVIDIA</a>, <a href="https://help.aliyun.com/zh/pai/use-cases/fine-tune-a-llama-3-model-with-llama-factory">Aliyun</a>, and more.
  </p>
</div>

<hr>

**LLaMA Factory** empowers you to fine-tune a wide range of Large Language Models (LLMs) with ease, offering a streamlined experience for both beginners and experts.  Leverage cutting-edge techniques and a comprehensive feature set to customize LLMs for your specific tasks. Explore the <a href="https://github.com/hiyouga/LLaMA-Factory">original repository</a>.

**Key Features:**

*   ✅ **Extensive Model Support:** Train LLaMA, LLaVA, Mistral, Mixtral-MoE, Qwen, DeepSeek, Yi, Gemma, ChatGLM, Phi, and many more!
*   ✅ **Diverse Training Approaches:** Pre-training, supervised fine-tuning, reward modeling, PPO, DPO, KTO, ORPO, and SimPO.
*   ✅ **Flexible Training Methods:** Supports 16-bit full-tuning, freeze-tuning, LoRA, and efficient QLoRA (2/3/4/5/6/8-bit) via AQLM/AWQ/GPTQ/LLM.int8/HQQ/EETQ.
*   ✅ **Advanced Algorithms:** Includes GaLore, BAdam, APOLLO, Adam-mini, Muon, DoRA, LongLoRA, LLaMA Pro, Mixture-of-Depths, LoRA+, LoftQ, and PiSSA.
*   ✅ **Performance Optimizations:** Leverages FlashAttention-2, Unsloth, Liger Kernel, RoPE scaling, NEFTune and rsLoRA for faster and more efficient training.
*   ✅ **Versatile Task Support:** Fine-tune models for multi-turn dialogues, tool use, image understanding, video recognition, and audio understanding.
*   ✅ **Experiment Tracking and Monitoring:** Integrates with LlamaBoard, TensorBoard, Wandb, MLflow, and SwanLab for comprehensive experiment management.
*   ✅ **Rapid Inference:** Offers OpenAI-style API, Gradio UI and CLI with [vLLM worker](https://github.com/vllm-project/vllm) or [SGLang worker](https://github.com/sgl-project/sglang).
*   ✅ **Day-N Support:** Support for cutting-edge models like Qwen3, Gemma 3, GLM-4.1V, InternLM 3, MiniCPM-o-2.6, Llama 3, GLM-4, Mistral Small, PaliGemma2, and Llama 4.

<hr>

**Quick Links:**

*   <b>Documentation (WIP)</b>: <a href="https://llamafactory.readthedocs.io/en/latest/">llamafactory.readthedocs.io/en/latest/</a>
*   <b>Documentation (AMD GPU)</b>: <a href="https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/fine_tune/llama_factory_llama3.html">rocm.docs.amd.com/...llama_factory_llama3.html</a>
*   <b>Colab (free)</b>: <a href="https://colab.research.google.com/drive/1eRTPn37ltBbYsISy9Aw2NuI2Aq5CQrD9?usp=sharing">colab.research.google.com/.../1eRTPn37ltBbYsISy9Aw2NuI2Aq5CQrD9</a>
*   <b>PAI-DSW (free trial)</b>: <a href="https://gallery.pai-ml.com/#/preview/deepLearning/nlp/llama_factory">gallery.pai-ml.com/.../llama_factory</a>
*   <b>Alaya NeW (cloud GPU deal)</b>: <a href="https://docs.alayanew.com/docs/documents/useGuide/LLaMAFactory/mutiple/?utm_source=LLaMA-Factory">docs.alayanew.com/.../mutiple</a>
*   <b>Hugging Face Spaces</b>: <a href="https://huggingface.co/spaces/hiyouga/LLaMA-Board">huggingface.co/spaces/hiyouga/LLaMA-Board</a>
*   <b>ModelScope Studios</b>: <a href="https://modelscope.cn/studios/hiyouga/LLaMA-Board">modelscope.cn/studios/hiyouga/LLaMA-Board</a>

<hr>

## Table of Contents

*   [Features](#features)
*   [Supported Models](#supported-models)
*   [Supported Training Approaches](#supported-training-approaches)
*   [Provided Datasets](#provided-datasets)
*   [Requirement](#requirement)
*   [Getting Started](#getting-started)
    *   [Installation](#installation)
    *   [Data Preparation](#data-preparation)
    *   [Quickstart](#quickstart)
    *   [Fine-Tuning with LLaMA Board GUI](#fine-tuning-with-llama-board-gui-powered-by-gradio)
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
*   [Star History](#star-history)

## Supported Models

(See full list in original README)

## Supported Training Approaches

(See full list in original README)

## Provided Datasets

(See full list in original README)

## Requirement

(See full list in original README)

## Getting Started

### Installation

(See original README)

### Data Preparation

(See original README)

### Quickstart

(See original README)

### Fine-Tuning with LLaMA Board GUI (powered by [Gradio](https://github.com/gradio-app/gradio))

```bash
llamafactory-cli webui
```

### Build Docker

(See original README)

### Deploy with OpenAI-style API and vLLM

```bash
API_PORT=8000 llamafactory-cli api examples/inference/llama3.yaml infer_backend=vllm vllm_enforce_eager=true
```

### Download from ModelScope Hub

(See original README)

### Download from Modelers Hub

(See original README)

### Use W&B Logger

(See original README)

### Use SwanLab Logger

(See original README)

## Projects using LLaMA Factory

(See original README)

## License

(See original README)

## Citation

(See original README)

## Acknowledgement

(See original README)

## Star History

(See original README)