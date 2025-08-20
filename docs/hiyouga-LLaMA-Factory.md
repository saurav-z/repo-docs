<div align="center">
  <img src="assets/logo.png" alt="LLaMA Factory Logo" width="300">
  <h1>LLaMA Factory: Effortlessly Fine-Tune Large Language Models</h1>
  <p>Unleash the power of LLMs by fine-tuning 100+ models with ease using our zero-code CLI and intuitive Web UI.</p>

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

  <p>Used by <a href="https://aws.amazon.com/cn/blogs/machine-learning/how-apoidea-group-enhances-visual-information-extraction-from-banking-documents-with-multimodal-models-using-llama-factory-on-amazon-sagemaker-hyperpod/">Amazon</a>, <a href="https://developer.nvidia.com/rtx/ai-toolkit">NVIDIA</a>, <a href="https://help.aliyun.com/zh/pai/use-cases/fine-tune-a-llama-3-model-with-llama-factory">Aliyun</a>, and more.</p>

  <div markdown="1">
  <h3>Supporters ❤️</h3>

  | <div style="text-align: center;"><a href="https://warp.dev/llama-factory"><img alt="Warp sponsorship" width="400" src="assets/warp.jpg"></a><br><a href="https://warp.dev/llama-factory" style="font-size:larger;">Warp, the agentic terminal for developers</a><br><a href="https://warp.dev/llama-factory">Available for MacOS, Linux, & Windows</a> | <a href="https://serpapi.com"><img alt="SerpAPI sponsorship" width="250" src="assets/serpapi.svg"> </a> |
  | ---- | ---- |

  ----

  <img src="https://trendshift.io/api/badge/repositories/4535" alt="GitHub Trend">

  </div>

  <p>👋 Join our <a href="assets/wechat.jpg">WeChat group</a>, <a href="assets/wechat_npu.jpg">NPU user group</a> or <a href="assets/wechat_alaya.png">Alaya NeW user group</a>.</p>

  [English | <a href="README_zh.md">中文</a>]

  <b><a href="https://github.com/hiyouga/LLaMA-Factory">Explore LLaMA Factory on GitHub</a></b>

  <img src="https://github.com/user-attachments/assets/3991a3a8-4276-4d30-9cab-4cb0c4b9b99e" alt="Fine-tuning in a nutshell">

  <b>Choose your path:</b>

  *   **Documentation (WIP)**: [https://llamafactory.readthedocs.io/en/latest/](https://llamafactory.readthedocs.io/en/latest/)
  *   **Documentation (AMD GPU)**: [https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/fine_tune/llama_factory_llama3.html](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/fine_tune/llama_factory_llama3.html)
  *   **Colab (free)**: [https://colab.research.google.com/drive/1eRTPn37ltBbYsISy9Aw2NuI2Aq5CQrD9?usp=sharing](https://colab.research.google.com/drive/1eRTPn37ltBbYsISy9Aw2NuI2Aq5CQrD9?usp=sharing)
  *   **Local machine**: Please refer to [usage](#getting-started)
  *   **PAI-DSW (free trial)**: [https://gallery.pai-ml.com/#/preview/deepLearning/nlp/llama_factory](https://gallery.pai-ml.com/#/preview/deepLearning/nlp/llama_factory)
  *   **Alaya NeW (cloud GPU deal)**: [https://docs.alayanew.com/docs/documents/useGuide/LLaMAFactory/mutiple/?utm_source=LLaMA-Factory](https://docs.alayanew.com/docs/documents/useGuide/LLaMAFactory/mutiple/?utm_source=LLaMA-Factory)

  > [!NOTE]
  > Except for the above links, all other websites are unauthorized third-party websites. Please carefully use them.
</div>

## Key Features

*   **Extensive Model Support:** Fine-tune a vast range of models, including LLaMA, LLaVA, Mistral, Mixtral-MoE, Qwen, DeepSeek, Yi, and more.
*   **Versatile Training Methods:** Utilize various training approaches such as (Continuous) pre-training, (multimodal) supervised fine-tuning, reward modeling, PPO, DPO, KTO, and ORPO.
*   **Optimized Training:** Leverage efficient training techniques including full-tuning, freeze-tuning, LoRA, QLoRA, and OFT, with support for 2/3/4/5/6/8-bit quantization.
*   **Advanced Algorithms:** Access cutting-edge algorithms like GaLore, BAdam, APOLLO, and others.
*   **Practical Enhancements:** Utilize FlashAttention-2, Unsloth, Liger Kernel, RoPE scaling, NEFTune, and rsLoRA for improved performance.
*   **Wide Range of Applications:** Address diverse tasks, including multi-turn dialogue, tool usage, image understanding, and more.
*   **Comprehensive Monitoring:** Monitor experiments using LlamaBoard, TensorBoard, Wandb, MLflow, and SwanLab.
*   **Fast Inference:** Benefit from an OpenAI-style API, Gradio UI, and CLI integration with vLLM or SGLang for accelerated inference.

### Day-N Support for Fine-Tuning Cutting-Edge Models

| Support Date | Model Name                                                           |
| ------------ | -------------------------------------------------------------------- |
| Day 0        | Qwen3 / Qwen2.5-VL / Gemma 3 / GLM-4.1V / InternLM 3 / MiniCPM-o-2.6 |
| Day 1        | Llama 3 / GLM-4 / Mistral Small / PaliGemma2 / Llama 4               |

## Blogs
See the [Blogs](#blogs) Section in original Readme.

## Changelog

See the [Changelog](#changelog) Section in original Readme.

## Supported Models
See the [Supported Models](#supported-models) Section in original Readme.

## Supported Training Approaches
See the [Supported Training Approaches](#supported-training-approaches) Section in original Readme.

## Provided Datasets
See the [Provided Datasets](#provided-datasets) Section in original Readme.

## Requirements
See the [Requirements](#requirement) Section in original Readme.

## Getting Started

### Installation

Detailed instructions are provided in the [Installation](#installation) section of the original README. Key steps include:

*   Install from source, using `pip install -e ".[torch,metrics]" --no-build-isolation`
*   Using Docker Images
*   Instructions for Windows, Ascend NPU and ROCm users

### Data Preparation

*   Refer to the [data/README.md](data/README.md) for dataset formatting details.
*   You can use datasets from Hugging Face / ModelScope / Modelers hub, local disk, or cloud storage (S3/GCS).
*   Update `data/dataset_info.json` for custom datasets.

### Quickstart

Run LoRA fine-tuning, inference, and merging of the Llama3-8B-Instruct model with these three commands.

```bash
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
llamafactory-cli chat examples/inference/llama3_lora_sft.yaml
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
```

See [examples/README.md](examples/README.md) for detailed usage, including distributed training.

### Fine-Tuning with LLaMA Board GUI

```bash
llamafactory-cli webui
```

### Build Docker

See the [Build Docker](#build-docker) Section in original Readme.

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

```yaml
report_to: wandb
run_name: test_run # optional
```

Set `WANDB_API_KEY` to [your key](https://wandb.ai/authorize) when launching training tasks.

### Use SwanLab Logger

```yaml
use_swanlab: true
swanlab_run_name: test_run # optional
```

See the [Use SwanLab Logger](#use-swanlab-logger) Section in original Readme.

## Projects using LLaMA Factory

See the [Projects using LLaMA Factory](#projects-using-llama-factory) Section in original Readme.

## License

See the [License](#license) Section in original Readme.

## Citation

See the [Citation](#citation) Section in original Readme.

## Acknowledgement

See the [Acknowledgement](#acknowledgement) Section in original Readme.

## Star History

See the [Star History](#star-history) Section in original Readme.