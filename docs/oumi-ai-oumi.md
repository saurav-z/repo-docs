![Oumi Logo](https://github.com/oumi-ai/oumi/raw/main/docs/_static/logo/header_logo.png)

[![Documentation](https://img.shields.io/badge/Documentation-oumi-blue.svg)](https://oumi.ai/docs/en/latest/index.html)
[![Blog](https://img.shields.io/badge/Blog-oumi-blue.svg)](https://oumi.ai/blog)
[![Twitter](https://img.shields.io/twitter/follow/Oumi_PBC)](https://x.com/Oumi_PBC)
[![Discord](https://img.shields.io/discord/1286348126797430814?label=Discord)](https://discord.gg/oumi)
[![PyPI version](https://badge.fury.io/py/oumi.svg)](https://badge.fury.io/py/oumi)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests](https://github.com/oumi-ai/oumi/actions/workflows/pretest.yaml/badge.svg?branch=main)](https://github.com/oumi-ai/oumi/actions/workflows/pretest.yaml)
[![GPU Tests](https://github.com/oumi-ai/oumi/actions/workflows/gpu_tests.yaml/badge.svg?branch=main)](https://github.com/oumi-ai/oumi/actions/workflows/gpu_tests.yaml)
[![GitHub Repo stars](https://img.shields.io/github/stars/oumi-ai/oumi)](https://github.com/oumi-ai/oumi/stargazers)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![About](https://img.shields.io/badge/About-oumi-blue.svg)](https://oumi.ai)

## Oumi: Build and Deploy State-of-the-Art Foundation Models End-to-End

Oumi is a fully open-source platform that empowers you to build, train, evaluate, and deploy foundation models with ease.  [Explore the Oumi repository on GitHub](https://github.com/oumi-ai/oumi) to get started!

## Key Features

*   **Simplified Workflow:** Streamline your model lifecycle with tools for data preparation, training, evaluation, and deployment.
*   **Model Versatility:**  Supports both text and multimodal models, including Llama, DeepSeek, Qwen, Phi, and more.
*   **Advanced Training Techniques:**  Train and fine-tune models from 10M to 405B parameters using SFT, LoRA, QLoRA, DPO, and other cutting-edge methods.
*   **Data Curation:**  Synthesize and curate training data with LLM judges.
*   **Efficient Deployment:**  Deploy models using popular inference engines like vLLM and SGLang.
*   **Comprehensive Evaluation:**  Evaluate models across standard benchmarks.
*   **Flexible Infrastructure:** Run anywhere - from laptops to clusters to clouds (AWS, Azure, GCP, Lambda, etc.).
*   **Broad Integration:** Integrate with both open models and commercial APIs (OpenAI, Anthropic, Vertex AI, Together, Parasail, ...).
*   **Community-Driven:**  Benefit from a 100% open-source platform with an active community.

## What's New

*   **[2025/07]** Recipe for [Qwen3 235B](https://github.com/oumi-ai/oumi/blob/main/configs/recipes/qwen3/inference/235b_a22b_together_infer.yaml)
*   **[2025/07]** July 24 webinar: ["Training a State-of-the-art Agent LLM with Oumi + Lambda"](https://youtu.be/f3SU_heBP54)
*   **[2025/06]** [Oumi v0.2.0 released](https://github.com/oumi-ai/oumi/releases/tag/v0.2.0) with support for GRPO fine-tuning, a plethora of new model support, and much more
*   **[2025/06]** Announcement of [Data Curation for Vision Language Models (DCVLR) competition](https://oumi.ai/blog/posts/announcing-dcvlr) at NeurIPS2025
*   **[2025/06]** Recipes for training, inference, and eval with the newly released [Falcon-H1](https://github.com/oumi-ai/oumi/tree/main/configs/recipes/falcon_h1) and [Falcon-E](https://github.com/oumi-ai/oumi/tree/main/configs/recipes/falcon_e) models
*   **[2025/05]** Support and recipes for [InternVL3 1B](https://github.com/oumi-ai/oumi/tree/main/configs/recipes/vision/internvl3)
*   **[2025/04]** Added support for training and inference with Llama 4 models: Scout (17B activated, 109B total) and Maverick (17B activated, 400B total) variants, including full fine-tuning, LoRA, and QLoRA configurations
*   **[2025/04]** Recipes for [Qwen3 model family](https://github.com/oumi-ai/oumi/tree/main/configs/recipes/qwen3)
*   **[2025/04]** Introducing HallOumi: a State-of-the-Art Claim-Verification Model [(technical overview)](https://oumi.ai/blog/posts/introducing-halloumi)
*   **[2025/04]** Oumi now supports two new Vision-Language models: [Phi4](https://github.com/oumi-ai/oumi/tree/main/configs/recipes/vision/phi4) and [Qwen 2.5](https://github.com/oumi-ai/oumi/tree/main/configs/recipes/vision/qwen2_5_vl_3b)

## Getting Started

### Notebooks

| Notebook                          | Try in Colab                                                                                                                                | Goal                                                                                                                                                                     |
| :-------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **🎯 Getting Started: A Tour**   | <a target="_blank" href="https://colab.research.google.com/github/oumi-ai/oumi/blob/main/notebooks/Oumi - A Tour.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | Quick tour of core features: training, evaluation, inference, and job management                                                                                           |
| **🔧 Model Finetuning Guide**    | <a target="_blank" href="https://colab.research.google.com/github/oumi-ai/oumi/blob/main/notebooks/Oumi - Finetuning Tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | End-to-end guide to LoRA tuning with data prep, training, and evaluation                                                                                                  |
| **📚 Model Distillation**         | <a target="_blank" href="https://colab.research.google.com/github/oumi-ai/oumi/blob/main/notebooks/Oumi - Distill a Large Model.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | Guide to distilling large models into smaller, efficient ones                                                                                                               |
| **📋 Model Evaluation**          | <a target="_blank" href="https://colab.research.google.com/github/oumi-ai/oumi/blob/main/notebooks/Oumi - Evaluation with Oumi.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | Comprehensive model evaluation using Oumi's evaluation framework                                                                                                             |
| **☁️ Remote Training**           | <a target="_blank" href="https://colab.research.google.com/github/oumi-ai/oumi/blob/main/notebooks/Oumi - Running Jobs Remotely.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | Launch and monitor training jobs on cloud (AWS, Azure, GCP, Lambda, etc.) platforms                                                                                       |
| **📈 LLM-as-a-Judge**             | <a target="_blank" href="https://colab.research.google.com/github/oumi-ai/oumi/blob/main/notebooks/Oumi - Simple Judge.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | Filter and curate training data with built-in judges                                                                                                                     |

## Installation

Install Oumi with the following commands:

```bash
# Install the package (CPU & NPU only)
pip install oumi  # For local development & testing

# OR, with GPU support (Requires Nvidia or AMD GPU)
pip install oumi[gpu]  # For GPU training

# To get the latest version, install from the source
pip install git+https://github.com/oumi-ai/oumi.git
```

For advanced installation options, see the [installation guide](https://oumi.ai/docs/en/latest/get_started/installation.html).

## Usage

### Oumi CLI

Use the `oumi` command-line interface for training, evaluation, and inference with existing recipes:

```bash
# Training
oumi train -c configs/recipes/smollm/sft/135m/quickstart_train.yaml

# Evaluation
oumi evaluate -c configs/recipes/smollm/evaluation/135m/quickstart_eval.yaml

# Inference
oumi infer -c configs/recipes/smollm/inference/135m_infer.yaml --interactive
```

For advanced options, see the [training](https://oumi.ai/docs/en/latest/user_guides/train/train.html), [evaluation](https://oumi.ai/docs/en/latest/user_guides/evaluate/evaluate.html), [inference](https://oumi.ai/docs/en/latest/user_guides/infer/infer.html), and [llm-as-a-judge](https://oumi.ai/docs/en/latest/user_guides/judge/judge.html) guides.

### Running Jobs Remotely

Launch jobs on cloud platforms (AWS, Azure, GCP, Lambda, etc.) using `oumi launch`:

```bash
# GCP
oumi launch up -c configs/recipes/smollm/sft/135m/quickstart_gcp_job.yaml

# AWS
oumi launch up -c configs/recipes/smollm/sft/135m/quickstart_gcp_job.yaml --resources.cloud aws

# Azure
oumi launch up -c configs/recipes/smollm/sft/135m/quickstart_gcp_job.yaml --resources.cloud azure

# Lambda
oumi launch up -c configs/recipes/smollm/sft/135m/quickstart_gcp_job.yaml --resources.cloud lambda
```

**Note:**  Oumi is in beta and under active development.  Core features are stable; however, some advanced features might change.

## Why Use Oumi?

Oumi is a great choice for a comprehensive platform for training, evaluating, or deploying foundation models.

*   **Zero Boilerplate**: Get started quickly with pre-built recipes for popular models and workflows.
*   **Enterprise-Grade**: Built and validated by teams training models at scale.
*   **Research-Ready**:  Ideal for ML research with easily reproducible experiments and customizable interfaces.
*   **Broad Model Support**: Supports a wide range of model architectures, from small to large, text-only to multimodal.
*   **SOTA Performance**:  Includes built-in support for distributed training (FSDP, DDP) and optimized inference engines (vLLM, SGLang).
*   **Community-First**: 100% open source with an active community for support and collaboration.

## Examples & Recipes

Explore ready-to-use configurations for state-of-the-art models and training workflows.  See the documentation for a more complete list of supported [models](https://oumi.ai/docs/en/latest/resources/models/supported_models.html), [datasets](https://oumi.ai/docs/en/latest/resources/datasets/sft_datasets.html).

### Qwen Family

| Model            | Example Configurations                                                                                                                                 |
| :--------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Qwen3 30B A3B    | [LoRA](/configs/recipes/qwen3/sft/30b_a3b_lora/train.yaml) • [Inference](/configs/recipes/qwen3/inference/30b_a3b_infer.yaml) • [Evaluation](/configs/recipes/qwen3/evaluation/30b_a3b_eval.yaml) |
| Qwen3 32B        | [LoRA](/configs/recipes/qwen3/sft/32b_lora/train.yaml) • [Inference](/configs/recipes/qwen3/inference/32b_infer.yaml) • [Evaluation](/configs/recipes/qwen3/evaluation/32b_eval.yaml) |
| QwQ 32B          | [FFT](/configs/recipes/qwq/sft/full_train.yaml) • [LoRA](/configs/recipes/qwq/sft/lora_train.yaml) • [QLoRA](/configs/recipes/qwq/sft/qlora_train.yaml) • [Inference](/configs/recipes/qwq/inference/infer.yaml) • [Evaluation](/configs/recipes/qwq/evaluation/eval.yaml) |
| Qwen2.5-VL 3B    | [SFT](/configs/recipes/vision/qwen2_5_vl_3b/sft/full/train.yaml) • [LoRA](/configs/recipes/vision/qwen2_5_vl_3b/sft/lora/train.yaml)• [Inference (vLLM)](configs/recipes/vision/qwen2_5_vl_3b/inference/vllm_infer.yaml) • [Inference](configs/recipes/vision/qwen2_5_vl_3b/inference/infer.yaml) |
| Qwen2-VL 2B      | [SFT](/configs/recipes/vision/qwen2_vl_2b/sft/full/train.yaml) • [LoRA](/configs/recipes/vision/qwen2_vl_2b/sft/lora/train.yaml) • [Inference (vLLM)](configs/recipes/vision/qwen2_vl_2b/inference/vllm_infer.yaml) • [Inference (SGLang)](configs/recipes/vision/qwen2_vl_2b/inference/sglang_infer.yaml) • [Inference](configs/recipes/vision/qwen2_vl_2b/inference/infer.yaml) • [Evaluation](configs/recipes/vision/qwen2_vl_2b/evaluation/eval.yaml) |

### DeepSeek R1 Family

| Model                  | Example Configurations                                                                                                                                      |
| :--------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| DeepSeek R1 671B       | [Inference (Together AI)](configs/recipes/deepseek_r1/inference/671b_together_infer.yaml)                                                                   |
| Distilled Llama 8B     | [FFT](/configs/recipes/deepseek_r1/sft/distill_llama_8b/full_train.yaml) • [LoRA](/configs/recipes/deepseek_r1/sft/distill_llama_8b/lora_train.yaml) • [QLoRA](/configs/recipes/deepseek_r1/sft/distill_llama_8b/qlora_train.yaml) • [Inference](configs/recipes/deepseek_r1/inference/distill_llama_8b_infer.yaml) • [Evaluation](/configs/recipes/deepseek_r1/evaluation/distill_llama_8b/eval.yaml) |
| Distilled Llama 70B    | [FFT](/configs/recipes/deepseek_r1/sft/distill_llama_70b/full_train.yaml) • [LoRA](/configs/recipes/deepseek_r1/sft/distill_llama_70b/lora_train.yaml) • [QLoRA](/configs/recipes/deepseek_r1/sft/distill_llama_70b/qlora_train.yaml) • [Inference](configs/recipes/deepseek_r1/inference/distill_llama_70b_infer.yaml) • [Evaluation](/configs/recipes/deepseek_r1/evaluation/distill_llama_70b/eval.yaml) |
| Distilled Qwen 1.5B    | [FFT](/configs/recipes/deepseek_r1/sft/distill_qwen_1_5b/full_train.yaml) • [LoRA](/configs/recipes/deepseek_r1/sft/distill_qwen_1_5b/lora_train.yaml) • [Inference](configs/recipes/deepseek_r1/inference/distill_qwen_1_5b_infer.yaml) • [Evaluation](/configs/recipes/deepseek_r1/evaluation/distill_qwen_1_5b/eval.yaml) |
| Distilled Qwen 32B     | [LoRA](/configs/recipes/deepseek_r1/sft/distill_qwen_32b/lora_train.yaml) • [Inference](configs/recipes/deepseek_r1/inference/distill_qwen_32b_infer.yaml) • [Evaluation](/configs/recipes/deepseek_r1/evaluation/distill_qwen_32b/eval.yaml) |

### 🦙 Llama Family

| Model                     | Example Configurations                                                                                                                                                                            |
| :------------------------ | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Llama 4 Scout Instruct 17B | [FFT](/configs/recipes/llama4/sft/scout_instruct_full/train.yaml) • [LoRA](/configs/recipes/llama4/sft/scout_instruct_lora/train.yaml) • [QLoRA](/configs/recipes/llama4/sft/scout_instruct_qlora/train.yaml) • [Inference (vLLM)](/configs/recipes/llama4/inference/scout_instruct_vllm_infer.yaml) • [Inference](/configs/recipes/llama4/inference/scout_instruct_infer.yaml) • [Inference (Together.ai)](/configs/recipes/llama4/inference/scout_instruct_together_infer.yaml) |
| Llama 4 Scout 17B         | [FFT](/configs/recipes/llama4/sft/scout_base_full/train.yaml)                                                                                                                                   |
| Llama 3.1 8B              | [FFT](/configs/recipes/llama3_1/sft/8b_full/train.yaml) • [LoRA](/configs/recipes/llama3_1/sft/8b_lora/train.yaml) • [QLoRA](/configs/recipes/llama3_1/sft/8b_qlora/train.yaml) • [Pre-training](/configs/recipes/llama3_1/pretraining/8b/train.yaml) • [Inference (vLLM)](configs/recipes/llama3_1/inference/8b_rvllm_infer.yaml) • [Inference](/configs/recipes/llama3_1/inference/8b_infer.yaml) • [Evaluation](/configs/recipes/llama3_1/evaluation/8b_eval.yaml) |
| Llama 3.1 70B             | [FFT](/configs/recipes/llama3_1/sft/70b_full/train.yaml) • [LoRA](/configs/recipes/llama3_1/sft/70b_lora/train.yaml) • [QLoRA](/configs/recipes/llama3_1/sft/70b_qlora/train.yaml) • [Inference](/configs/recipes/llama3_1/inference/70b_infer.yaml) • [Evaluation](/configs/recipes/llama3_1/evaluation/70b_eval.yaml) |
| Llama 3.1 405B            | [FFT](/configs/recipes/llama3_1/sft/405b_full/train.yaml) • [LoRA](/configs/recipes/llama3_1/sft/405b_lora/train.yaml) • [QLoRA](/configs/recipes/llama3_1/sft/405b_qlora/train.yaml)                                                                                                                                         |
| Llama 3.2 1B              | [FFT](/configs/recipes/llama3_2/sft/1b_full/train.yaml) • [LoRA](/configs/recipes/llama3_2/sft/1b_lora/train.yaml) • [QLoRA](/configs/recipes/llama3_2/sft/1b_qlora/train.yaml) • [Inference (vLLM)](/configs/recipes/llama3_2/inference/1b_vllm_infer.yaml) • [Inference (SGLang)](/configs/recipes/llama3_2/inference/1b_sglang_infer.yaml) • [Inference](/configs/recipes/llama3_2/inference/1b_infer.yaml) • [Evaluation](/configs/recipes/llama3_2/evaluation/1b_eval.yaml) |
| Llama 3.2 3B              | [FFT](/configs/recipes/llama3_2/sft/3b_full/train.yaml) • [LoRA](/configs/recipes/llama3_2/sft/3b_lora/train.yaml) • [QLoRA](/configs/recipes/llama3_2/sft/3b_qlora/train.yaml) • [Inference (vLLM)](/configs/recipes/llama3_2/inference/3b_vllm_infer.yaml) • [Inference (SGLang)](/configs/recipes/llama3_2/inference/3b_sglang_infer.yaml) • [Inference](/configs/recipes/llama3_2/inference/3b_infer.yaml) • [Evaluation](/configs/recipes/llama3_2/evaluation/3b_eval.yaml) |
| Llama 3.3 70B             | [FFT](/configs/recipes/llama3_3/sft/70b_full/train.yaml) • [LoRA](/configs/recipes/llama3_3/sft/70b_lora/train.yaml) • [QLoRA](/configs/recipes/llama3_3/sft/70b_qlora/train.yaml) • [Inference (vLLM)](/configs/recipes/llama3_3/inference/70b_vllm_infer.yaml) • [Inference](/configs/recipes/llama3_3/inference/70b_infer.yaml) • [Evaluation](/configs/recipes/llama3_3/evaluation/70b_eval.yaml) |
| Llama 3.2 Vision 11B      | [SFT](/configs/recipes/vision/llama3_2_vision/sft/11b_full/train.yaml) • [Inference (vLLM)](/configs/recipes/vision/llama3_2_vision/inference/11b_rvllm_infer.yaml) • [Inference (SGLang)](/configs/recipes/vision/llama3_2_vision/inference/11b_sglang_infer.yaml) • [Evaluation](/configs/recipes/vision/llama3_2_vision/evaluation/11b_eval.yaml) |

### 🦅 Falcon family

| Model                          | Example Configurations                                                                        |
| :----------------------------- | :-------------------------------------------------------------------------------------------- |
| [Falcon-H1](https://huggingface.co/collections/tiiuae/falcon-h1-6819f2795bc406da60fab8df) | [FFT](/configs/recipes/falcon_h1/sft/) • [Inference](/configs/recipes/falcon_h1/inference/) • [Evaluation](/configs/recipes/falcon_h1/evaluation/) |
| [Falcon-E (BitNet)](https://huggingface.co/collections/tiiuae/falcon-edge-series-6804fd13344d6d8a8fa71130) | [FFT](/configs/recipes/falcon_e/sft/) • [DPO](/configs/recipes/falcon_e/dpo/) • [Evaluation](/configs/recipes/falcon_e/evaluation/) |

### 🎨 Vision Models

| Model                     | Example Configurations                                                                                                                                                                                   |
| :------------------------ | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Llama 3.2 Vision 11B      | [SFT](/configs/recipes/vision/llama3_2_vision/sft/11b_full/train.yaml) • [LoRA](/configs/recipes/vision/llama3_2_vision/sft/11b_lora/train.yaml) • [Inference (vLLM)](/configs/recipes/vision/llama3_2_vision/inference/11b_rvllm_infer.yaml) • [Inference (SGLang)](/configs/recipes/vision/llama3_2_vision/inference/11b_sglang_infer.yaml) • [Evaluation](/configs/recipes/vision/llama3_2_vision/evaluation/11b_eval.yaml) |
| LLaVA 7B                  | [SFT](/configs/recipes/vision/llava_7b/sft/train.yaml) • [Inference (vLLM)](configs/recipes/vision/llava_7b/inference/vllm_infer.yaml) • [Inference](/configs/recipes/vision/llava_7b/inference/infer.yaml)   |
| Phi3 Vision 4.2B          | [SFT](/configs/recipes/vision/phi3/sft/full/train.yaml) • [LoRA](/configs/recipes/vision/phi3/sft/lora/train.yaml) • [Inference (vLLM)](configs/recipes/vision/phi3/inference/vllm_infer.yaml)                  |
| Phi4 Vision 5.6B          | [SFT](/configs/recipes/vision/phi4/sft/full/train.yaml) • [LoRA](/configs/recipes/vision/phi4/sft/lora/train.yaml) • [Inference (vLLM)](configs/recipes/vision/phi4/inference/vllm_infer.yaml) • [Inference](/configs/recipes/vision/phi4/inference/infer.yaml)        |
| Qwen2-VL 2B               | [SFT](/configs/recipes/vision/qwen2_vl_2b/sft/full/train.yaml) • [LoRA](/configs/recipes/vision/qwen2_vl_2b/sft/lora/train.yaml) • [Inference (vLLM)](configs/recipes/vision/qwen2_vl_2b/inference/vllm_infer.yaml) • [Inference (SGLang)](configs/recipes/vision/qwen2_vl_2b/inference/sglang_infer.yaml) • [Inference](configs/recipes/vision/qwen2_vl_2b/inference/infer.yaml) • [Evaluation](configs/recipes/vision/qwen2_vl_2b/evaluation/eval.yaml) |
| Qwen2.5-VL 3B             | [SFT](/configs/recipes/vision/qwen2_5_vl_3b/sft/full/train.yaml) • [LoRA](/configs/recipes/vision/qwen2_5_vl_3b/sft/lora/train.yaml)• [Inference (vLLM)](configs/recipes/vision/qwen2_5_vl_3b/inference/vllm_infer.yaml) • [Inference](configs/recipes/vision/qwen2_5_vl_3b/inference/infer.yaml) |
| SmolVLM-Instruct 2B      | [SFT](/configs/recipes/vision/smolvlm/sft/full/train.yaml) • [LoRA](/configs/recipes/vision/smolvlm/sft/lora/train.yaml)                                                                                                                 |

### 🔍 More Models &  Options

Oumi supports many models!  Thanks to the integration with Hugging Face Transformers, you can use any of these models for training, evaluation, or inference.

<details>
<summary>📋 Click to see more supported models</summary>

#### Instruct Models

| Model | Size | Paper | HF Hub  | License  | Open [^1] | Recommended Parameters |
|-------|------|-------|---------|----------|------|------------------------|
| ✅ SmolLM-Instruct | 135M/360M/1.7B | [Blog](https://huggingface.co/blog/smollm) | [Hub](https://huggingface.co/HuggingFaceTB/SmolLM-135M-Instruct) | Apache 2.0 | ✅ | |
| ✅ DeepSeek R1 Family | 1.5B/8B/32B/70B/671B | [Blog](https://api-docs.deepseek.com/news/news250120) | [Hub](https://huggingface.co/deepseek-ai/DeepSeek-R1) | MIT | ❌ | |
| ✅ Llama 3.1 Instruct | 8B/70B/405B | [Paper](https://arxiv.org/abs/2407.21783) | [Hub](https://huggingface.co/meta-llama/Llama-3.1-70b-instruct) | [License](https://llama.meta.com/llama3/license/) | ❌  | |
| ✅ Llama 3.2 Instruct | 1B/3B | [Paper](https://arxiv.org/abs/2407.21783) | [Hub](https://huggingface.co/meta-llama/Llama-3.2-3b-instruct) | [License](https://llama.meta.com/llama3/license/) | ❌  | |
| ✅ Llama 3.3 Instruct | 70B | [Paper](https://arxiv.org/abs/2407.21783) | [Hub](https://huggingface.co/meta-llama/Llama-3.3-70b-instruct) | [License](https://llama.meta.com/llama3/license/) | ❌  | |
| ✅ Phi-3.5-Instruct | 4B/14B | [Paper](https://arxiv.org/abs/2404.14219) | [Hub](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) | [License](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/main/LICENSE) | ❌  | |
| Qwen2.5-Instruct | 0.5B-70B | [Paper](https://arxiv.org/abs/2309.16609) | [Hub](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | [License](https://github.com/QwenLM/Qwen/blob/main/LICENSE) | ❌  | |
| OLMo 2 Instruct | 7B | [Paper](https://arxiv.org/abs/2402.00838) | [Hub](https://huggingface.co/allenai/OLMo-2-1124-7B) | Apache 2.0 | ✅ | |
| MPT-Instruct | 7B | [Blog](https://www.mosaicml.com/blog/mpt-7b) | [Hub](https://huggingface.co/mosaicml/mpt-7b-instruct) | Apache 2.0 | ✅ | |
| Command R | 35B/104B | [Blog](https://cohere.com/blog/command-r7b) | [Hub](https://huggingface.co/CohereForAI/c4ai-command-r-plus) | [License](https://cohere.com/c4ai-cc-by-nc-license) | ❌ | |
| Granite-3.1-Instruct | 2B/8B | [Paper](https://github.com/ibm-granite/granite-3.0-language-models/blob/main/paper.pdf) | [Hub](https://huggingface.co/ibm-granite/granite-3.1-8b-instruct) | Apache 2.0 | ❌ | |
| Gemma 2 Instruct | 2B/9B | [Blog](https://ai.google.dev/gemma) | [Hub](https://huggingface.co/google/gemma-2-2b-it) | [License](https://ai.google.dev/gemma/terms) | ❌ | |
| DBRX-Instruct