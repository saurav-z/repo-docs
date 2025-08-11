<!-- Oumi: Build State-of-the-Art Foundation Models, End-to-End -->
<!-- SEO-optimized README -->

# Oumi: Your Open-Source Platform for Building Foundation Models

**Unlock the power of state-of-the-art foundation models with Oumi, a comprehensive, open-source platform for data preparation, training, evaluation, and deployment. 🚀**

[![Oumi Logo](https://github.com/oumi-ai/oumi/raw/main/docs/_static/logo/header_logo.png)](https://github.com/oumi-ai/oumi)

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

<p align="center">
  <a href="https://trendshift.io/repositories/12865">
    <img alt="GitHub trending" src="https://trendshift.io/api/badge/repositories/12865" />
  </a>
</p>

**[Explore the Oumi GitHub Repository](https://github.com/oumi-ai/oumi)**

## Key Features

*   **End-to-End Workflow:** Streamline your entire foundation model lifecycle from data preparation to deployment.
*   **Training & Fine-tuning:** Train and fine-tune models from 10M to 405B parameters using SOTA techniques.
*   **Model Support:** Works with both text and multimodal models, including Llama, DeepSeek, Qwen, and Phi.
*   **Data Curation:** Synthesize and curate training data with LLM judges.
*   **Efficient Deployment:** Deploy models efficiently with popular inference engines like vLLM and SGLang.
*   **Comprehensive Evaluation:** Evaluate models across standard benchmarks.
*   **Flexible Deployment:** Run Oumi on laptops, clusters, and clouds (AWS, Azure, GCP, Lambda).
*   **Integration:** Integrates with open models and commercial APIs (OpenAI, Anthropic, Vertex AI, etc.)
*   **Production-Ready:** Provides a consistent API and production-grade reliability.

## News

*   **August 14 Webinar:** [OpenAI's gpt-oss: Separating the Substance from the Hype](https://lu.ma/qd9fhau9?utm_source=oumi_readme) - Limited spots available!
*   **Oumi v0.3.0 Released:** Includes model quantization (AWQ), an improved LLM-as-a-Judge API, and Adaptive Inference.
*   **Qwen3 235B Recipe:** Available for training.
*   **July 24 Webinar Recording:** ["Training a State-of-the-art Agent LLM with Oumi + Lambda"](https://youtu.be/f3SU_heBP54).
*   **Oumi v0.2.0 Released:** Includes GRPO fine-tuning, new model support, and more.
*   **DCVLR Competition Announcement:** Data Curation for Vision Language Models (DCVLR) at NeurIPS2025.
*   **Recipes for Falcon Models:** Training, inference, and evaluation recipes for Falcon-H1 and Falcon-E models.
*   **InternVL3 1B Support:** Recipes for InternVL3 1B.
*   **Llama 4 Support:** Support for Llama 4 models (Scout and Maverick) including full fine-tuning, LoRA, and QLoRA configurations.
*   **Qwen3 Recipes:** Recipes for the Qwen3 model family.
*   **HallOumi Introduction:** A State-of-the-Art Claim-Verification Model [(technical overview)](https://oumi.ai/blog/posts/introducing-halloumi).
*   **Vision-Language Model Support:** Support for Phi4 and Qwen 2.5 models.

## About Oumi

Oumi is an open-source platform designed to simplify the entire lifecycle of foundation models. From data preparation and training to evaluation and deployment, Oumi provides the tools and workflows you need to develop and deploy models efficiently.  Whether you are developing on your laptop, scaling up your experiments, or deploying in production, Oumi simplifies the process and provides consistent and reliable tools.

Learn more at [oumi.ai](https://oumi.ai/docs) or jump right in with the [quickstart guide](https://oumi.ai/docs/en/latest/get_started/quickstart.html).

## Getting Started

| Notebook                      | Try in Colab                                                                                                                                 | Goal                                                                                     |
| :---------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------- |
| **🎯 Getting Started: A Tour** | <a target="_blank" href="https://colab.research.google.com/github/oumi-ai/oumi/blob/main/notebooks/Oumi - A Tour.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | Quick tour of core features: training, evaluation, inference, and job management            |
| **🔧 Model Finetuning Guide**  | <a target="_blank" href="https://colab.research.google.com/github/oumi-ai/oumi/blob/main/notebooks/Oumi - Finetuning Tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | End-to-end guide to LoRA tuning with data prep, training, and evaluation                  |
| **📚 Model Distillation**     | <a target="_blank" href="https://colab.research.google.com/github/oumi-ai/oumi/blob/main/notebooks/Oumi - Distill a Large Model.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | Guide to distilling large models into smaller, efficient ones                              |
| **📋 Model Evaluation**      | <a target="_blank" href="https://colab.research.google.com/github/oumi-ai/oumi/blob/main/notebooks/Oumi - Evaluation with Oumi.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | Comprehensive model evaluation using Oumi's evaluation framework                         |
| **☁️ Remote Training**        | <a target="_blank" href="https://colab.research.google.com/github/oumi-ai/oumi/blob/main/notebooks/Oumi - Running Jobs Remotely.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | Launch and monitor training jobs on cloud (AWS, Azure, GCP, Lambda, etc.) platforms |
| **📈 LLM-as-a-Judge**         | <a target="_blank" href="https://colab.research.google.com/github/oumi-ai/oumi/blob/main/notebooks/Oumi - Simple Judge.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | Filter and curate training data with built-in judges                                     |

## Usage

### Installation

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

### Oumi CLI

Use the `oumi` CLI to train, evaluate, and infer models:

```bash
# Training
oumi train -c configs/recipes/smollm/sft/135m/quickstart_train.yaml

# Evaluation
oumi evaluate -c configs/recipes/smollm/evaluation/135m/quickstart_eval.yaml

# Inference
oumi infer -c configs/recipes/smollm/inference/135m_infer.yaml --interactive
```

See the [training](https://oumi.ai/docs/en/latest/user_guides/train/train.html), [evaluation](https://oumi.ai/docs/en/latest/user_guides/evaluate/evaluate.html), [inference](https://oumi.ai/docs/en/latest/user_guides/infer/infer.html), and [llm-as-a-judge](https://oumi.ai/docs/en/latest/user_guides/judge/judge.html) guides for advanced options.

### Running Jobs Remotely

Run jobs on cloud platforms:

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

**Note:** Oumi is in <ins>beta</ins> and under active development. The core features are stable, but some advanced features might change as the platform improves.

## Why Use Oumi?

Oumi provides a comprehensive platform for training, evaluating, and deploying foundation models.

*   **Zero Boilerplate:** Ready-to-use recipes get you started in minutes.
*   **Enterprise-Grade:** Built and validated for large-scale model training.
*   **Research Ready:** Reproducible experiments and flexible interfaces.
*   **Broad Model Support:** Works with various model architectures, text-only, and multimodal models.
*   **SOTA Performance:** Native support for distributed training and optimized inference engines.
*   **Community-First:** 100% open source with an active community.

## Examples & Recipes

Explore ready-to-use configurations:

**Note:** Find a more exhaustive list of supported [models](https://oumi.ai/docs/en/latest/resources/models/supported_models.html) and datasets in the oumi documentation.

### Qwen Family

| Model         | Example Configurations                                                                                             |
| :------------ | :----------------------------------------------------------------------------------------------------------------- |
| Qwen3 30B A3B | [LoRA](/configs/recipes/qwen3/sft/30b_a3b_lora/train.yaml) • [Inference](/configs/recipes/qwen3/inference/30b_a3b_infer.yaml) • [Evaluation](/configs/recipes/qwen3/evaluation/30b_a3b_eval.yaml) |
| Qwen3 32B     | [LoRA](/configs/recipes/qwen3/sft/32b_lora/train.yaml) • [Inference](/configs/recipes/qwen3/inference/32b_infer.yaml) • [Evaluation](/configs/recipes/qwen3/evaluation/32b_eval.yaml) |
| QwQ 32B       | [FFT](/configs/recipes/qwq/sft/full_train.yaml) • [LoRA](/configs/recipes/qwq/sft/lora_train.yaml) • [QLoRA](/configs/recipes/qwq/sft/qlora_train.yaml) • [Inference](/configs/recipes/qwq/inference/infer.yaml) • [Evaluation](/configs/recipes/qwq/evaluation/eval.yaml) |
| Qwen2.5-VL 3B | [SFT](/configs/recipes/vision/qwen2_5_vl_3b/sft/full/train.yaml) • [LoRA](/configs/recipes/vision/qwen2_5_vl_3b/sft/lora/train.yaml)• [Inference (vLLM)](configs/recipes/vision/qwen2_5_vl_3b/inference/vllm_infer.yaml) • [Inference](configs/recipes/vision/qwen2_5_vl_3b/inference/infer.yaml) |
| Qwen2-VL 2B   | [SFT](/configs/recipes/vision/qwen2_vl_2b/sft/full/train.yaml) • [LoRA](/configs/recipes/vision/qwen2_vl_2b/sft/lora/train.yaml) • [Inference (vLLM)](configs/recipes/vision/qwen2_vl_2b/inference/vllm_infer.yaml) • [Inference (SGLang)](configs/recipes/vision/qwen2_vl_2b/inference/sglang_infer.yaml) • [Inference](configs/recipes/vision/qwen2_vl_2b/inference/infer.yaml) • [Evaluation](configs/recipes/vision/qwen2_vl_2b/evaluation/eval.yaml) |

### DeepSeek R1 Family

| Model                     | Example Configurations                                                                                |
| :------------------------ | :----------------------------------------------------------------------------------------------------- |
| DeepSeek R1 671B          | [Inference (Together AI)](configs/recipes/deepseek_r1/inference/671b_together_infer.yaml)            |
| Distilled Llama 8B        | [FFT](/configs/recipes/deepseek_r1/sft/distill_llama_8b/full_train.yaml) • [LoRA](/configs/recipes/deepseek_r1/sft/distill_llama_8b/lora_train.yaml) • [QLoRA](/configs/recipes/deepseek_r1/sft/distill_llama_8b/qlora_train.yaml) • [Inference](configs/recipes/deepseek_r1/inference/distill_llama_8b_infer.yaml) • [Evaluation](/configs/recipes/deepseek_r1/evaluation/distill_llama_8b/eval.yaml) |
| Distilled Llama 70B       | [FFT](/configs/recipes/deepseek_r1/sft/distill_llama_70b/full_train.yaml) • [LoRA](/configs/recipes/deepseek_r1/sft/distill_llama_70b/lora_train.yaml) • [QLoRA](/configs/recipes/deepseek_r1/sft/distill_llama_70b/qlora_train.yaml) • [Inference](configs/recipes/deepseek_r1/inference/distill_llama_70b_infer.yaml) • [Evaluation](/configs/recipes/deepseek_r1/evaluation/distill_llama_70b/eval.yaml) |
| Distilled Qwen 1.5B       | [FFT](/configs/recipes/deepseek_r1/sft/distill_qwen_1_5b/full_train.yaml) • [LoRA](/configs/recipes/deepseek_r1/sft/distill_qwen_1_5b/lora_train.yaml) • [Inference](configs/recipes/deepseek_r1/inference/distill_qwen_1_5b_infer.yaml) • [Evaluation](/configs/recipes/deepseek_r1/evaluation/distill_qwen_1_5b/eval.yaml) |
| Distilled Qwen 32B        | [LoRA](/configs/recipes/deepseek_r1/sft/distill_qwen_32b/lora_train.yaml) • [Inference](configs/recipes/deepseek_r1/inference/distill_qwen_32b_infer.yaml) • [Evaluation](/configs/recipes/deepseek_r1/evaluation/distill_qwen_32b/eval.yaml) |

### Llama Family

| Model                         | Example Configurations                                                                                                    |
| :---------------------------- | :------------------------------------------------------------------------------------------------------------------------ |
| Llama 4 Scout Instruct 17B    | [FFT](/configs/recipes/llama4/sft/scout_instruct_full/train.yaml) • [LoRA](/configs/recipes/llama4/sft/scout_instruct_lora/train.yaml) • [QLoRA](/configs/recipes/llama4/sft/scout_instruct_qlora/train.yaml) • [Inference (vLLM)](/configs/recipes/llama4/inference/scout_instruct_vllm_infer.yaml) • [Inference](/configs/recipes/llama4/inference/scout_instruct_infer.yaml) • [Inference (Together.ai)](/configs/recipes/llama4/inference/scout_instruct_together_infer.yaml) |
| Llama 4 Scout 17B             | [FFT](/configs/recipes/llama4/sft/scout_base_full/train.yaml)                                                              |
| Llama 3.1 8B                  | [FFT](/configs/recipes/llama3_1/sft/8b_full/train.yaml) • [LoRA](/configs/recipes/llama3_1/sft/8b_lora/train.yaml) • [QLoRA](/configs/recipes/llama3_1/sft/8b_qlora/train.yaml) • [Pre-training](/configs/recipes/llama3_1/pretraining/8b/train.yaml) • [Inference (vLLM)](configs/recipes/llama3_1/inference/8b_rvllm_infer.yaml) • [Inference](/configs/recipes/llama3_1/inference/8b_infer.yaml) • [Evaluation](/configs/recipes/llama3_1/evaluation/8b_eval.yaml) |
| Llama 3.1 70B                 | [FFT](/configs/recipes/llama3_1/sft/70b_full/train.yaml) • [LoRA](/configs/recipes/llama3_1/sft/70b_lora/train.yaml) • [QLoRA](/configs/recipes/llama3_1/sft/70b_qlora/train.yaml) • [Inference](/configs/recipes/llama3_1/inference/70b_infer.yaml) • [Evaluation](/configs/recipes/llama3_1/evaluation/70b_eval.yaml) |
| Llama 3.1 405B                | [FFT](/configs/recipes/llama3_1/sft/405b_full/train.yaml) • [LoRA](/configs/recipes/llama3_1/sft/405b_lora/train.yaml) • [QLoRA](/configs/recipes/llama3_1/sft/405b_qlora/train.yaml) |
| Llama 3.2 1B                  | [FFT](/configs/recipes/llama3_2/sft/1b_full/train.yaml) • [LoRA](/configs/recipes/llama3_2/sft/1b_lora/train.yaml) • [QLoRA](/configs/recipes/llama3_2/sft/1b_qlora/train.yaml) • [Inference (vLLM)](/configs/recipes/llama3_2/inference/1b_vllm_infer.yaml) • [Inference (SGLang)](/configs/recipes/llama3_2/inference/1b_sglang_infer.yaml) • [Inference](/configs/recipes/llama3_2/inference/1b_infer.yaml) • [Evaluation](/configs/recipes/llama3_2/evaluation/1b_eval.yaml) |
| Llama 3.2 3B                  | [FFT](/configs/recipes/llama3_2/sft/3b_full/train.yaml) • [LoRA](/configs/recipes/llama3_2/sft/3b_lora/train.yaml) • [QLoRA](/configs/recipes/llama3_2/sft/3b_qlora/train.yaml) • [Inference (vLLM)](/configs/recipes/llama3_2/inference/3b_vllm_infer.yaml) • [Inference (SGLang)](/configs/recipes/llama3_2/inference/3b_sglang_infer.yaml) • [Inference](/configs/recipes/llama3_2/inference/3b_infer.yaml) • [Evaluation](/configs/recipes/llama3_2/evaluation/3b_eval.yaml) |
| Llama 3.3 70B                 | [FFT](/configs/recipes/llama3_3/sft/70b_full/train.yaml) • [LoRA](/configs/recipes/llama3_3/sft/70b_lora/train.yaml) • [QLoRA](/configs/recipes/llama3_3/sft/70b_qlora/train.yaml) • [Inference (vLLM)](/configs/recipes/llama3_3/inference/70b_vllm_infer.yaml) • [Inference](/configs/recipes/llama3_3/inference/70b_infer.yaml) • [Evaluation](/configs/recipes/llama3_3/evaluation/70b_eval.yaml) |
| Llama 3.2 Vision 11B          | [SFT](/configs/recipes/vision/llama3_2_vision/sft/11b_full/train.yaml) • [Inference (vLLM)](/configs/recipes/vision/llama3_2_vision/inference/11b_rvllm_infer.yaml) • [Inference (SGLang)](/configs/recipes/vision/llama3_2_vision/inference/11b_sglang_infer.yaml) • [Evaluation](/configs/recipes/vision/llama3_2_vision/evaluation/11b_eval.yaml) |

### Falcon Family

| Model                       | Example Configurations                                                                                                                                            |
| :-------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Falcon-H1](https://huggingface.co/collections/tiiuae/falcon-h1-6819f2795bc406da60fab8df) | [FFT](/configs/recipes/falcon_h1/sft/) • [Inference](/configs/recipes/falcon_h1/inference/) • [Evaluation](/configs/recipes/falcon_h1/evaluation/)                                   |
| [Falcon-E (BitNet)](https://huggingface.co/collections/tiiuae/falcon-edge-series-6804fd13344d6d8a8fa71130) | [FFT](/configs/recipes/falcon_e/sft/) • [DPO](/configs/recipes/falcon_e/dpo/) • [Evaluation](/configs/recipes/falcon_e/evaluation/)                                               |

### Vision Models

| Model                       | Example Configurations                                                                                                                                                                                    |
| :-------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Llama 3.2 Vision 11B          | [SFT](/configs/recipes/vision/llama3_2_vision/sft/11b_full/train.yaml) • [LoRA](/configs/recipes/vision/llama3_2_vision/sft/11b_lora/train.yaml) • [Inference (vLLM)](/configs/recipes/vision/llama3_2_vision/inference/11b_rvllm_infer.yaml) • [Inference (SGLang)](/configs/recipes/vision/llama3_2_vision/inference/11b_sglang_infer.yaml) • [Evaluation](/configs/recipes/vision/llama3_2_vision/evaluation/11b_eval.yaml) |
| LLaVA 7B                    | [SFT](/configs/recipes/vision/llava_7b/sft/train.yaml) • [Inference (vLLM)](configs/recipes/vision/llava_7b/inference/vllm_infer.yaml) • [Inference](/configs/recipes/vision/llava_7b/inference/infer.yaml) |
| Phi3 Vision 4.2B            | [SFT](/configs/recipes/vision/phi3/sft/full/train.yaml) • [LoRA](/configs/recipes/vision/phi3/sft/lora/train.yaml) • [Inference (vLLM)](configs/recipes/vision/phi3/inference/vllm_infer.yaml) |
| Phi4 Vision 5.6B            | [SFT](/configs/recipes/vision/phi4/sft/full/train.yaml) • [LoRA](/configs/recipes/vision/phi4/sft/lora/train.yaml) • [Inference (vLLM)](configs/recipes/vision/phi4/inference/vllm_infer.yaml) • [Inference](/configs/recipes/vision/phi4/inference/infer.yaml) |
| Qwen2-VL 2B                 | [SFT](/configs/recipes/vision/qwen2_vl_2b/sft/full/train.yaml) • [LoRA](/configs/recipes/vision/qwen2_vl_2b/sft/lora/train.yaml) • [Inference (vLLM)](configs/recipes/vision/qwen2_vl_2b/inference/vllm_infer.yaml) • [Inference (SGLang)](configs/recipes/vision/qwen2_vl_2b/inference/sglang_infer.yaml) • [Inference](configs/recipes/vision/qwen2_vl_2b/inference/infer.yaml) • [Evaluation](configs/recipes/vision/qwen2_vl_2b/evaluation/eval.yaml) |
| Qwen2.5-VL 3B               | [SFT](/configs/recipes/vision/qwen2_5_vl_3b/sft/full/train.yaml) • [LoRA](/configs/recipes/vision/qwen2_5_vl_3b/sft/lora/train.yaml)• [Inference (vLLM)](configs/recipes/vision/qwen2_5_vl_3b/inference/vllm_infer.yaml) • [Inference](configs/recipes/vision/qwen2_5_vl_3b/inference/infer.yaml) |
| SmolVLM-Instruct 2B         | [SFT](/configs/recipes/vision/smolvlm/sft/full/train.yaml) • [LoRA](/configs/recipes/vision/smolvlm/sft/lora/train.yaml) |

### Even More Options

Thanks to the integration with the [🤗 Transformers](https://github.com/huggingface/transformers) library, Oumi supports a wide range of models.

<details>
<summary>📋 Click to see more supported models</summary>

#### Instruct Models

| Model                        | Size       | Paper                                                                     | HF Hub                                                                                | License      | Open [^1] | Recommended Parameters |
| :--------------------------- | :--------- | :------------------------------------------------------------------------ | :------------------------------------------------------------------------------------ | :----------- | :-------- | :--------------------- |
| ✅ SmolLM-Instruct            | 135M/360M/1.7B | [Blog](https://huggingface.co/blog/smollm)                                | [Hub](https://huggingface.co/HuggingFaceTB/SmolLM-135M-Instruct)                       | Apache 2.0   | ✅        |                      |
| ✅ DeepSeek R1 Family         | 1.5B/8B/32B/70B/671B | [Blog](https://api-docs.deepseek.com/news/news250120)               | [Hub](https://huggingface.co/deepseek-ai/DeepSeek-R1)                                | MIT          | ❌       |                      |
| ✅ Llama 3.1 Instruct         | 8B/70B/405B    | [Paper](https://arxiv.org/abs/2407.21783)                                 | [Hub](https://huggingface.co/meta-llama/Llama-3.1-70b-instruct)                    | [License](https://llama.meta.com/llama3/license/) | ❌       |                      |
| ✅ Llama 3.2 Instruct         | 1B/3B        | [Paper](https://arxiv.org/abs/2407.21783)                                 | [Hub](https://huggingface.co/meta-llama/Llama-3.2-3b-instruct)                    | [License](https://llama.meta.com/llama3/license/) | ❌       |                      |
| ✅ Llama 3.3 Instruct         | 70B          | [Paper](https://arxiv.org/abs/2407.21783)                                 | [Hub](https://huggingface.co/meta-llama/Llama-3.3-70b-instruct)                    | [License](https://llama.meta.com/llama3/license/) | ❌       |                      |
| ✅ Phi-3.5-Instruct          | 4B/14B       | [Paper](https://arxiv.org/abs/2404.14219)                                 | [Hub](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)                     | [License](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/main/LICENSE) | ❌       |                      |
| Qwen2.5-Instruct             | 0.5B-70B     | [Paper](https://arxiv.org/abs/2309.16609)                                 | [Hub](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)                               | [License](https://github.com/QwenLM/Qwen/blob/main/LICENSE) | ❌       |                      |
| OLMo 2 Instruct              | 7B           | [Paper](https://arxiv.org/abs/2402.00838)                                 | [Hub](https://huggingface.co/allenai/OLMo-2-1124-7B)                             | Apache 2.0   | ✅        |                      |
| MPT-Instruct                 | 7B           | [Blog](https://www.mosaicml.com/blog/mpt-7b)                             | [Hub](https://huggingface.co/mosaicml/mpt-7b-instruct)                          | Apache 2.0   | ✅        |                      |
| Command R                    | 35B/104B     | [Blog](https://cohere.com/blog/command-r7b)                              | [Hub](https://huggingface.co/CohereForAI/c4ai-command-r-plus)                    | [License](https://cohere.com/c4ai-cc-by-nc-license)  | ❌       |                      |
| Granite-3.1-Instruct         | 2B/8B        | [Paper](https://github.com/ibm-granite/granite-3.0-language-models/blob/main/paper.pdf) | [Hub](https://huggingface.co/ibm-granite/granite-3.1-8b-instruct)              | Apache 2.0   | ❌       |                      |
| Gemma 2 Instruct             | 2B/9B        | [Blog](https://ai.google.dev/gemma)                                       | [Hub](https://huggingface.co/google/gemma-2-2b-it)                               | [License](https://ai.google.dev/gemma/terms)  | ❌       |                      |
| DBRX-Instruct                | 130B MoE     | [Blog](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm) | [Hub](https://huggingface.co/databricks/dbrx-instruct)                          | Apache 2.0   | ❌       |                      |
| Falcon-Instruct              | 7B/40B       | [Paper](https://arxiv.org/abs/2306.01116)                                 | [Hub](https://huggingface.co/