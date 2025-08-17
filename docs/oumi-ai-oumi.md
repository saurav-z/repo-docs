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

## Oumi: Build State-of-the-Art Foundation Models, End-to-End

Oumi is a comprehensive, open-source platform designed to streamline the entire lifecycle of foundation models, from data preparation and training to evaluation and deployment. ([Back to the original repo](https://github.com/oumi-ai/oumi))

## Key Features

*   **Training and Fine-tuning:** Train and fine-tune models from 10M to 405B parameters using cutting-edge techniques (SFT, LoRA, QLoRA, DPO, and more).
*   **Model Support:** Work with both text and multimodal models, including Llama, DeepSeek, Qwen, Phi, and many others.
*   **Data Curation:** Synthesize and curate training data with LLM judges.
*   **Efficient Deployment:** Deploy models efficiently with popular inference engines (vLLM, SGLang).
*   **Comprehensive Evaluation:** Evaluate models comprehensively across standard benchmarks.
*   **Run Anywhere:** Run on various platforms - laptops, clusters, clouds (AWS, Azure, GCP, Lambda, etc.).
*   **Seamless Integration:** Integrate with both open models and commercial APIs (OpenAI, Anthropic, Vertex AI, Together, Parasail, ...).
*   **Open Source & Community Driven:** 100% open source with an active community, ensuring no vendor lock-in and full flexibility.

## What's New?

*   **Inference Support:** Added inference support for OpenAI's `gpt-oss-20b` and `gpt-oss-120b` (recipes available).
*   **Webinar:** Explore the features of `gpt-oss` in the "OpenAI's gpt-oss: Separating the Substance from the Hype" webinar.
*   **Recent Releases:** Explore releases with model quantization (AWQ), an improved LLM-as-a-Judge API, and Adaptive Inference.
*   **New Recipes & Models:** Recipes and support added for Qwen3 235B, Falcon-H1, Falcon-E, InternVL3 1B, Llama 4 models (Scout and Maverick), Qwen3 model family, and more.

## Getting Started

**Quickly get up and running with these interactive Colab notebooks:**

| Notebook                                        | Colab Link                                                                                                                          | Description                                                                                               |
| :---------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------- |
| 🎯 Getting Started: A Tour                      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oumi-ai/oumi/blob/main/notebooks/Oumi%20-%20A%20Tour.ipynb) | Quick tour of core features: training, evaluation, inference, and job management                         |
| 🔧 Model Finetuning Guide                      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oumi-ai/oumi/blob/main/notebooks/Oumi%20-%20Finetuning%20Tutorial.ipynb) | End-to-end guide to LoRA tuning with data prep, training, and evaluation                                  |
| 📚 Model Distillation                           | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oumi-ai/oumi/blob/main/notebooks/Oumi%20-%20Distill%20a%20Large%20Model.ipynb) | Guide to distilling large models into smaller, efficient ones                                               |
| 📋 Model Evaluation                             | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oumi-ai/oumi/blob/main/notebooks/Oumi%20-%20Evaluation%20with%20Oumi.ipynb) | Comprehensive model evaluation using Oumi's evaluation framework                                         |
| ☁️ Remote Training                             | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oumi-ai/oumi/blob/main/notebooks/Oumi%20-%20Running%20Jobs%20Remotely.ipynb) | Launch and monitor training jobs on cloud (AWS, Azure, GCP, Lambda, etc.) platforms                    |
| 📈 LLM-as-a-Judge                               | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oumi-ai/oumi/blob/main/notebooks/Oumi%20-%20Simple%20Judge.ipynb) | Filter and curate training data with built-in judges                                                      |

## Installation

Install Oumi easily with:

```bash
pip install oumi  # CPU & NPU only

pip install oumi[gpu]  # With GPU support

pip install git+https://github.com/oumi-ai/oumi.git  # Latest version
```

For advanced installation and setup, see the [installation guide](https://oumi.ai/docs/en/latest/get_started/installation.html).

## CLI Usage

Use the `oumi` command for quick model training, evaluation, and inference via the [recipes](/configs/recipes):

```bash
# Training
oumi train -c configs/recipes/smollm/sft/135m/quickstart_train.yaml

# Evaluation
oumi evaluate -c configs/recipes/smollm/evaluation/135m/quickstart_eval.yaml

# Inference
oumi infer -c configs/recipes/smollm/inference/135m_infer.yaml --interactive
```

Find more options in the [training](https://oumi.ai/docs/en/latest/user_guides/train/train.html), [evaluation](https://oumi.ai/docs/en/latest/user_guides/evaluate/evaluate.html), [inference](https://oumi.ai/docs/en/latest/user_guides/infer/infer.html), and [llm-as-a-judge](https://oumi.ai/docs/en/latest/user_guides/judge/judge.html) guides.

## Running Jobs Remotely

Deploy jobs remotely on cloud platforms (AWS, Azure, GCP, Lambda, etc.):

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

**Note:** Oumi is in beta and actively developed. The core features are stable, but advanced features may evolve.

## Why Choose Oumi?

Oumi provides a robust platform for model training, evaluation, and deployment. It's a great choice because it offers:

*   **Zero Boilerplate:** Get started quickly with ready-to-use recipes.
*   **Enterprise-Grade:** Built for training models at scale.
*   **Research Ready:** Perfect for ML research with reproducible experiments.
*   **Broad Model Support:** Compatibility with various architectures, from small to large models, text-only to multimodal.
*   **SOTA Performance:** Support for distributed training techniques (FSDP, DDP) and optimized inference engines (vLLM, SGLang).
*   **Community Driven:** Fully open source, fostering community contributions and eliminating vendor lock-in.

## Recipes & Examples

Explore a growing selection of ready-to-use configurations. For a complete list of supported models and datasets, see the oumi documentation.

### Qwen Family

```
| Model          | Example Configurations                                                                                      |
| -------------- | ----------------------------------------------------------------------------------------------------------- |
| Qwen3 30B A3B  | [LoRA](/configs/recipes/qwen3/sft/30b_a3b_lora/train.yaml) • [Inference](/configs/recipes/qwen3/inference/30b_a3b_infer.yaml) • [Evaluation](/configs/recipes/qwen3/evaluation/30b_a3b_eval.yaml) |
| Qwen3 32B  | [LoRA](/configs/recipes/qwen3/sft/32b_lora/train.yaml) • [Inference](/configs/recipes/qwen3/inference/32b_infer.yaml) • [Evaluation](/configs/recipes/qwen3/evaluation/32b_eval.yaml) |
| QwQ 32B      | [FFT](/configs/recipes/qwq/sft/full_train.yaml) • [LoRA](/configs/recipes/qwq/sft/lora_train.yaml) • [QLoRA](/configs/recipes/qwq/sft/qlora_train.yaml) • [Inference](/configs/recipes/qwq/inference/infer.yaml) • [Evaluation](/configs/recipes/qwq/evaluation/eval.yaml) |
| Qwen2.5-VL 3B | [SFT](/configs/recipes/vision/qwen2_5_vl_3b/sft/full/train.yaml) • [LoRA](/configs/recipes/vision/qwen2_5_vl_3b/sft/lora/train.yaml)• [Inference (vLLM)](configs/recipes/vision/qwen2_5_vl_3b/inference/vllm_infer.yaml) • [Inference](configs/recipes/vision/qwen2_5_vl_3b/inference/infer.yaml) |
| Qwen2-VL 2B   | [SFT](/configs/recipes/vision/qwen2_vl_2b/sft/full/train.yaml) • [LoRA](/configs/recipes/vision/qwen2_vl_2b/sft/lora/train.yaml) • [Inference (vLLM)](configs/recipes/vision/qwen2_vl_2b/inference/vllm_infer.yaml) • [Inference (SGLang)](configs/recipes/vision/qwen2_vl_2b/inference/sglang_infer.yaml) • [Inference](configs/recipes/vision/qwen2_vl_2b/inference/infer.yaml) • [Evaluation](configs/recipes/vision/qwen2_vl_2b/evaluation/eval.yaml) |
```

### DeepSeek R1 Family

```
| Model              | Example Configurations                                                                                             |
| ------------------ | ------------------------------------------------------------------------------------------------------------------ |
| DeepSeek R1 671B    | [Inference (Together AI)](configs/recipes/deepseek_r1/inference/671b_together_infer.yaml)                         |
| Distilled Llama 8B | [FFT](/configs/recipes/deepseek_r1/sft/distill_llama_8b/full_train.yaml) • [LoRA](/configs/recipes/deepseek_r1/sft/distill_llama_8b/lora_train.yaml) • [QLoRA](/configs/recipes/deepseek_r1/sft/distill_llama_8b/qlora_train.yaml) • [Inference](configs/recipes/deepseek_r1/inference/distill_llama_8b_infer.yaml) • [Evaluation](/configs/recipes/deepseek_r1/evaluation/distill_llama_8b/eval.yaml) |
| Distilled Llama 70B | [FFT](/configs/recipes/deepseek_r1/sft/distill_llama_70b/full_train.yaml) • [LoRA](/configs/recipes/deepseek_r1/sft/distill_llama_70b/lora_train.yaml) • [QLoRA](/configs/recipes/deepseek_r1/sft/distill_llama_70b/qlora_train.yaml) • [Inference](configs/recipes/deepseek_r1/inference/distill_llama_70b_infer.yaml) • [Evaluation](/configs/recipes/deepseek_r1/evaluation/distill_llama_70b/eval.yaml) |
| Distilled Qwen 1.5B | [FFT](/configs/recipes/deepseek_r1/sft/distill_qwen_1_5b/full_train.yaml) • [LoRA](/configs/recipes/deepseek_r1/sft/distill_qwen_1_5b/lora_train.yaml) • [Inference](configs/recipes/deepseek_r1/inference/distill_qwen_1_5b_infer.yaml) • [Evaluation](/configs/recipes/deepseek_r1/evaluation/distill_qwen_1_5b/eval.yaml) |
| Distilled Qwen 32B | [LoRA](/configs/recipes/deepseek_r1/sft/distill_qwen_32b/lora_train.yaml) • [Inference](configs/recipes/deepseek_r1/inference/distill_qwen_32b_infer.yaml) • [Evaluation](/configs/recipes/deepseek_r1/evaluation/distill_qwen_32b/eval.yaml) |
```

### Llama Family

```
| Model                           | Example Configurations                                                                                                                                                                                               |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Llama 4 Scout Instruct 17B   | [FFT](/configs/recipes/llama4/sft/scout_instruct_full/train.yaml) • [LoRA](/configs/recipes/llama4/sft/scout_instruct_lora/train.yaml) • [QLoRA](/configs/recipes/llama4/sft/scout_instruct_qlora/train.yaml) • [Inference (vLLM)](/configs/recipes/llama4/inference/scout_instruct_vllm_infer.yaml) • [Inference](/configs/recipes/llama4/inference/scout_instruct_infer.yaml) • [Inference (Together.ai)](/configs/recipes/llama4/inference/scout_instruct_together_infer.yaml) |
| Llama 4 Scout 17B             | [FFT](/configs/recipes/llama4/sft/scout_base_full/train.yaml)                                                                                                                                                                |
| Llama 3.1 8B                | [FFT](/configs/recipes/llama3_1/sft/8b_full/train.yaml) • [LoRA](/configs/recipes/llama3_1/sft/8b_lora/train.yaml) • [QLoRA](/configs/recipes/llama3_1/sft/8b_qlora/train.yaml) • [Pre-training](/configs/recipes/llama3_1/pretraining/8b/train.yaml) • [Inference (vLLM)](configs/recipes/llama3_1/inference/8b_rvllm_infer.yaml) • [Inference](/configs/recipes/llama3_1/inference/8b_infer.yaml) • [Evaluation](/configs/recipes/llama3_1/evaluation/8b_eval.yaml) |
| Llama 3.1 70B               | [FFT](/configs/recipes/llama3_1/sft/70b_full/train.yaml) • [LoRA](/configs/recipes/llama3_1/sft/70b_lora/train.yaml) • [QLoRA](/configs/recipes/llama3_1/sft/70b_qlora/train.yaml) • [Inference](/configs/recipes/llama3_1/inference/70b_infer.yaml) • [Evaluation](/configs/recipes/llama3_1/evaluation/70b_eval.yaml) |
| Llama 3.1 405B              | [FFT](/configs/recipes/llama3_1/sft/405b_full/train.yaml) • [LoRA](/configs/recipes/llama3_1/sft/405b_lora/train.yaml) • [QLoRA](/configs/recipes/llama3_1/sft/405b_qlora/train.yaml)                                                |
| Llama 3.2 1B                | [FFT](/configs/recipes/llama3_2/sft/1b_full/train.yaml) • [LoRA](/configs/recipes/llama3_2/sft/1b_lora/train.yaml) • [QLoRA](/configs/recipes/llama3_2/sft/1b_qlora/train.yaml) • [Inference (vLLM)](/configs/recipes/llama3_2/inference/1b_vllm_infer.yaml) • [Inference (SGLang)](/configs/recipes/llama3_2/inference/1b_sglang_infer.yaml) • [Inference](/configs/recipes/llama3_2/inference/1b_infer.yaml) • [Evaluation](/configs/recipes/llama3_2/evaluation/1b_eval.yaml) |
| Llama 3.2 3B                | [FFT](/configs/recipes/llama3_2/sft/3b_full/train.yaml) • [LoRA](/configs/recipes/llama3_2/sft/3b_lora/train.yaml) • [QLoRA](/configs/recipes/llama3_2/sft/3b_qlora/train.yaml) • [Inference (vLLM)](/configs/recipes/llama3_2/inference/3b_vllm_infer.yaml) • [Inference (SGLang)](/configs/recipes/llama3_2/inference/3b_sglang_infer.yaml) • [Inference](/configs/recipes/llama3_2/inference/3b_infer.yaml) • [Evaluation](/configs/recipes/llama3_2/evaluation/3b_eval.yaml) |
| Llama 3.3 70B               | [FFT](/configs/recipes/llama3_3/sft/70b_full/train.yaml) • [LoRA](/configs/recipes/llama3_3/sft/70b_lora/train.yaml) • [QLoRA](/configs/recipes/llama3_3/sft/70b_qlora/train.yaml) • [Inference (vLLM)](/configs/recipes/llama3_3/inference/70b_vllm_infer.yaml) • [Inference](/configs/recipes/llama3_3/inference/70b_infer.yaml) • [Evaluation](/configs/recipes/llama3_3/evaluation/70b_eval.yaml) |
| Llama 3.2 Vision 11B        | [SFT](/configs/recipes/vision/llama3_2_vision/sft/11b_full/train.yaml) • [Inference (vLLM)](/configs/recipes/vision/llama3_2_vision/inference/11b_rvllm_infer.yaml) • [Inference (SGLang)](/configs/recipes/vision/llama3_2_vision/inference/11b_sglang_infer.yaml) • [Evaluation](/configs/recipes/vision/llama3_2_vision/evaluation/11b_eval.yaml) |
```

### Falcon Family

```
| Model                                            | Example Configurations                                                 |
| ------------------------------------------------ | -------------------------------------------------------------------- |
| [Falcon-H1](https://huggingface.co/collections/tiiuae/falcon-h1-6819f2795bc406da60fab8df) | [FFT](/configs/recipes/falcon_h1/sft/) • [Inference](/configs/recipes/falcon_h1/inference/) • [Evaluation](/configs/recipes/falcon_h1/evaluation/) |
| [Falcon-E (BitNet)](https://huggingface.co/collections/tiiuae/falcon-edge-series-6804fd13344d6d8a8fa71130) | [FFT](/configs/recipes/falcon_e/sft/) • [DPO](/configs/recipes/falcon_e/dpo/) • [Evaluation](/configs/recipes/falcon_e/evaluation/) |
```

### Vision Models

```
| Model                      | Example Configurations                                                                                                                                                                                                                                  |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Llama 3.2 Vision 11B       | [SFT](/configs/recipes/vision/llama3_2_vision/sft/11b_full/train.yaml) • [LoRA](/configs/recipes/vision/llama3_2_vision/sft/11b_lora/train.yaml) • [Inference (vLLM)](/configs/recipes/vision/llama3_2_vision/inference/11b_rvllm_infer.yaml) • [Inference (SGLang)](/configs/recipes/vision/llama3_2_vision/inference/11b_sglang_infer.yaml) • [Evaluation](/configs/recipes/vision/llama3_2_vision/evaluation/11b_eval.yaml) |
| LLaVA 7B                   | [SFT](/configs/recipes/vision/llava_7b/sft/train.yaml) • [Inference (vLLM)](configs/recipes/vision/llava_7b/inference/vllm_infer.yaml) • [Inference](/configs/recipes/vision/llava_7b/inference/infer.yaml)                                           |
| Phi3 Vision 4.2B           | [SFT](/configs/recipes/vision/phi3/sft/full/train.yaml) • [LoRA](/configs/recipes/vision/phi3/sft/lora/train.yaml) • [Inference (vLLM)](configs/recipes/vision/phi3/inference/vllm_infer.yaml)                                                            |
| Phi4 Vision 5.6B           | [SFT](/configs/recipes/vision/phi4/sft/full/train.yaml) • [LoRA](/configs/recipes/vision/phi4/sft/lora/train.yaml) • [Inference (vLLM)](configs/recipes/vision/phi4/inference/vllm_infer.yaml) • [Inference](/configs/recipes/vision/phi4/inference/infer.yaml) |
| Qwen2-VL 2B                | [SFT](/configs/recipes/vision/qwen2_vl_2b/sft/full/train.yaml) • [LoRA](/configs/recipes/vision/qwen2_vl_2b/sft/lora/train.yaml) • [Inference (vLLM)](configs/recipes/vision/qwen2_vl_2b/inference/vllm_infer.yaml) • [Inference (SGLang)](configs/recipes/vision/qwen2_vl_2b/inference/sglang_infer.yaml) • [Inference](configs/recipes/vision/qwen2_vl_2b/inference/infer.yaml) • [Evaluation](configs/recipes/vision/qwen2_vl_2b/evaluation/eval.yaml) |
| Qwen2.5-VL 3B              | [SFT](/configs/recipes/vision/qwen2_5_vl_3b/sft/full/train.yaml) • [LoRA](/configs/recipes/vision/qwen2_5_vl_3b/sft/lora/train.yaml)• [Inference (vLLM)](configs/recipes/vision/qwen2_5_vl_3b/inference/vllm_infer.yaml) • [Inference](configs/recipes/vision/qwen2_5_vl_3b/inference/infer.yaml) |
| SmolVLM-Instruct 2B        | [SFT](/configs/recipes/vision/smolvlm/sft/full/train.yaml) • [LoRA](/configs/recipes/vision/smolvlm/sft/lora/train.yaml)                                                                                                                                         |
```

## Documentation

Explore the full capabilities of Oumi in the [Oumi Documentation](https://oumi.ai/docs).

## Community

Join our active community!

*   Contribute: Learn how to contribute to the `oumi` repository by checking the [`CONTRIBUTING.md`](https://github.com/oumi-ai/oumi/blob/main/CONTRIBUTING.md) file for guidance.
*   Discord: Get help, share experiences, and contribute to the project through our [Discord community](https://discord.gg/oumi).
*   Open Collaboration: Check out our [open collaboration](https://oumi.ai/community) page to join community open-science efforts.

## Acknowledgements

Oumi leverages numerous open-source libraries. We appreciate the contributions of these projects.

## Citation

If you use Oumi in your research, please cite it:

```bibtex
@software{oumi2025,
  author = {Oumi Community},
  title = {Oumi: an Open, End-to-end Platform for Building Large Foundation Models},
  month = {January},
  year = {2025},
  url = {https://github.com/oumi-ai/oumi}
}
```

## License

Licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file.