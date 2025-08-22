<div align="center">
  <img src="assets/logo.png" alt="LLaMA Factory Logo" width="300">
</div>

# LLaMA Factory: Your All-in-One Solution for Efficient LLM Fine-Tuning

**Fine-tune any large language model with ease using LLaMA Factory, from Llama to Mistral, Qwen, and beyond!** [Explore the repo](https://github.com/hiyouga/LLaMA-Factory)

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

### Used by [Amazon](https://aws.amazon.com/cn/blogs/machine-learning/how-apoidea-group-enhances-visual-information-extraction-from-banking-documents-with-multimodal-models-using-llama-factory-on-amazon-sagemaker-hyperpod/), [NVIDIA](https://developer.nvidia.com/rtx/ai-toolkit), [Aliyun](https://help.aliyun.com/zh/pai/use-cases/fine-tune-a-llama-3-model-with-llama-factory), and many others!

<div align="center" markdown="1">

### Supporters ❤️

| <div style="text-align: center;"><a href="https://warp.dev/llama-factory"><img alt="Warp sponsorship" width="400" src="assets/warp.jpg"></a><br><a href="https://warp.dev/llama-factory" style="font-size:larger;">Warp, the agentic terminal for developers</a><br><a href="https://warp.dev/llama-factory">Available for MacOS, Linux, & Windows</a> | <a href="https://serpapi.com"><img alt="SerpAPI sponsorship" width="250" src="assets/serpapi.svg"> </a> |
| ---- | ---- |

----

### Fine-tune 100+ LLMs effortlessly with our zero-code [CLI](#quickstart) and [Web UI](#fine-tuning-with-llama-board-gui-powered-by-gradio)!

![GitHub Trend](https://trendshift.io/api/badge/repositories/4535)

</div>

👋 Join our [WeChat group](assets/wechat.jpg), [NPU user group](assets/wechat_npu.jpg) or [Alaya NeW user group](assets/wechat_alaya.png).

\[ English | [中文](README_zh.md) \]

**Easily fine-tuning a large language model is now a reality...**

[Explore this exciting functionality](https://github.com/hiyouga/LLaMA-Factory)

https://github.com/user-attachments/assets/3991a3a8-4276-4d30-9cab-4cb0c4b9b99e

Choose your path:

- **Documentation (WIP)**: https://llamafactory.readthedocs.io/en/latest/
- **Documentation (AMD GPU)**: https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/fine_tune/llama_factory_llama3.html
- **Colab (free)**: https://colab.research.google.com/drive/1eRTPn37ltBbYsISy9Aw2NuI2Aq5CQrD9?usp=sharing
- **Local machine**: Please refer to [usage](#getting-started)
- **PAI-DSW (free trial)**: https://gallery.pai-ml.com/#/preview/deepLearning/nlp/llama_factory
- **Alaya NeW (cloud GPU deal)**: https://docs.alayanew.com/docs/documents/useGuide/LLaMAFactory/mutiple/?utm_source=LLaMA-Factory

> [!NOTE]
> Except for the above links, all other websites are unauthorized third-party websites. Please carefully use them.

## Key Features

*   **Extensive Model Support:** LLaMA, LLaVA, Mistral, Mixtral-MoE, Qwen, Qwen2-VL, DeepSeek, Yi, Gemma, ChatGLM, Phi, and more.
*   **Versatile Training Methods:** (Continuous) Pre-training, (multimodal) Supervised Fine-tuning, Reward Modeling, PPO, DPO, KTO, ORPO.
*   **Flexible Scaling:** Full-tuning, Freeze-tuning, LoRA, and efficient QLoRA via AQLM/AWQ/GPTQ/LLM.int8/HQQ/EETQ.
*   **Advanced Optimization:** GaLore, BAdam, APOLLO, Adam-mini, Muon, OFT, DoRA, LongLoRA, LLaMA Pro, Mixture-of-Depths, LoRA+, LoftQ and PiSSA.
*   **Performance Enhancements:** FlashAttention-2, Unsloth, Liger Kernel, RoPE scaling, NEFTune and rsLoRA.
*   **Broad Task Applicability:** Multi-turn dialogue, tool use, image understanding, visual grounding, video recognition, audio understanding, and beyond.
*   **Comprehensive Monitoring:** LlamaBoard, TensorBoard, Wandb, MLflow, SwanLab, and more for effective experiment tracking.
*   **Accelerated Inference:** OpenAI-style API, Gradio UI, and CLI integration with [vLLM](https://github.com/vllm-project/vllm) or [SGLang](https://github.com/sgl-project/sglang).
---

### Day-N Support for Fine-Tuning Cutting-Edge Models

| Support Date | Model Name                                                           |
| ------------ | -------------------------------------------------------------------- |
| Day 0        | Qwen3 / Qwen2.5-VL / Gemma 3 / GLM-4.1V / InternLM 3 / MiniCPM-o-2.6 |
| Day 1        | Llama 3 / GLM-4 / Mistral Small / PaliGemma2 / Llama 4               |

## Table of Contents

*   [Key Features](#key-features)
*   [Blogs](#blogs)
*   [Changelog](#changelog)
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

## Blogs

*   [Fine-tune GPT-OSS for Role-Playing using LLaMA-Factory](https://docs.llamafactory.com.cn/docs/documents/best-practice/gptroleplay/?utm_source=LLaMA-Factory) (Chinese)
*   [Fine-tune Llama3.1-70B for Medical Diagnosis using LLaMA-Factory](https://docs.alayanew.com/docs/documents/bestPractice/bigModel/llama70B/?utm_source=LLaMA-Factory) (Chinese)
*   [A One-Stop Code-Free Model Reinforcement Learning and Deployment Platform based on LLaMA-Factory and EasyR1](https://aws.amazon.com/cn/blogs/china/building-llm-model-hub-based-on-llamafactory-and-easyr1/) (Chinese)
*   [How Apoidea Group enhances visual information extraction from banking documents with multimodal models using LLaMA-Factory on Amazon SageMaker HyperPod](https://aws.amazon.com/cn/blogs/machine-learning/how-apoidea-group-enhances-visual-information-extraction-from-banking-documents-with-multimodal-models-using-llama-factory-on-amazon-sagemaker-hyperpod/) (English)
*   [Easy Dataset × LLaMA Factory: Enabling LLMs to Efficiently Learn Domain Knowledge](https://buaa-act.feishu.cn/wiki/GVzlwYcRFiR8OLkHbL6cQpYin7g) (English)

<details><summary>All Blogs</summary>

- [Fine-tune Qwen2.5-VL for Autonomous Driving using LLaMA-Factory](https://docs.alayanew.com/docs/documents/useGuide/LLaMAFactory/mutiple/?utm_source=LLaMA-Factory) (Chinese)
- [LLaMA Factory: Fine-tuning the DeepSeek-R1-Distill-Qwen-7B Model for News Classifier](https://gallery.pai-ml.com/#/preview/deepLearning/nlp/llama_factory_deepseek_r1_distill_7b) (Chinese)
- [A One-Stop Code-Free Model Fine-Tuning \& Deployment Platform based on SageMaker and LLaMA-Factory](https://aws.amazon.com/cn/blogs/china/a-one-stop-code-free-model-fine-tuning-deployment-platform-based-on-sagemaker-and-llama-factory/) (Chinese)
- [LLaMA Factory Multi-Modal Fine-Tuning Practice: Fine-Tuning Qwen2-VL for Personal Tourist Guide](https://gallery.pai-ml.com/#/preview/deepLearning/nlp/llama_factory_qwen2vl) (Chinese)
- [LLaMA Factory: Fine-tuning Llama3 for Role-Playing](https://gallery.pai-ml.com/#/preview/deepLearning/nlp/llama_factory) (Chinese)

</details>

## Changelog

[25/08/20] We supported fine-tuning the **[Intern-S1-mini](https://huggingface.co/internlm/Intern-S1-mini)** models. See [PR #8976](https://github.com/hiyouga/LLaMA-Factory/pull/8976) to get started.

[25/08/06] We supported fine-tuning the **[GPT-OSS](https://github.com/openai/gpt-oss)** models. See [PR #8826](https://github.com/hiyouga/LLaMA-Factory/pull/8826) to get started.

[25/07/02] We supported fine-tuning the **[GLM-4.1V-9B-Thinking](https://github.com/THUDM/GLM-4.1V-Thinking)** model.

[25/04/28] We supported fine-tuning the **[Qwen3](https://qwenlm.github.io/blog/qwen3/)** model family.

<details><summary>Full Changelog</summary>

[25/04/21] We supported the **[Muon](https://github.com/KellerJordan/Muon)** optimizer. See [examples](examples/README.md) for usage. Thank [@tianshijing](https://github.com/tianshijing)'s PR.

[25/04/16] We supported fine-tuning the **[InternVL3](https://huggingface.co/OpenGVLab/InternVL3-8B)** model. See [PR #7258](https://github.com/hiyouga/LLaMA-Factory/pull/7258) to get started.

[25/04/14] We supported fine-tuning the **[GLM-Z1](https://huggingface.co/THUDM/GLM-Z1-9B-0414)** and **[Kimi-VL](https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct)** models.

[25/04/06] We supported fine-tuning the **[Llama 4](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)** model. See [PR #7611](https://github.com/hiyouga/LLaMA-Factory/pull/7611) to get started.

[25/03/31] We supported fine-tuning the **[Qwen2.5 Omni](https://qwenlm.github.io/blog/qwen2.5-omni/)** model. See [PR #7537](https://github.com/hiyouga/LLaMA-Factory/pull/7537) to get started.

[25/03/15] We supported **[SGLang](https://github.com/sgl-project/sglang)** as inference backend. Try `infer_backend: sglang` to accelerate inference.

[25/03/12] We supported fine-tuning the **[Gemma 3](https://huggingface.co/blog/gemma3)** model.

[25/02/24] Announcing **[EasyR1](https://github.com/hiyouga/EasyR1)**, an efficient, scalable and multi-modality RL training framework for efficient GRPO training.

[25/02/11] We supported saving the **[Ollama](https://github.com/ollama/ollama)** modelfile when exporting the model checkpoints. See [examples](examples/README.md) for usage.

[25/02/05] We supported fine-tuning the **[Qwen2-Audio](Qwen/Qwen2-Audio-7B-Instruct)** and **[MiniCPM-o-2.6](https://huggingface.co/openbmb/MiniCPM-o-2_6)** on audio understanding tasks.

[25/01/31] We supported fine-tuning the **[DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1)** and **[Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)** models.

[25/01/15] We supported **[APOLLO](https://arxiv.org/abs/2412.05270)** optimizer. See [examples](examples/README.md) for usage.

[25/01/14] We supported fine-tuning the **[MiniCPM-o-2.6](https://huggingface.co/openbmb/MiniCPM-o-2_6)** and **[MiniCPM-V-2.6](https://huggingface.co/openbmb/MiniCPM-V-2_6)** models. Thank [@BUAADreamer](https://github.com/BUAADreamer)'s PR.

[25/01/14] We supported fine-tuning the **[InternLM 3](https://huggingface.co/collections/internlm/)** models. Thank [@hhaAndroid](https://github.com/hhaAndroid)'s PR.

[25/01/10] We supported fine-tuning the **[Phi-4](https://huggingface.co/microsoft/phi-4)** model.

[24/12/21] We supported using **[SwanLab](https://github.com/SwanHubX/SwanLab)** for experiment tracking and visualization. See [this section](#use-swanlab-logger) for details.

[24/11/27] We supported fine-tuning the **[Skywork-o1](https://huggingface.co/Skywork/Skywork-o1-Open-Llama-3.1-8B)** model and the **[OpenO1](https://huggingface.co/datasets/O1-OPEN/OpenO1-SFT)** dataset.

[24/10/09] We supported downloading pre-trained models and datasets from the **[Modelers Hub](https://modelers.cn/models)**. See [this tutorial](#download-from-modelers-hub) for usage.

[24/09/19] We supported fine-tuning the **[Qwen2.5](https://qwenlm.github.io/blog/qwen2.5/)** models.

[24/08/30] We supported fine-tuning the **[Qwen2-VL](https://qwenlm.github.io/blog/qwen2-vl/)** models. Thank [@simonJJJ](https://github.com/simonJJJ)'s PR.

[24/08/27] We supported **[Liger Kernel](https://github.com/linkedin/Liger-Kernel)**. Try `enable_liger_kernel: true` for efficient training.

[24/08/09] We supported **[Adam-mini](https://github.com/zyushun/Adam-mini)** optimizer. See [examples](examples/README.md) for usage. Thank [@relic-yuexi](https://github.com/relic-yuexi)'s PR.

[24/07/04] We supported [contamination-free packed training](https://github.com/MeetKai/functionary/tree/main/functionary/train/packing). Use `neat_packing: true` to activate it. Thank [@chuan298](https://github.com/chuan298)'s PR.

[24/06/16] We supported **[PiSSA](https://arxiv.org/abs/2404.02948)** algorithm. See [examples](examples/README.md) for usage.

[24/06/07] We supported fine-tuning the **[Qwen2](https://qwenlm.github.io/blog/qwen2/)** and **[GLM-4](https://github.com/THUDM/GLM-4)** models.

[24/05/26] We supported **[SimPO](https://arxiv.org/abs/2405.14734)** algorithm for preference learning. See [examples](examples/README.md) for usage.

[24/05/20] We supported fine-tuning the **PaliGemma** series models. Note that the PaliGemma models are pre-trained models, you need to fine-tune them with `paligemma` template for chat completion.

[24/05/18] We supported **[KTO](https://arxiv.org/abs/2402.01306)** algorithm for preference learning. See [examples](examples/README.md) for usage.

[24/05/14] We supported training and inference on the Ascend NPU devices. Check [installation](#installation) section for details.

[24/04/26] We supported fine-tuning the **LLaVA-1.5** multimodal LLMs. See [examples](examples/README.md) for usage.

[24/04/22] We provided a **[Colab notebook](https://colab.research.google.com/drive/1eRTPn37ltBbYsISy9Aw2NuI2Aq5CQrD9?usp=sharing)** for fine-tuning the Llama-3 model on a free T4 GPU. Two Llama-3-derived models fine-tuned using LLaMA Factory are available at Hugging Face, check [Llama3-8B-Chinese-Chat](https://huggingface.co/shenzhi-wang/Llama3-8B-Chinese-Chat) and [Llama3-Chinese](https://huggingface.co/zhichen/Llama3-Chinese) for details.

[24/04/21] We supported **[Mixture-of-Depths](https://arxiv.org/abs/2404.02258)** according to [AstraMindAI's implementation](https://github.com/astramind-ai/Mixture-of-depths). See [examples](examples/README.md) for usage.

[24/04/16] We supported **[BAdam](https://arxiv.org/abs/2404.02827)** optimizer. See [examples](examples/README.md) for usage.

[24/04/16] We supported **[unsloth](https://github.com/unslothai/unsloth)**'s long-sequence training (Llama-2-7B-56k within 24GB). It achieves **117%** speed and **50%** memory compared with FlashAttention-2, more benchmarks can be found in [this page](https://github.com/hiyouga/LLaMA-Factory/wiki/Performance-comparison).

[24/03/31] We supported **[ORPO](https://arxiv.org/abs/2403.07691)**. See [examples](examples/README.md) for usage.

[24/03/21] Our paper "[LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models](https://arxiv.org/abs/2403.13372)" is available at arXiv!

[24/03/20] We supported **FSDP+QLoRA** that fine-tunes a 70B model on 2x24GB GPUs. See [examples](examples/README.md) for usage.

[24/03/13] We supported **[LoRA+](https://arxiv.org/abs/2402.12354)**. See [examples](examples/README.md) for usage.

[24/03/07] We supported **[GaLore](https://arxiv.org/abs/2403.03507)** optimizer. See [examples](examples/README.md) for usage.

[24/03/07] We integrated **[vLLM](https://github.com/vllm-project/vllm)** for faster and concurrent inference. Try `infer_backend: vllm` to enjoy **270%** inference speed.

[24/02/28] We supported weight-decomposed LoRA (**[DoRA](https://arxiv.org/abs/2402.09353)**). Try `use_dora: true` to activate DoRA training.

[24/02/15] We supported **block expansion** proposed by [LLaMA Pro](https://github.com/TencentARC/LLaMA-Pro). See [examples](examples/README.md) for usage.

[24/02/05] Qwen1.5 (Qwen2 beta version) series models are supported in LLaMA-Factory. Check this [blog post](https://qwenlm.github.io/blog/qwen1.5/) for details.

[24/01/18] We supported **agent tuning** for most models, equipping model with tool using abilities by fine-tuning with `dataset: glaive_toolcall_en`.

[23/12/23] We supported **[unsloth](https://github.com/unslothai/unsloth)**'s implementation to boost LoRA tuning for the LLaMA, Mistral and Yi models. Try `use_unsloth: true` argument to activate unsloth patch. It achieves **170%** speed in our benchmark, check [this page](https://github.com/hiyouga/LLaMA-Factory/wiki/Performance-comparison) for details.

[23/12/12] We supported fine-tuning the latest MoE model **[Mixtral 8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)** in our framework. See hardware requirement [here](#hardware-requirement).

[23/12/01] We supported downloading pre-trained models and datasets from the **[ModelScope Hub](https://modelscope.cn/models)**. See [this tutorial](#download-from-modelscope-hub) for usage.

[23/10/21] We supported **[NEFTune](https://arxiv.org/abs/2310.05914)** trick for fine-tuning. Try `neftune_noise_alpha: 5` argument to activate NEFTune.

[23/09/27] We supported **$S^2$-Attn** proposed by [LongLoRA](https://github.com/dvlab-research/LongLoRA) for the LLaMA models. Try `shift_attn: true` argument to enable shift short attention.

[23/09/23] We integrated MMLU, C-Eval and CMMLU benchmarks in this repo. See [examples](examples/README.md) for usage.

[23/09/10] We supported **[FlashAttention-2](https://github.com/Dao-AILab/flash-attention)**. Try `flash_attn: fa2` argument to enable FlashAttention-2 if you are using RTX4090, A100 or H100 GPUs.

[23/08/12] We supported **RoPE scaling** to extend the context length of the LLaMA models. Try `rope_scaling: linear` argument in training and `rope_scaling: dynamic` argument at inference to extrapolate the position embeddings.

[23/08/11] We supported **[DPO training](https://arxiv.org/abs/2305.18290)** for instruction-tuned models. See [examples](examples/README.md) for usage.

[23/07/31] We supported **dataset streaming**. Try `streaming: true` and `max_steps: 10000` arguments to load your dataset in streaming mode.

[23/07/29] We released two instruction-tuned 13B models at Hugging Face. See these Hugging Face Repos ([LLaMA-2](https://huggingface.co/hiyouga/Llama-2-Chinese-13b-chat) / [Baichuan](https://huggingface.co/hiyouga/Baichuan-13B-sft)) for details.

[23/07/18] We developed an **all-in-one Web UI** for training, evaluation and inference. Try `train_web.py` to fine-tune models in your Web browser. Thank [@KanadeSiina](https://github.com/KanadeSiina) and [@codemayq](https://github.com/codemayq) for their efforts in the development.

[23/07/09] We released **[FastEdit](https://github.com/hiyouga/FastEdit)** ⚡🩹, an easy-to-use package for editing the factual knowledge of large language models efficiently. Please follow [FastEdit](https://github.com/hiyouga/FastEdit) if you are interested.

[23/06/29] We provided a **reproducible example** of training a chat model using instruction-following datasets, see [Baichuan-7B-sft](https://huggingface.co/hiyouga/Baichuan-7B-sft) for details.

[23/06/22] We aligned the [demo API](src/api_demo.py) with the [OpenAI's](https://platform.openai.com/docs/api-reference/chat) format where you can insert the fine-tuned model in **arbitrary ChatGPT-based applications**.

[23/06/03] We supported quantized training and inference (aka **[QLoRA](https://github.com/artidoro/qlora)**). See [examples](examples/README.md) for usage.

</details>

> [!TIP]
> If you cannot use the latest feature, please pull the latest code and install LLaMA-Factory again.

## Supported Models

| Model                                                             | Model size                       | Template            |
| ----------------------------------------------------------------- | -------------------------------- | ------------------- |
| [Baichuan 2](https://huggingface.co/baichuan-inc)                 | 7B/13B                           | baichuan2           |
| [BLOOM/BLOOMZ](https://huggingface.co/bigscience)                 | 560M/1.1B/1.7B/3B/7.1B/176B      | -                   |
| [ChatGLM3](https://huggingface.co/THUDM)                          | 6B                               | chatglm3            |
| [Command R](https://huggingface.co/CohereForAI)                   | 35B/104B                         | cohere              |
| [DeepSeek (Code/MoE)](https://huggingface.co/deepseek-ai)         | 7B/16B/67B/236B                  | deepseek            |
| [DeepSeek 2.5/3](https://huggingface.co/deepseek-ai)              | 236B/671B                        | deepseek3           |
| [DeepSeek R1 (Distill)](https://huggingface.co/deepseek-ai)       | 1.5B/7B/8B/14B/32B/70B/671B      | deepseekr1          |
| [Falcon](https://huggingface.co/tiiuae)                           | 7B/11B/40B/180B                  | falcon              |
| [Falcon-H1](https://huggingface.co/tiiuae)                        | 0.5B/1.5B/3B/7B/34B              | falcon_h1           |
| [Gemma/Gemma 2/CodeGemma](https://huggingface.co/google)          | 2B/7B/9B/27B                     | gemma/gemma2        |
| [Gemma 3/Gemma 3n](https://huggingface.co/google)                 | 270M/1B/4B/6B/8B/12B/27B         | gemma3/gemma3n      |
| [GLM-4/GLM-4-0414/GLM-Z1](https://huggingface.co/zai-org)         | 9B/32B                           | glm4/glmz1          |
| [GLM-4.1V](https://huggingface.co/zai-org)                        | 9B                               | glm4v               |
| [GLM-4.5/GLM-4.5V](https://huggingface.co/zai-org)*               | 106B/355B                        | glm4_moe/glm4v_moe  |
| [GPT-2](https://huggingface.co/openai-community)                  | 0.1B/0.4B/0.8B/1.5B              | -                   |
| [GPT-OSS](https://huggingface.co/openai)                          | 20B/120B                         | gpt                 |
| [Granite 3.0-3.3](https://huggingface.co/ibm-granite)             | 1B