<div align="center">
  <img src="assets/logo.png" alt="LLaMA Factory Logo" width="200">
  <h1>LLaMA Factory: Fine-tune Any LLM, Effortlessly</h1>
  <p>
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
      <img src="https://github.com/hiyouga/LLaMA-Factory/actions/workflows/tests.yml/badge.svg" alt="Workflow Status">
    </a>
    <a href="https://pypi.org/project/llamafactory/">
      <img src="https://img.shields.io/pypi/v/llamafactory" alt="PyPI Version">
    </a>
    <a href="https://scholar.google.com/scholar?cites=12620864006390196564">
      <img src="https://img.shields.io/badge/citation-760-green" alt="Citations">
    </a>
    <a href="https://hub.docker.com/r/hiyouga/llamafactory/tags">
      <img src="https://img.shields.io/docker/pulls/hiyouga/llamafactory" alt="Docker Pulls">
    </a>
  </p>
  <p>Fine-tune over 100 large language models, including Llama 3, Qwen, Mistral, and more, with a user-friendly interface and powerful features.</p>

  <p>
    <a href="https://twitter.com/llamafactory_ai">
      <img src="https://img.shields.io/twitter/follow/llamafactory_ai" alt="Follow on Twitter">
    </a>
    <a href="https://discord.gg/rKfvV9r9FK">
      <img src="https://dcbadge.vercel.app/api/server/rKfvV9r9FK?compact=true&style=flat" alt="Join Discord">
    </a>
  </p>

  <p>
    <a href="https://colab.research.google.com/drive/1eRTPn37ltBbYsISy9Aw2NuI2Aq5CQrD9?usp=sharing">
      <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
    </a>
    <a href="https://gallery.pai-ml.com/#/preview/deepLearning/nlp/llama_factory">
      <img src="https://gallery.pai-ml.com/assets/open-in-dsw.svg" alt="Open in DSW">
    </a>
    <a href="https://docs.alayanew.com/docs/documents/newActivities/llamafactory/?utm_source=LLaMA-Factory">
      <img src="assets/alaya_new.svg" alt="Open in Alaya">
    </a>
    <a href="https://huggingface.co/spaces/hiyouga/LLaMA-Board">
      <img src="https://img.shields.io/badge/🤗-Open%20in%20Spaces-blue" alt="Open in Spaces">
    </a>
    <a href="https://modelscope.cn/studios/hiyouga/LLaMA-Board">
      <img src="https://img.shields.io/badge/ModelScope-Open%20in%20Studios-blue" alt="Open in Studios">
    </a>
    <a href="https://novita.ai/templates-library/105981?sharer=88115474-394e-4bda-968e-b88e123d0c47">
      <img src="https://img.shields.io/badge/Novita-Deploy%20Template-blue" alt="Deploy on Novita">
    </a>
  </p>

  <p>Used by <a href="https://aws.amazon.com/cn/blogs/machine-learning/how-apoidea-group-enhances-visual-information-extraction-from-banking-documents-with-multimodal-models-using-llama-factory-on-amazon-sagemaker-hyperpod/">Amazon</a>, <a href="https://developer.nvidia.com/rtx/ai-toolkit">NVIDIA</a>, <a href="https://help.aliyun.com/zh/pai/use-cases/fine-tune-a-llama-3-model-with-llama-factory">Aliyun</a>, and more.</p>

  <div align="center" markdown="1">
    <h3>Supporters ❤️</h3>

    <p>
      <a href="https://warp.dev/llama-factory">
        <img alt="Warp sponsorship" width="400" src="assets/warp.jpg">
        <br>
        <a href="https://warp.dev/llama-factory" style="font-size:larger;">Warp, the agentic terminal for developers</a><br><a href="https://warp.dev/llama-factory">Available for MacOS, Linux, & Windows</a>
      </a>
      <a href="https://serpapi.com">
        <img alt="SerpAPI sponsorship" width="250" src="assets/serpapi.svg">
      </a>
    </p>
  </div>

  <hr>

  <p>
    <a href="#getting-started">Easily fine-tune 100+ large language models with zero-code CLI and Web UI.</a>
  </p>

  <img src="https://trendshift.io/api/badge/repositories/4535" alt="GitHub Trend">

  <p>👋 Join our <a href="assets/wechat.jpg">WeChat group</a>, <a href="assets/wechat_npu.jpg">NPU user group</a> or <a href="assets/wechat_alaya.png">Alaya NeW user group</a>.</p>

  <p>[ <a href="README_zh.md">中文</a> | English ]</p>

  <img src="https://github.com/user-attachments/assets/3991a3a8-4276-4d30-9cab-4cb0c4b9b99e" alt="Fine-tuning Process">

  <!-- Quick Links -->
  <p>Choose your path:</p>
  <ul>
    <li><a href="https://llamafactory.readthedocs.io/en/latest/">Documentation (WIP)</a></li>
    <li><a href="https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/fine_tune/llama_factory_llama3.html">Documentation (AMD GPU)</a></li>
    <li><a href="https://colab.research.google.com/drive/1eRTPn37ltBbYsISy9Aw2NuI2Aq5CQrD9?usp=sharing">Colab (free)</a></li>
    <li>Please refer to <a href="#getting-started">Getting Started</a> for local machine setup.</li>
    <li><a href="https://gallery.pai-ml.com/#/preview/deepLearning/nlp/llama_factory">PAI-DSW (free trial)</a></li>
    <li><a href="https://docs.alayanew.com/docs/documents/useGuide/LLaMAFactory/mutiple/?utm_source=LLaMA-Factory">Alaya NeW (cloud GPU deal)</a></li>
  </ul>

  <p>
    > [!NOTE]
    > Except for the above links, all other websites are unauthorized third-party websites. Please carefully use them.
  </p>
</div>

## Table of Contents

- [Key Features](#key-features)
- [Blogs](#blogs)
- [Changelog](#changelog)
- [Supported Models](#supported-models)
- [Supported Training Approaches](#supported-training-approaches)
- [Provided Datasets](#provided-datasets)
- [Requirements](#requirement)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Data Preparation](#data-preparation)
  - [Quickstart](#quickstart)
  - [Fine-Tuning with LLaMA Board GUI](#fine-tuning-with-llama-board-gui-powered-by-gradio)
  - [Build Docker](#build-docker)
  - [Deploy with OpenAI-style API and vLLM](#deploy-with-openai-style-api-and-vllm)
  - [Download from ModelScope Hub](#download-from-modelscope-hub)
  - [Download from Modelers Hub](#download-from-modelers-hub)
  - [Use W&B Logger](#use-wb-logger)
  - [Use SwanLab Logger](#use-swanlab-logger)
- [Projects using LLaMA Factory](#projects-using-llama-factory)
- [License](#license)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

## Key Features

*   **Model Support**: Comprehensive support for a wide array of models including LLaMA, Qwen, Mistral, and many more.
*   **Training Methods**: Integrated support for Pre-training, Supervised Fine-tuning, Reward Modeling, PPO, DPO, KTO, ORPO and SimPO.
*   **Efficient Tuning**: Utilize full-tuning, freeze-tuning, LoRA, QLoRA, and OFT for scalable training.
*   **Advanced Algorithms**: Incorporates cutting-edge algorithms like GaLore, BAdam, APOLLO, and PiSSA to boost performance.
*   **Practical Enhancements**: Includes FlashAttention-2, Unsloth, and Liger Kernel for speed and efficiency.
*   **Task Versatility**: Supports a range of tasks like multi-turn dialogue, tool usage, image understanding, and more.
*   **Monitoring Tools**: Integrates with LlamaBoard, TensorBoard, Wandb, MLflow, and SwanLab for experiment tracking.
*   **Inference Optimization**: Offers faster inference with OpenAI-style API and Gradio UI using vLLM or SGLang.

## Blogs

*   [Fine-tune GPT-OSS for Role-Playing using LLaMA-Factory](https://docs.llamafactory.com.cn/docs/documents/best-practice/gptroleplay/?utm_source=LLaMA-Factory) (Chinese)
*   [Fine-tune Llama3.1-70B for Medical Diagnosis using LLaMA-Factory](https://docs.alayanew.com/docs/documents/bestPractice/bigModel/llama70B/?utm_source=LLaMA-Factory) (Chinese)
*   [A One-Stop Code-Free Model Reinforcement Learning and Deployment Platform based on LLaMA-Factory and EasyR1](https://aws.amazon.com/cn/blogs/china/building-llm-model-hub-based-on-llamafactory-and-easyr1/) (Chinese)
*   [How Apoidea Group enhances visual information extraction from banking documents with multimodal models using LLaMA-Factory on Amazon SageMaker HyperPod](https://aws.amazon.com/cn/blogs/machine-learning/how-apoidea-group-enhances-visual-information-extraction-from-banking-documents-with-multimodal-models-using-llama-factory-on-amazon-sagemaker-hyperpod/) (English)
*   [Easy Dataset × LLaMA Factory: Enabling LLMs to Efficiently Learn Domain Knowledge](https://buaa-act.feishu.cn/wiki/GVzlwYcRFiR8OLkHbL6cQpYin7g) (English)

<details>
<summary>All Blogs</summary>

*   [Fine-tune Qwen2.5-VL for Autonomous Driving using LLaMA-Factory](https://docs.alayanew.com/docs/documents/useGuide/LLaMAFactory/mutiple/?utm_source=LLaMA-Factory) (Chinese)
*   [LLaMA Factory: Fine-tuning the DeepSeek-R1-Distill-Qwen-7B Model for News Classifier](https://gallery.pai-ml.com/#/preview/deepLearning/nlp/llama_factory_deepseek_r1_distill_7b) (Chinese)
*   [A One-Stop Code-Free Model Fine-Tuning & Deployment Platform based on SageMaker and LLaMA-Factory](https://aws.amazon.com/cn/blogs/china/a-one-stop-code-free-model-fine-tuning-deployment-platform-based-on-sagemaker-and-llama-factory/) (Chinese)
*   [LLaMA Factory Multi-Modal Fine-Tuning Practice: Fine-Tuning Qwen2-VL for Personal Tourist Guide](https://gallery.pai-ml.com/#/preview/deepLearning/nlp/llama_factory_qwen2vl) (Chinese)
*   [LLaMA Factory: Fine-tuning Llama3 for Role-Playing](https://gallery.pai-ml.com/#/preview/deepLearning/nlp/llama_factory) (Chinese)

</details>

## Changelog

*   \[25/08/20] Support fine-tuning the **[Intern-S1-mini](https://huggingface.co/internlm/Intern-S1-mini)** models.
*   \[25/08/06] Support fine-tuning the **[GPT-OSS](https://huggingface.co/openai)** models.
*   \[25/07/02] Support fine-tuning the **[GLM-4.1V-9B-Thinking](https://github.com/THUDM/GLM-4.1V-Thinking)** model.
*   \[25/04/28] Support fine-tuning the **[Qwen3](https://qwenlm.github.io/blog/qwen3/)** model family.

<details>
<summary>Full Changelog</summary>

*   \[25/04/21] Supported the **[Muon](https://github.com/KellerJordan/Muon)** optimizer.
*   \[25/04/16] Supported fine-tuning the **[InternVL3](https://huggingface.co/OpenGVLab/InternVL3-8B)** model.
*   \[25/04/14] Supported fine-tuning the **[GLM-Z1](https://huggingface.co/THUDM/GLM-Z1-9B-0414)** and **[Kimi-VL](https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct)** models.
*   \[25/04/06] Supported fine-tuning the **[Llama 4](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)** model.
*   \[25/03/31] Supported fine-tuning the **[Qwen2.5 Omni](https://qwenlm.github.io/blog/qwen2.5-omni/)** model.
*   \[25/03/15] Supported **[SGLang](https://github.com/sgl-project/sglang)** as inference backend.
*   \[25/03/12] Supported fine-tuning the **[Gemma 3](https://huggingface.co/blog/gemma3)** model.
*   \[25/02/24] Announcing **[EasyR1](https://github.com/hiyouga/EasyR1)**, an efficient, scalable and multi-modality RL training framework for efficient GRPO training.
*   \[25/02/11] Supported saving the **[Ollama](https://github.com/ollama/ollama)** modelfile when exporting the model checkpoints.
*   \[25/02/05] Supported fine-tuning the **[Qwen2-Audio](Qwen/Qwen2-Audio-7B-Instruct)** and **[MiniCPM-o-2.6](https://huggingface.co/openbmb/MiniCPM-o-2_6)** on audio understanding tasks.
*   \[25/01/31] Supported fine-tuning the **[DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1)** and **[Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)** models.
*   \[25/01/15] Supported **[APOLLO](https://arxiv.org/abs/2412.05270)** optimizer.
*   \[25/01/14] Supported fine-tuning the **[MiniCPM-o-2.6](https://huggingface.co/openbmb/MiniCPM-o-2_6)** and **[MiniCPM-V-2.6](https://huggingface.co/openbmb/MiniCPM-V-2_6)** models.
*   \[25/01/14] Supported fine-tuning the **[InternLM 3](https://huggingface.co/collections/internlm/)** models.
*   \[25/01/10] Supported fine-tuning the **[Phi-4](https://huggingface.co/microsoft/phi-4)** model.
*   \[24/12/21] Supported using **[SwanLab](https://github.com/SwanHubX/SwanLab)** for experiment tracking and visualization.
*   \[24/11/27] Supported fine-tuning the **[Skywork-o1](https://huggingface.co/Skywork/Skywork-o1-Open-Llama-3.1-8B)** model and the **[OpenO1](https://huggingface.co/datasets/O1-OPEN/OpenO1-SFT)** dataset.
*   \[24/10/09] Supported downloading pre-trained models and datasets from the **[Modelers Hub](https://modelers.cn/models)**.
*   \[24/09/19] Supported fine-tuning the **[Qwen2.5](https://qwenlm.github.io/blog/qwen2.5/)** models.
*   \[24/08/30] Supported fine-tuning the **[Qwen2-VL](https://qwenlm.github.io/blog/qwen2-vl/)** models.
*   \[24/08/27] Supported **[Liger Kernel](https://github.com/linkedin/Liger-Kernel)**.
*   \[24/08/09] Supported **[Adam-mini](https://github.com/zyushun/Adam-mini)** optimizer.
*   \[24/07/04] Supported [contamination-free packed training](https://github.com/MeetKai/functionary/tree/main/functionary/train/packing).
*   \[24/06/16] Supported **[PiSSA](https://arxiv.org/abs/2404.02948)** algorithm.
*   \[24/06/07] Supported fine-tuning the **[Qwen2](https://qwenlm.github.io/blog/qwen2/)** and **[GLM-4](https://github.com/THUDM/GLM-4)** models.
*   \[24/05/26] Supported **[SimPO](https://arxiv.org/abs/2405.14734)** algorithm for preference learning.
*   \[24/05/20] Supported fine-tuning the **PaliGemma** series models.
*   \[24/05/18] Supported **[KTO](https://arxiv.org/abs/2402.01306)** algorithm for preference learning.
*   \[24/05/14] Supported training and inference on the Ascend NPU devices.
*   \[24/04/26] Supported fine-tuning the **LLaVA-1.5** multimodal LLMs.
*   \[24/04/22] Provided a **[Colab notebook](https://colab.research.google.com/drive/1eRTPn37ltBbYsISy9Aw2NuI2Aq5CQrD9?usp=sharing)** for fine-tuning the Llama-3 model on a free T4 GPU.
*   \[24/04/21] Supported **[Mixture-of-Depths](https://arxiv.org/abs/2404.02258)** according to [AstraMindAI's implementation](https://github.com/astramind-ai/Mixture-of-depths).
*   \[24/04/16] Supported **[BAdam](https://arxiv.org/abs/2404.02827)** optimizer.
*   \[24/04/16] Supported **[unsloth](https://github.com/unslothai/unsloth)**'s long-sequence training.
*   \[24/03/31] Supported **[ORPO](https://arxiv.org/abs/2403.07691)**.
*   \[24/03/21] Published paper "[LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models](https://arxiv.org/abs/2403.13372)" at arXiv!
*   \[24/03/20] Supported **FSDP+QLoRA**.
*   \[24/03/13] Supported **[LoRA+](https://arxiv.org/abs/2402.12354)**.
*   \[24/03/07] Supported **[GaLore](https://arxiv.org/abs/2403.03507)** optimizer.
*   \[24/03/07] Integrated **[vLLM](https://github.com/vllm-project/vllm)** for faster and concurrent inference.
*   \[24/02/28] Supported weight-decomposed LoRA (**[DoRA](https://arxiv.org/abs/2402.09353)**).
*   \[24/02/15] Supported **block expansion** proposed by [LLaMA Pro](https://github.com/TencentARC/LLaMA-Pro).
*   \[24/02/05] Qwen1.5 (Qwen2 beta version) series models are supported in LLaMA-Factory.
*   \[24/01/18] Supported **agent tuning** for most models.
*   \[23/12/23] Supported **[unsloth](https://github.com/unslothai/unsloth)**'s implementation to boost LoRA tuning for the LLaMA, Mistral and Yi models.
*   \[23/12/12] Supported fine-tuning the latest MoE model **[Mixtral 8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)** in our framework.
*   \[23/12/01] Supported downloading pre-trained models and datasets from the **[ModelScope Hub](https://modelscope.cn/models)**.
*   \[23/10/21] Supported **[NEFTune](https://arxiv.org/abs/2310.05914)** trick for fine-tuning.
*   \[23/09/27] Supported **$S^2$-Attn** proposed by [LongLoRA](https://github.com/dvlab-research/LongLoRA) for the LLaMA models.
*   \[23/09/23] Integrated MMLU, C-Eval and CMMLU benchmarks in this repo.
*   \[23/09/10] Supported **[FlashAttention-2](https://github.com/Dao-AILab/flash-attention)**.
*   \[23/08/12] Supported **RoPE scaling** to extend the context length of the LLaMA models.
*   \[23/08/11] Supported **[DPO training](https://arxiv.org/abs/2305.18290)** for instruction-tuned models.
*   \[23/07/31] Supported **dataset streaming**.
*   \[23/07/29] Released two instruction-tuned 13B models at Hugging Face.
*   \[23/07/18] Developed an **all-in-one Web UI** for training, evaluation and inference.
*   \[23/07/09] Released **[FastEdit](https://github.com/hiyouga/FastEdit)** ⚡🩹.
*   \[23/06/29] Provided a **reproducible example** of training a chat model.
*   \[23/06/22] Aligned the [demo API](src/api_demo.py) with the [OpenAI's](https://platform.openai.com/docs/api-reference/chat) format.
*   \[23/06/03] Supported quantized training and inference (aka **[QLoRA](https://github.com/artidoro/qlora)**).
</details>

> [!TIP]
> If you cannot use the latest feature, please pull the latest code and install LLaMA-Factory again.

## Supported Models

A vast array of models are supported, including:

*   Baichuan 2, BLOOM, ChatGLM3, Command R, DeepSeek (Code/MoE), Falcon, Gemma, GLM-4, GPT-2, GPT-OSS, Granite, Hunyuan, Index, InternLM, Kimi-VL, Llama, Llama 2, Llama 3, LLaVA, Mistral, Phi, Qwen, Yi, and many more!

  *Please refer to [constants.py](src/llamafactory/extras/constants.py) for a complete list.*

## Supported Training Approaches

*   Pre-Training, Supervised Fine-Tuning, Reward Modeling, PPO, DPO, KTO, ORPO and SimPO.
    *  Full-tuning, Freeze-tuning, LoRA, QLoRA, OFT, and QOFT are supported for each approach.

## Provided Datasets

*   A variety of pre-training, supervised fine-tuning, and preference datasets are provided.
    *   Includes datasets from Hugging Face, ModelScope, and more.
    *   Details on dataset formats can be found in [data/README.md](data/README.md).

## Requirements

*   Python 3.9+
*   PyTorch 2.0.0+
*   Transformers 4.49.0+
*   Accelerate 0.34.0+
*   PEFT 0.14.0+
*   TRL 0.8.6+
*   And more... (see the full list in the original README)

## Getting Started

### Installation

Install LLaMA Factory from source:

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
```

Or use the Docker image: `docker run -it --rm --gpus=all --ipc=host hiyouga/llamafactory:latest`

### Data Preparation

Prepare your datasets by following the instructions in [data/README.md](data/README.md).

### Quickstart

Fine-tune, infer, and merge a Llama3-8B-Instruct model using:

```bash
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
llamafactory-cli chat examples/inference/llama3_lora_sft.yaml
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
```

Refer to [examples/README.md](examples/README.md) for advanced usage and distributed training.

### Fine-Tuning with LLaMA Board GUI (powered by Gradio)

```bash
llamafactory-cli webui
```

### Build Docker

Build and run a Docker container (CUDA, Ascend NPU, or ROCm):

```bash
#CUDA
cd docker/docker-cuda/
docker compose up -d
docker compose exec llamafactory bash

#Ascend NPU
cd docker/docker-npu/
docker compose up -d
docker compose exec llamafactory bash

#ROCm
cd docker/docker-rocm/
docker compose up -d
docker compose exec llamafactory bash
```

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

Add `report_to: wandb` and `run_name: test_run` to your YAML file and set your `WANDB_API_KEY`.

### Use SwanLab Logger

Add `use_swanlab: true` and `swanlab_run_name: test_run` to your YAML file, and configure your SwanLab API key or login.

## Projects using LLaMA Factory

(List of projects using LLaMA Factory - see the original README)

## License

This project is licensed under the [Apache-2.0 License](LICENSE).

## Citation

(See the original README for the BibTeX citation.)

## Acknowledgement

This project is built upon the shoulders of giants, including PEFT, TRL, QLoRA, and FastChat.