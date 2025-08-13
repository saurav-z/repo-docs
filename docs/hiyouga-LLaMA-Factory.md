<div align="center">
  <a href="https://github.com/hiyouga/LLaMA-Factory">
    <img src="assets/logo.png" alt="LLaMA Factory Logo" width="200"/>
  </a>
  <h1>LLaMA Factory: Fine-tune Your LLMs with Ease</h1>
  <p><b>Easily fine-tune 100+ large language models with zero-code CLI and Web UI.</b></p>
</div>

[![GitHub Repo stars](https://img.shields.io/github/stars/hiyouga/LLaMA-Factory?style=social)](https://github.com/hiyouga/LLaMA-Factory/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/hiyouga/LLaMA-Factory)](https://github.com/hiyouga/LLaMA-Factory/commits/main)
[![GitHub contributors](https://img.shields.io/github/contributors/hiyouga/LLaMA-Factory?color=orange)](https://github.com/hiyouga/LLaMA-Factory/graphs/contributors)
[![GitHub workflow](https://github.com/hiyouga/LLaMA-Factory/actions/workflows/tests.yml/badge.svg)](https://github.com/hiyouga/LLaMA-Factory/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/llamafactory)](https://pypi.org/project/llamafactory/)
[![Citation](https://img.shields.io/badge/citation-760-green)](https://scholar.google.com/scholar?cites=12620864006390196564)
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

<p>Used by <a href="https://aws.amazon.com/cn/blogs/machine-learning/how-apoidea-group-enhances-visual-information-extraction-from-banking-documents-with-multimodal-models-using-llama-factory-on-amazon-sagemaker-hyperpod/">Amazon</a>, <a href="https://developer.nvidia.com/rtx/ai-toolkit">NVIDIA</a>, <a href="https://help.aliyun.com/zh/pai/use-cases/fine-tune-a-llama-3-model-with-llama-factory/">Aliyun</a>, and more!</p>

<div align="center" markdown="1">
  <a href="https://warp.dev/llama-factory"><img alt="Warp sponsorship" width="400" src="assets/warp.jpg"></a><br><a href="https://warp.dev/llama-factory" style="font-size:larger;">Warp, the agentic terminal for developers</a><br><a href="https://warp.dev/llama-factory">Available for MacOS, Linux, & Windows</a>
  <a href="https://serpapi.com"><img alt="SerpAPI sponsorship" width="250" src="assets/serpapi.svg"> </a>
</div>

----
<div align="center">
  <img src="https://trendshift.io/api/badge/repositories/4535" alt="GitHub Trend">
</div>

👋 Join our [WeChat group](assets/wechat.jpg), [NPU user group](assets/wechat_npu.jpg) or [Alaya NeW user group](assets/wechat_alaya.png).

<p>[ English | <a href="README_zh.md">中文</a> ]</p>

<img src="https://github.com/user-attachments/assets/3991a3a8-4276-4d30-9cab-4cb0c4b9b99e" alt="Fine-tuning Example" width="600">

<br>

**Explore the possibilities and start fine-tuning with LLaMA Factory today!**

-   [Documentation (WIP)](https://llamafactory.readthedocs.io/en/latest/)
-   [Documentation (AMD GPU)](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/fine_tune/llama_factory_llama3.html)
-   [Colab (free)](https://colab.research.google.com/drive/1eRTPn37ltBbYsISy9Aw2NuI2Aq5CQrD9?usp=sharing)
-   [Local machine](#getting-started)
-   [PAI-DSW (free trial)](https://gallery.pai-ml.com/#/preview/deepLearning/nlp/llama_factory)
-   [Alaya NeW (cloud GPU deal)](https://docs.alayanew.com/docs/documents/useGuide/LLaMAFactory/mutiple/?utm_source=LLaMA-Factory)

> [!NOTE]
> Please be cautious when using third-party websites outside of the provided links.

## Key Features

*   🚀 **Extensive Model Support:** Fine-tune a vast range of models including LLaMA, Mistral, Mixtral, Qwen, Gemma, and more.
*   🛠️ **Versatile Training Methods:** Implement various methods, including pre-training, supervised fine-tuning, reward modeling, PPO, and DPO.
*   ⚙️ **Optimized Performance:** Benefit from LoRA, QLoRA, and other efficient techniques, along with advanced algorithms like GaLore and BAdam.
*   🧠 **Advanced Techniques:** Leverage cutting-edge features like FlashAttention-2, Unsloth, RoPE scaling, and NEFTune.
*   🎯 **Diverse Task Applications:** Tackle multi-turn dialogues, tool usage, image understanding, and other complex tasks.
*   📈 **Comprehensive Monitoring:** Utilize LlamaBoard, TensorBoard, Wandb, and other tools for experiment tracking.
*   ⚡ **Faster Inference:** Deploy with an OpenAI-style API, Gradio UI, and CLI using vLLM or SGLang for accelerated inference.

## Table of Contents

-   [Features](#key-features)
-   [Supported Models](#supported-models)
-   [Supported Training Approaches](#supported-training-approaches)
-   [Provided Datasets](#provided-datasets)
-   [Requirement](#requirement)
-   [Getting Started](#getting-started)
    -   [Installation](#installation)
    -   [Data Preparation](#data-preparation)
    -   [Quickstart](#quickstart)
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
---

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
| [Gemma 3/Gemma 3n](https://huggingface.co/google)                 | 1B/4B/6B/8B/12B/27B              | gemma3/gemma3n      |
| [GLM-4/GLM-4-0414/GLM-Z1](https://huggingface.co/zai-org)         | 9B/32B                           | glm4/glmz1          |
| [GLM-4.1V](https://huggingface.co/zai-org)                        | 9B                               | glm4v               |
| [GLM-4.5/GLM-4.5V](https://huggingface.co/zai-org)*               | 106B/355B                        | glm4_moe/glm4v_moe  |
| [GPT-2](https://huggingface.co/openai-community)                  | 0.1B/0.4B/0.8B/1.5B              | -                   |
| [GPT-OSS](https://huggingface.co/openai)                          | 20B/120B                         | gpt                 |
| [Granite 3.0-3.3](https://huggingface.co/ibm-granite)             | 1B/2B/3B/8B                      | granite3            |
| [Granite 4](https://huggingface.co/ibm-granite)                   | 7B                               | granite4            |
| [Hunyuan](https://huggingface.co/tencent/)                        | 7B                               | hunyuan             |
| [Index](https://huggingface.co/IndexTeam)                         | 1.9B                             | index               |
| [InternLM 2-3](https://huggingface.co/internlm)                   | 7B/8B/20B                        | intern2             |
| [InternVL 2.5-3](https://huggingface.co/OpenGVLab)                | 1B/2B/8B/14B/38B/78B             | intern_vl           |
| [Kimi-VL](https://huggingface.co/moonshotai)                      | 16B                              | kimi_vl             |
| [Llama](https://github.com/facebookresearch/llama)                | 7B/13B/33B/65B                   | -                   |
| [Llama 2](https://huggingface.co/meta-llama)                      | 7B/13B/70B                       | llama2              |
| [Llama 3-3.3](https://huggingface.co/meta-llama)                  | 1B/3B/8B/70B                     | llama3              |
| [Llama 4](https://huggingface.co/meta-llama)                      | 109B/402B                        | llama4              |
| [Llama 3.2 Vision](https://huggingface.co/meta-llama)             | 11B/90B                          | mllama              |
| [LLaVA-1.5](https://huggingface.co/llava-hf)                      | 7B/13B                           | llava               |
| [LLaVA-NeXT](https://huggingface.co/llava-hf)                     | 7B/8B/13B/34B/72B/110B           | llava_next          |
| [LLaVA-NeXT-Video](https://huggingface.co/llava-hf)               | 7B/34B                           | llava_next_video    |
| [MiMo](https://huggingface.co/XiaomiMiMo)                         | 7B                               | mimo                |
| [MiniCPM](https://huggingface.co/openbmb)                         | 0.5B/1B/2B/4B/8B                 | cpm/cpm3/cpm4       |
| [MiniCPM-o-2.6/MiniCPM-V-2.6](https://huggingface.co/openbmb)     | 8B                               | minicpm_o/minicpm_v |
| [Ministral/Mistral-Nemo](https://huggingface.co/mistralai)        | 8B/12B                           | ministral           |
| [Mistral/Mixtral](https://huggingface.co/mistralai)               | 7B/8x7B/8x22B                    | mistral             |
| [Mistral Small](https://huggingface.co/mistralai)                 | 24B                              | mistral_small       |
| [OLMo](https://huggingface.co/allenai)                            | 1B/7B                            | -                   |
| [PaliGemma/PaliGemma2](https://huggingface.co/google)             | 3B/10B/28B                       | paligemma           |
| [Phi-1.5/Phi-2](https://huggingface.co/microsoft)                 | 1.3B/2.7B                        | -                   |
| [Phi-3/Phi-3.5](https://huggingface.co/microsoft)                 | 4B/14B                           | phi                 |
| [Phi-3-small](https://huggingface.co/microsoft)                   | 7B                               | phi_small           |
| [Phi-4](https://huggingface.co/microsoft)                         | 14B                              | phi4                |
| [Pixtral](https://huggingface.co/mistralai)                       | 12B                              | pixtral             |
| [Qwen (1-2.5) (Code/Math/MoE/QwQ)](https://huggingface.co/Qwen)   | 0.5B/1.5B/3B/7B/14B/32B/72B/110B | qwen                |
| [Qwen3 (MoE/Instruct/Thinking)](https://huggingface.co/Qwen)      | 0.6B/1.7B/4B/8B/14B/32B/235B     | qwen3/qwen3_nothink |
| [Qwen2-Audio](https://huggingface.co/Qwen)                        | 7B                               | qwen2_audio         |
| [Qwen2.5-Omni](https://huggingface.co/Qwen)                       | 3B/7B                            | qwen2_omni          |
| [Qwen2-VL/Qwen2.5-VL/QVQ](https://huggingface.co/Qwen)            | 2B/3B/7B/32B/72B                 | qwen2_vl            |
| [Seed Coder](https://huggingface.co/ByteDance-Seed)               | 8B                               | seed_coder          |
| [Skywork o1](https://huggingface.co/Skywork)                      | 8B                               | skywork_o1          |
| [StarCoder 2](https://huggingface.co/bigcode)                     | 3B/7B/15B                        | -                   |
| [TeleChat2](https://huggingface.co/Tele-AI)                       | 3B/7B/35B/115B                   | telechat2           |
| [XVERSE](https://huggingface.co/xverse)                           | 7B/13B/65B                       | xverse              |
| [Yi/Yi-1.5 (Code)](https://huggingface.co/01-ai)                  | 1.5B/6B/9B/34B                   | yi                  |
| [Yi-VL](https://huggingface.co/01-ai)                             | 6B/34B                           | yi_vl               |
| [Yuan 2](https://huggingface.co/IEITYuan)                         | 2B/51B/102B                      | yuan                |

> [!NOTE]
> For the "base" models, the `template` argument can be chosen from `default`, `alpaca`, `vicuna` etc. But make sure to use the **corresponding template** for the "instruct/chat" models.
>
> Remember to use the **SAME** template in training and inference.
>
> \*: You should install the `transformers` from main branch and use `DISABLE_VERSION_CHECK=1` to skip version check.
>
> \*\*: You need to install a specific version of `transformers` to use the corresponding model.

Please refer to [constants.py](src/llamafactory/extras/constants.py) for a full list of models we supported.

You also can add a custom chat template to [template.py](src/llamafactory/data/template.py).

---

## Supported Training Approaches

| Approach               |     Full-tuning    |    Freeze-tuning   |       LoRA         |       QLoRA        |
| ---------------------- | ------------------ | ------------------ | ------------------ | ------------------ |
| Pre-Training           | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| Supervised Fine-Tuning | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| Reward Modeling        | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| PPO Training           | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| DPO Training           | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| KTO Training           | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| ORPO Training          | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| SimPO Training         | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |

> [!TIP]
> The implementation details of PPO can be found in [this blog](https://newfacade.github.io/notes-on-reinforcement-learning/17-ppo-trl.html).

---

## Provided Datasets

<details><summary>Pre-training datasets</summary>

-   [Wiki Demo (en)](data/wiki_demo.txt)
-   [RefinedWeb (en)](https://huggingface.co/datasets/tiiuae/falcon-refinedweb)
-   [RedPajama V2 (en)](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2)
-   [Wikipedia (en)](https://huggingface.co/datasets/olm/olm-wikipedia-20221220)
-   [Wikipedia (zh)](https://huggingface.co/datasets/pleisto/wikipedia-cn-20230720-filtered)
-   [Pile (en)](https://huggingface.co/datasets/EleutherAI/pile)
-   [SkyPile (zh)](https://huggingface.co/datasets/Skywork/SkyPile-150B)
-   [FineWeb (en)](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
-   [FineWeb-Edu (en)](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
-   [The Stack (en)](https://huggingface.co/datasets/bigcode/the-stack)
-   [StarCoder (en)](https://huggingface.co/datasets/bigcode/starcoderdata)

</details>

<details><summary>Supervised fine-tuning datasets</summary>

-   [Identity (en&zh)](data/identity.json)
-   [Stanford Alpaca (en)](https://github.com/tatsu-lab/stanford_alpaca)
-   [Stanford Alpaca (zh)](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3)
-   [Alpaca GPT4 (en&zh)](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)
-   [Glaive Function Calling V2 (en&zh)](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)
-   [LIMA (en)](https://huggingface.co/datasets/GAIR/lima)
-   [Guanaco Dataset (multilingual)](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset)
-   [BELLE 2M (zh)](https://huggingface.co/datasets/BelleGroup/train_2M_CN)
-   [BELLE 1M (zh)](https://huggingface.co/datasets/BelleGroup/train_1M_CN)
-   [BELLE 0.5M (zh)](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN)
-   [BELLE Dialogue 0.4M (zh)](https://huggingface.co/datasets/BelleGroup/generated_chat_0.4M)
-   [BELLE School Math 0.25M (zh)](https://huggingface.co/datasets/BelleGroup/school_math_0.25M)
-   [BELLE Multiturn Chat 0.8M (zh)](https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M)
-   [UltraChat (en)](https://github.com/thunlp/UltraChat)
-   [OpenPlatypus (en)](https://huggingface.co/datasets/garage-bAInd/Open-Platypus)
-   [CodeAlpaca 20k (en)](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k)
-   [Alpaca CoT (multilingual)](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT)
-   [OpenOrca (en)](https://huggingface.co/datasets/Open-Orca/OpenOrca)
-   [SlimOrca (en)](https://huggingface.co/datasets/Open-Orca/SlimOrca)
-   [MathInstruct (en)](https://huggingface.co/datasets/TIGER-Lab/MathInstruct)
-   [Firefly 1.1M (zh)](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M)
-   [Wiki QA (en)](https://huggingface.co/datasets/wiki_qa)
-   [Web QA (zh)](https://huggingface.co/datasets/suolyer/webqa)
-   [WebNovel (zh)](https://huggingface.co/datasets/zxbsmk/webnovel_cn)
-   [Nectar (en)](https://huggingface.co/datasets/berkeley-nest/Nectar)
-   [deepctrl (en&zh)](https://www.modelscope.cn/datasets/deepctrl/deepctrl-sft-data)
-   [Advertise Generating (zh)](https://huggingface.co/datasets/HasturOfficial/adgen)
-   [ShareGPT Hyperfiltered (en)](https://huggingface.co/datasets/totally-not-an-llm/sharegpt-hyperfiltered-3k)
-   [ShareGPT4 (en&zh)](https://huggingface.co/datasets/shibing624/sharegpt_gpt4)
-   [UltraChat 200k (en)](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)
-   [AgentInstruct (en)](https://huggingface.co/datasets/THUDM/AgentInstruct)
-   [LMSYS Chat 1M (en)](https://huggingface.co/datasets/lmsys/lmsys-chat-1m)
-   [Evol Instruct V2 (en)](https://huggingface.co/datasets/WizardLM/WizardLM_evol_instruct_V2_196k)
-   [Cosmopedia (en)](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia)
-   [STEM (zh)](https://huggingface.co/datasets/hfl/stem_zh_instruction)
-   [Ruozhiba (zh)](https://huggingface.co/datasets/hfl/ruozhiba_gpt4_turbo)
-   [Neo-sft (zh)](https://huggingface.co/datasets/m-a-p/neo_sft_phase2)
-   [Magpie-Pro-300K-Filtered (en)](https://huggingface.co/datasets/Magpie-Align/Magpie-Pro-300K-Filtered)
-   [Magpie-ultra-v0.1 (en)](https://huggingface.co/datasets/argilla/magpie-ultra-v0.1)
-   [WebInstructSub (en)](https://huggingface.co/datasets/TIGER-Lab/WebInstructSub)
-   [OpenO1-SFT (en&zh)](https://huggingface.co/datasets/O1-OPEN/OpenO1-SFT)
-   [Open-Thoughts (en)](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k)
-   [Open-R1-Math (en)](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k)
-   [Chinese-DeepSeek-R1-Distill (zh)](https://huggingface.co/datasets/Congliu/Chinese-DeepSeek-R1-Distill-data-110k-SFT)
-   [LLaVA mixed (en&zh)](https://huggingface.co/datasets/BUAADreamer/llava-en-zh-300k)
-   [Pokemon-gpt4o-captions (en&zh)](https://huggingface.co/datasets/jugg1024/pokemon-gpt4o-captions)
-   [Open Assistant (de)](https://huggingface.co/datasets/mayflowergmbh/oasst_de)
-   [Dolly 15k (de)](https://huggingface.co/datasets/mayflowergmbh/dolly-15k_de)
-   [Alpaca GPT4 (de)](https://huggingface.co/datasets/mayflowergmbh/alpaca-gpt4_de)
-   [OpenSchnabeltier (de)](https://huggingface.co/datasets/mayflowergmbh/openschnabeltier_de)
-   [Evol Instruct (de)](https://huggingface.co/datasets/mayflowergmbh/evol-instruct_de)
-   [Dolphin (de)](https://huggingface.co/datasets/mayflowergmbh/dolphin_de)
-   [Booksum (de)](https://huggingface.co/datasets/mayflowergmbh/booksum_de)
-   [Airoboros (de)](https://huggingface.co/datasets/mayflowergmbh/airoboros-3.0_de)
-   [Ultrachat (de)](https://huggingface.co/datasets/mayflowergmbh/ultra-chat_de)

</details>

<details><summary>Preference datasets</summary>

-   [DPO mixed (en&zh)](https://huggingface.co/datasets/hiyouga/DPO-En-Zh-20k)
-   [UltraFeedback (en)](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized)
-   [COIG-P (zh)](https://huggingface.co/datasets/m-a-p/COIG-P)
-   [RLHF-V (en)](https://huggingface.co/datasets/openbmb/RLHF-V-Dataset)
-   [VLFeedback (en)](https://huggingface.co/datasets/Zhihui/VLFeedback)
-   [RLAIF-V (en)](https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset)
-   [Orca DPO Pairs (en)](https://huggingface.co/datasets/Intel/orca_dpo_pairs)
-   [HH-RLHF (en)](https://huggingface.co/datasets/Anthropic/hh-rlhf)
-   [Nectar (en)](https://huggingface.co/datasets/berkeley-nest/Nectar)
-   [Orca DPO (de)](https://huggingface.co/datasets/mayflowergmbh/intel_orca_dpo_pairs_de)
-   [KTO mixed (en)](https://huggingface.co/datasets/argilla/kto-mix-15k)

</details>

> [!NOTE]
> Some datasets require confirmation before using them, so we recommend logging in with your Hugging Face account using these commands.

```bash
pip install --upgrade huggingface_hub
huggingface-cli login
```

---

## Requirement

| Mandatory    | Minimum | Recommend |
| ------------ | ------- | --------- |
| python       | 3.9     | 3.10      |
| torch        | 2.0.0   | 2.6.0     |
| torchvision  | 0.15.0  | 0.21.0    |
| transformers | 4.49.0  | 4.50.0    |
| datasets     | 2.16.0  | 3.2.0     |
| accelerate   | 0.34.0  | 1.2.1     |
| peft         | 0.14.0  | 0.15.1    |
| trl          | 0.8.6   | 0.9.6     |

| Optional     | Minimum | Recommend |
| ------------ | ------- | --------- |
| CUDA         | 11.6    | 12.2      |
| deepspeed    | 0.10.0  | 0.16.4    |
| bitsandbytes | 0.39.0  | 0.43.1    |
| vllm         | 0.4.3   | 0.8.2     |
| flash-attn   | 2.5.6   | 2.7.2     |

### Hardware Requirement