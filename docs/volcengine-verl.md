# verl: Unleash the Power of Reinforcement Learning for LLMs

**verl is a versatile and efficient RL training library that enables you to train and optimize large language models (LLMs) for advanced performance.**  Get started today and explore the possibilities by visiting the [verl GitHub repository](https://github.com/volcengine/verl)!

## Key Features

*   **Flexible and Extensible:** Easily integrate and customize diverse RL algorithms. Build RL dataflows such as GRPO, PPO in a few lines of code with the hybrid-controller programming model.
*   **Seamless LLM Integration:**  Compatible with existing LLM infrastructure, including FSDP, Megatron-LM, vLLM, and SGLang, through modular APIs.
*   **Efficient Resource Management:** Supports flexible device mapping for optimal GPU utilization across various cluster sizes.
*   **Hugging Face Integration:** Ready-to-use with popular Hugging Face models.
*   **State-of-the-Art Performance:** Achieves SOTA LLM training and inference throughput.
*   **3D-HybridEngine:** Eliminates memory redundancy and significantly reduces communication overhead, improving training speed.
*   **Pre-trained LLM Support:** verl is compatible with Hugging Face Transformers and Modelscope Hub: [Qwen-3](https://github.com/volcengine/verl/blob/main/examples/grpo_trainer/run_qwen3-8b.sh), Qwen-2.5, Llama3.1, Gemma2, DeepSeek-LLM, etc
*   **SFT & RL Support:** Support for Supervised fine-tuning, Reinforcement learning with [PPO](examples/ppo_trainer/), [GRPO](examples/grpo_trainer/), [ReMax](examples/remax_trainer/), [REINFORCE++](https://verl.readthedocs.io/en/latest/examples/config.html#algorithm), [RLOO](examples/rloo_trainer/), [PRIME](recipe/prime/), [DAPO](recipe/dapo/), [DrGRPO](recipe/drgrpo), [KL_Cov & Clip_Cov](recipe/entropy) etc.
*   **Advanced RL Capabilities:** Support model-based reward and function-based reward (verifiable reward) for math, [coding](https://github.com/volcengine/verl/tree/main/recipe/dapo), etc
*   **Multi-Modal and Multi-turn Support:** Support vision-language models (VLMs) and [multi-modal RL](examples/grpo_trainer/run_qwen2_5_vl-7b.sh) with Qwen2.5-vl, Kimi-VL and [multi-turn with tool calling](https://github.com/volcengine/verl/tree/main/examples/sglang_multiturn)
*   **Advanced Features:** LLM alignment recipes such as [Self-play preference optimization (SPPO)](https://github.com/volcengine/verl/tree/main/recipe/sppo) and others.
*   **Performance Enhancements:** Includes Flash attention 2, [sequence packing](examples/ppo_trainer/run_qwen2-7b_seq_balance.sh), [sequence parallelism](examples/ppo_trainer/run_deepseek7b_llm_sp2.sh) support via DeepSpeed Ulysses, [LoRA](examples/sft/gsm8k/run_qwen_05_peft.sh), [Liger-kernel](examples/sft/gsm8k/run_qwen_05_sp2_liger.sh).
*   **Scalability:** Scales up to 671B models and hundreds of GPUs with [expert parallelism](https://github.com/volcengine/verl/pull/1467)
*   **Memory Efficiency:** Multi-gpu [LoRA RL](https://verl.readthedocs.io/en/latest/advance/ppo_lora.html) support to save memory.
*   **Experiment Tracking:** Experiment tracking with wandb, swanlab, mlflow and tensorboard.

## News

*   \[2025/07] verl keynote at [AWS AI Hours Singapore](https://pages.awscloud.com/aws-ai-hours-sg.html#agenda) on 7/8, verl & verl-agent project updates at [Agent for SWE meetup](https://lu.ma/e498qhsi) by LF AI & Data Singapore on 7/11, and ICML @ Vancouver on 7/16.
*   \[2025/06] verl with Megatron backend enables large MoE models such as [DeepSeek-671b and Qwen3-236b](https://verl.readthedocs.io/en/latest/perf/dpsk.html).
*   \[2025/06] verl team will provide latest project updates at [PyTorch Day China](https://www.lfasiallc.com/pytorch-day-china/) on June 7th. Meet our dev team in Beijing!
*   \[2025/04] [Seed-Thinking-v1.5](https://github.com/ByteDance-Seed/Seed-Thinking-v1.5/blob/main/seed-thinking-v1.5.pdf) tech report is released! Trained with verl, Seed-Thinking-v1.5 achieves 86.7 on AIME 2024, 55.0 on Codeforces and 77.3 on GPQA, demonstrating excellent reasoning abilities in STEM and coding. Beyond reasoning tasks, the method demonstrates notable generalization across diverse domains.
*   \[2025/03] [DAPO](https://dapo-sia.github.io/) is the open-sourced SOTA RL algorithm that achieves 50 points on AIME 2024 based on the Qwen2.5-32B pre-trained model, surpassing the previous SOTA achieved by DeepSeek's GRPO (DeepSeek-R1-Zero-Qwen-32B). DAPO's training is fully powered by verl and the reproduction code is available in `recipe/dapo` now.
    *   *See the full list of news and updates in the original README.*

## Getting Started

*   **Documentation:** <a href="https://verl.readthedocs.io/en/latest/index.html"><b>verl Documentation</b></a>

**Quickstart:**

*   [Installation](https://verl.readthedocs.io/en/latest/start/install.html)
*   [Quickstart](https://verl.readthedocs.io/en/latest/start/quickstart.html)
*   [Programming Guide](https://verl.readthedocs.io/en/latest/hybrid_flow.html) & [Tech Talk](https://hcqnc.xetlk.com/sl/3vACOK) (in Chinese)
*   [PPO in verl](https://verl.readthedocs.io/en/latest/algo/ppo.html)
*   [GRPO in verl](https://verl.readthedocs.io/en/latest/algo/grpo.html)

**Running a PPO example step-by-step:**

*   [Prepare Data for Post-Training](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html)
*   [Implement Reward Function for Dataset](https://verl.readthedocs.io/en/latest/preparation/reward_function.html)
*   [PPO Example Architecture](https://verl.readthedocs.io/en/latest/examples/ppo_code_architecture.html)
*   [Config Explanation](https://verl.readthedocs.io/en/latest/examples/config.html)

**Reproducible algorithm baselines:**

*   [RL performance on coding, math](https://verl.readthedocs.io/en/latest/algo/baseline.html)

**For code explanation and advance usage (extension):**

*   PPO Trainer and Workers
    *   [PPO Ray Trainer](https://verl.readthedocs.io/en/latest/workers/ray_trainer.html)
    *   [PyTorch FSDP Backend](https://verl.readthedocs.io/en/latest/workers/fsdp_workers.html)
    *   [Megatron-LM Backend](https://verl.readthedocs.io/en/latest/index.html)
*   Advanced Usage and Extension
    *   [Add Models with the FSDP Backend](https://verl.readthedocs.io/en/latest/advance/fsdp_extension.html)
    *   [Add Models with the Megatron-LM Backend](https://verl.readthedocs.io/en/latest/advance/megatron_extension.html)
    *   [Multi-turn Rollout Support](https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html)
    *   [Search Tool Integration](https://verl.readthedocs.io/en/latest/sglang_multiturn/search_tool_example.html)
    *   [Sandbox Fusion Integration](https://verl.readthedocs.io/en/latest/examples/sandbox_fusion_example.html)
    *   [Deployment using Separate GPU Resources](https://github.com/volcengine/verl/tree/main/examples/split_placement)
    *   [Extend to Other RL(HF) algorithms](https://verl.readthedocs.io/en/latest/advance/dpo_extension.html)
    *   [Ray API design tutorial](https://verl.readthedocs.io/en/latest/advance/placement.html)

**Blogs from the community**

*   *See the full list of blogs in the original README.*

## Performance Tuning Guide

For optimal performance, consult our detailed [performance tuning guide](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html).

## Upgrade to vLLM >= v0.8.2

verl now supports vLLM>=0.8.2 when using FSDP as the training backend. Please refer to [this document](https://github.com/volcengine/verl/blob/main/docs/README_vllm0.8.md) for the installation guide and more information. Please avoid vllm 0.7.x, which contains bugs that may lead to OOMs and unexpected errors.

## Use Latest SGLang

SGLang is fully supported with verl, and SGLang RL Group is working extensively on building unique features, including multi-turn agentic RL, VLM RLHF, server-based RL, and partial rollout. Please refer to [this document](https://verl.readthedocs.io/en/latest/workers/sglang_worker.html) for the installation guide and more information.

## Upgrade to FSDP2

verl is fully embracing FSDP2! FSDP2 is recommended by torch distributed team, providing better throughput and memory usage, and is composible with other features (e.g. torch.compile). To enable FSDP2, simply use verl main and set the following options:
```
actor_rollout_ref.ref.strategy=fsdp2
actor_rollout_ref.actor.strategy=fsdp2
critic.strategy=fsdp2 
reward_model.strategy=fsdp2 
```
Furthermore, FSDP2 cpu offloading is compatible with gradient accumulation. You can turn it on to save memory with `actor_rollout_ref.actor.fsdp_config.offload_policy=True`. For more details, see https://github.com/volcengine/verl/pull/1026

## AMD Support (ROCm Kernel)

verl now supports FSDP as the training engine (Megatron support coming soon) and both integrates with vLLM and SGLang as inference engines. Please refer to [this document](https://github.com/volcengine/verl/blob/main/docs/amd_tutorial/amd_build_dockerfile_page.rst) for the installation guide and more information, and [this document](https://github.com/volcengine/verl/blob/main/docs/amd_tutorial/amd_vllm_page.rst) for the vLLM performance tuning for ROCm.

## Citation and Acknowledgements

If you use verl in your research, please cite the following papers:

*   [HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256v2)
*   [A Framework for Training Large Language Models for Code Generation via Proximal Policy Optimization](https://i.cs.hku.hk/~cwu/papers/gmsheng-NL2Code24.pdf)

```bibtex
@article{sheng2024hybridflow,
  title   = {HybridFlow: A Flexible and Efficient RLHF Framework},
  author  = {Guangming Sheng and Chi Zhang and Zilingfeng Ye and Xibin Wu and Wang Zhang and Ru Zhang and Yanghua Peng and Haibin Lin and Chuan Wu},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2409.19256}
}
```

*verl* is inspired by the design of Nemo-Aligner, Deepspeed-chat, and OpenRLHF.  It is a collaborative project supported by Bytedance, Anyscale, LMSys.org, and contributions from various academic and industry partners.  (See original README for full list).

## Awesome work using verl

*   *See the full list of projects using verl in the original README.*

## Contribution Guide

For information on contributing to verl, please refer to the [contributions guide](CONTRIBUTING.md).

## About [ByteDance Seed Team](https://team.doubao.com/)

The ByteDance Seed Team is dedicated to advancing AI foundation models.  Learn more about them via the links below:
<div>
  <a href="https://team.doubao.com/">
    <img src="https://img.shields.io/badge/Website-%231e37ff?style=for-the-badge&logo=bytedance&logoColor=white"></a>
  <a href="https://github.com/user-attachments/assets/469535a8-42f2-4797-acdf-4f7a1d4a0c3e">
    <img src="https://img.shields.io/badge/WeChat-07C160?style=for-the-badge&logo=wechat&logoColor=white"></a>
 <a href="https://www.xiaohongshu.com/user/profile/668e7e15000000000303157d?xsec_token=ABl2-aqekpytY6A8TuxjrwnZskU-6BsMRE_ufQQaSAvjc%3D&xsec_source=pc_search">
    <img src="https://img.shields.io/badge/Xiaohongshu-%23FF2442?style=for-the-badge&logo=xiaohongshu&logoColor=white"></a>
  <a href="https://www.zhihu.com/org/dou-bao-da-mo-xing-tuan-dui/">
    <img src="https://img.shields.io/badge/zhihu-%230084FF?style=for-the-badge&logo=zhihu&logoColor=white"></a>

</div>
---

**Join the Team:**  We are HIRING!  Contact [haibin.lin@bytedance.com](mailto:haibin.lin@bytedance.com) if you're interested in RL opportunities.