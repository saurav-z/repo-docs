<div align="center">
  <h1>verl: Revolutionizing LLM Training with Reinforcement Learning</h1>
  <p>verl is an open-source, high-performance RL library from ByteDance Seed team, designed for efficient and flexible RL training of large language models.</p>
</div>

<div align="center">
  [<img src="https://devin.ai/assets/deepwiki-badge.png" alt="Ask DeepWiki.com" height="20"/>](https://deepwiki.com/volcengine/verl)
  [![GitHub Repo stars](https://img.shields.io/github/stars/volcengine/verl)](https://github.com/volcengine/verl/stargazers)
  [![Twitter](https://img.shields.io/twitter/follow/verl_project)](https://twitter.com/verl_project)
  <a href="https://join.slack.com/t/verlgroup/shared_invite/zt-2w5p9o4c3-yy0x2Q56s_VlGLsJ93A6vA"><img src="https://img.shields.io/badge/Slack-verl-blueviolet?logo=slack&amp"></a>
  <a href="https://arxiv.org/pdf/2409.19256"><img src="https://img.shields.io/static/v1?label=EuroSys&message=Paper&color=red"></a>
  [![Documentation](https://img.shields.io/badge/documentation-blue)](https://verl.readthedocs.io/en/latest/)
  <a href="https://raw.githubusercontent.com/eric-haibin-lin/verl-community/refs/heads/main/WeChat.JPG"><img src="https://img.shields.io/badge/微信-green?logo=wechat&amp"></a>
</div>

![seed logo](https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216)

## Key Features

*   **Flexible RL Algorithms:** Easily extend and implement diverse RL algorithms like GRPO, PPO, and more.
*   **Seamless LLM Integration:** Integrates with popular LLM frameworks such as FSDP, Megatron-LM, vLLM, and SGLang.
*   **Efficient Device Mapping:** Supports flexible model placement across GPUs for optimal resource utilization and scalability.
*   **Hugging Face Compatibility:** Ready-to-use with popular Hugging Face models.
*   **State-of-the-Art Performance:** Achieves high throughput with SOTA LLM training and inference integrations.
*   **3D-HybridEngine:** Efficient actor model resharding to reduce communication overhead.
*   **LoRA and other advanced features** Includes flash attention 2, sequence packing, sequence parallelism via DeepSpeed Ulysses, LoRA, Liger-kernel and scales up to 671B models.
*   **Experiment Tracking:** Supports wandb, swanlab, mlflow, and tensorboard.

## News

*   **[2025/06]**: verl supports MoE models.
*   **[2025/06]**: verl team will provide latest project updates at PyTorch Day China.
*   **[2025/05]**: PF-PPO, accepted to ICML 2025, is now supported in verl!
*   **[2025/04]**: Seed-Thinking-v1.5 tech report is released!
*   **[2025/04]**: VAPO (value-based augmented PPO) paper covers our latest RL method for reasoning models.
*   **[2025/03]**: DAPO, open-sourced SOTA RL algorithm that achieves 50 points on AIME 2024.
<details><summary> more... </summary>
<ul>

  <li>[2025/04] verl will be presented at ICLR 2025 Expo, SCI-FM workshop and LMSys afterparty.</li>
  <li>[2025/03] verl v0.3.0.post1 is released!</li>
  <li>[2025/05] verl will be presented at A2M Shanghai.</li>
  <li>[2025/05] verl will be presented at GOSIM x PyTorch Day 2025.</li>
  <li>[2025/03] verl programming model at the vLLM Beijing Meetup and SGLang-LMSYS Org Meetup in Sunnyvale.</li>
  <li>[2025/03] verl(HybridFlow) will be presented at EuroSys 2025.</li>
  <li>[2025/02] verl v0.2.0.post2 is released!</li>
  <li>[2025/02] verl in the Bytedance/NVIDIA/Anyscale Ray Meetup.</li>
  <li>[2025/01] Doubao-1.5-pro is released with SOTA-level performance.</li>
  <li>[2024/12] verl is presented at Ray Forward 2024.</li>
  <li>[2024/12] The team presented Post-training LLMs: From Algorithms to Infrastructure at NeurIPS 2024.</li>
  <li>[2024/10] verl is presented at Ray Summit.</li>
  <li>[2024/08] HybridFlow (verl) is accepted to EuroSys 2025.</li>
</ul>
</details>

## Getting Started

*   **[Documentation](https://verl.readthedocs.io/en/latest/index.html)**
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

*   [When Reasoning Models Break Tokenization: The Hidden Complexity of Multiturn Training](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/fast_tokenization/multiturn_tokenization_and_masking.md)
*   [verl deployment on AWS SageMaker](https://medium.com/@kaige.yang0110/run-verl-on-sagemaker-using-4x8-l40s-gpus-8e6d5c3c61d3)
*   [verl x SGLang Multi-turn Code Walkthrough](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/code-walk-through/readme_EN.md)
*   [Optimizing SGLang Memory Usage in verl](https://hebiao064.github.io/rl-memory-management)
*   [SGLang, verl, OpenBMB and Tsinghua University: Pioneering End-to-End Multi-Turn RLHF](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/verl-multiturn-rollout-Release.md)
*   [Reinforcement Learning from Human Feedback on AMD GPUs with verl and ROCm Integration](https://rocm.blogs.amd.com/artificial-intelligence/verl-large-scale/README.html)
*   [veMLP x verl ：玩转强化学习训练](https://mp.weixin.qq.com/s/7nbqxk4knMGd-hQE9ls2tA)
*   [使用 verl 进行 GRPO 分布式强化学习训练最佳实践](https://www.volcengine.com/docs/6459/1463942)
*   [HybridFlow verl 原文浅析](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/readme.md)
*   [最高提升 20 倍吞吐量！豆包大模型团队发布全新 RLHF 框架，现已开源！](https://team.doubao.com/en/blog/%E6%9C%80%E9%AB%98%E6%8F%90%E5%8D%8720%E5%80%8D%E5%90%9E%E5%90%90%E9%87%8F-%E8%B1%86%E5%8C%85%E5%A4%A7%E6%A0%A1%E5%9E%8B%E5%9B%A2%E9%98%9F%E5%8F%91%E5%B8%83%E5%85%A8%E6%96%B0-rlhf-%E6%A1%86%E6%9E%B6-%E7%8E%B0%E5%B7%B2%E5%BC%80%E6%BA%90)

## Performance Tuning Guide

*   [Performance Tuning Guide](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html)

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

verl draws inspiration from Nemo-Aligner, Deepspeed-chat, and OpenRLHF, and is supported by contributions from Bytedance, Anyscale, LMSys.org, [Alibaba Qwen team](https://github.com/QwenLM/), Shanghai AI Lab, Tsinghua University, UC Berkeley, UCLA, UIUC, University of Hong Kong, ke.com, [All Hands AI](https://www.all-hands.dev/), [ModelBest](http://modelbest.cn/), JD AI Lab, Microsoft Research, [StepFun](https://www.stepfun.com/), Amazon, LinkedIn, Meituan, [Camel-AI](https://www.camel-ai.org/), [OpenManus](https://github.com/OpenManus), Xiaomi, NVIDIA research, [Baichuan](https://www.baichuan-ai.com/home), [RedNote](https://www.xiaohongshu.com/), [SwissAI](https://www.swiss-ai.org/), [Moonshot AI (Kimi)](https://www.moonshot-ai.com/), Baidu, Snowflake, Skywork.ai, JetBrains, [IceSword Lab](https://www.iceswordlab.com), and many more.

## Awesome Projects Using verl

A curated list of projects built with verl:

*   [TinyZero](https://github.com/Jiayi-Pan/TinyZero): DeepSeek R1 Zero recipe reproduction for reasoning tasks.
*   [SkyThought](https://github.com/NovaSky-AI/SkyThought): RL training for Sky-T1-7B.
*   [simpleRL-reason](https://github.com/hkust-nlp/simpleRL-reason): SimpleRL-Zoo: Zero Reinforcement Learning for Open Base Models.
*   [Easy-R1](https://github.com/hiyouga/EasyR1): Multi-modal RL training framework.
*   [OpenManus-RL](https://github.com/OpenManus/OpenManus-RL): LLM Agents RL tuning framework.
*   [rllm](https://github.com/agentica-project/rllm): Async RL training with verl-pipeline.
*   [RAGEN](https://github.com/ZihanWang314/ragen): General-purpose reasoning agent training framework.
*   [Search-R1](https://github.com/PeterGriffinJin/Search-R1): RL with reasoning and searching LLMs.
*   [ReSearch](https://github.com/Agent-RL/ReSearch): Learning to Reason with Search for LLMs.
*   [Skywork-OR1](https://github.com/SkyworkAI/Skywork-OR1): Skywork open reasoner series.
*   [ToRL](https://github.com/GAIR-NLP/ToRL): Scaling tool-integrated RL.
*   [Absolute Zero Reasoner](https://github.com/LeapLabTHU/Absolute-Zero-Reasoner): A no human curated data self-play framework for reasoning.
*   [verl-agent](https://github.com/langfengQ/verl-agent): A scalable training framework for long-horizon LLM/VLM agents.
*   [RL-Factory](https://github.com/Simple-Efficient/RL-Factory): Easy and efficient RL post-training framework for Agentic Learning.
*   [ReTool](https://retool-rl.github.io/): ReTool: reinforcement learning for strategic tool use in LLMs.
*   [verl-tool](https://github.com/TIGER-AI-Lab/verl-tool): An unified and easy-to-extend tool-agent training framework based on verl.
*   [PRIME](https://github.com/PRIME-RL/PRIME): Process reinforcement through implicit rewards.
*   [GUI-R1](https://github.com/ritzz-ai/GUI-R1): GUI-R1: A Generalist R1-style Vision-Language Action Model For GUI Agents.
*   [DeepRetrieval](https://github.com/pat-jj/DeepRetrieval): RL Training of Search Agent with Search/Retrieval Outcome.
*   [Code-R1](https://github.com/ganler/code-r1): Reproducing R1 for Code with Reliable Rewards.
*   [DeepResearcher](https://github.com/GAIR-NLP/DeepResearcher): Scaling deep research via reinforcement learning in real-world environments.
*   [VAGEN](https://github.com/RAGEN-AI/VAGEN): Training VLM agents with multi-turn reinforcement learning.
*   [RM-R1](https://arxiv.org/abs/2505.02387): RL training of reasoning reward models.
*   [LUFFY](https://arxiv.org/pdf/2504.14945): Learning to Reason under Off-Policy Guidance.
*   [DeepMath](https://github.com/zwhe99/DeepMath): DeepMath-103K data and series models for math reasoning.
*   [Entropy Mechanism of RL](https://github.com/PRIME-RL/Entropy-Mechanism-of-RL): The Entropy Mechanism of Reinforcement Learning for Large Language Model Reasoning.
*   [LLaSA-TTS-GRPO](https://github.com/channel-io/ch-tts-llasa-rl-grpo): TTS fine-tuning with GRPO optimization based on LLASA models.
*   [PF-PPO](https://arxiv.org/abs/2409.06957): Policy Filtration for PPO.
*   [RACRO](https://github.com/gyhdog99/RACRO2): Build multi-modal reasoning models via decoupling it into query-conditioned captioning and text-only reasoning.

Explore more projects in the [recipe](recipe/README.md) section.

## Contribution Guide

We welcome contributions!  Please refer to the [contribution guide](CONTRIBUTING.md) for details.

## About ByteDance Seed Team

verl is a project of the ByteDance Seed Team, a research group dedicated to building state-of-the-art AI foundation models.

*   [ByteDance Seed Team Website](https://team.doubao.com/)
*   [WeChat](https://github.com/user-attachments/assets/469535a8-42f2-4797-acdf-4f7a1d4a0c3e)
*   [Xiaohongshu](https://www.xiaohongshu.com/user/profile/668e7e15000000000303157d?xsec_token=ABl2-aqekpytY6A8TuxjrwnZskU-6BsMRE_ufQQaSAvjc%3D&xsec_source=pc_search)
*   [Zhihu](https://www.zhihu.com/org/dou-bao-da-mo-xing-tuan-dui/)

---

<div align="center">
  We are hiring!  Contact <a href="mailto:haibin.lin@bytedance.com">haibin.lin@bytedance.com</a> for internship/FTE opportunities in RL for agents.
</div>