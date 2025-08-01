# verl: Production-Ready Reinforcement Learning for LLMs

**Supercharge your LLMs with verl, a flexible and efficient RL training library, enabling cutting-edge performance and easy integration.**  ([Back to Original Repo](https://github.com/volcengine/verl))

<div align="center">
    
[![GitHub Repo stars](https://img.shields.io/github/stars/volcengine/verl)](https://github.com/volcengine/verl/stargazers)
[![Twitter](https://img.shields.io/twitter/follow/verl_project)](https://twitter.com/verl_project)
<a href="https://join.slack.com/t/verlgroup/shared_invite/zt-2w5p9o4c3-yy0x2Q56s_VlGLsJ93A6vA"><img src="https://img.shields.io/badge/Slack-verl-blueviolet?logo=slack&amp"></a>
[![Documentation](https://img.shields.io/badge/documentation-blue)](https://verl.readthedocs.io/en/latest/)
<a href="https://raw.githubusercontent.com/eric-haibin-lin/verl-community/refs/heads/main/WeChat.JPG"><img src="https://img.shields.io/badge/微信-green?logo=wechat&amp"></a>

</div>

verl is a powerful RL training library initiated by the ByteDance Seed team and supported by a vibrant community, designed to streamline and accelerate the training of large language models (LLMs).  It is the open-source version of the [HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256v2) paper.

**Key Features:**

*   **Flexible RL Algorithms:** Easily implement diverse RL algorithms like PPO, GRPO, and more with the hybrid-controller programming model.
*   **Seamless LLM Integration:**  Integrates effortlessly with existing LLM frameworks, including FSDP, Megatron-LM, vLLM, and SGLang, through modular APIs.
*   **Efficient Resource Utilization:** Supports flexible device mapping for optimal GPU utilization and scalability.
*   **Hugging Face Compatibility:** Ready-to-use with popular Hugging Face models.
*   **High Performance:** Achieve state-of-the-art LLM training throughput.
*   **3D-HybridEngine:** Eliminates memory redundancy, reducing communication overhead for efficient training.

## What's New
*   **[2025/07]** The [ReTool](https://arxiv.org/pdf/2504.11536) recipe is fully open sourced. [Blog](https://www.notion.so/verl-reTool-recipe-Using-multi-round-conversations-and-code-sandboxing-to-improve-the-math-of-large-23a8b5b7feba80b386b2e5b5e3c1cde0)
*   **[2025/07]** The first verl meetup will be held at ICML Vancouver on July 16th! Please [join us](https://lu.ma/0ek2nyao) if you are at ICML! (onsite only)
*   **[2025/07]** verl keynote at [AWS AI Hours Singapore](https://pages.awscloud.com/aws-ai-hours-sg.html#agenda) on 7/8, verl & verl-agent project updates at [Agent for SWE meetup](https://lu.ma/e498qhsi) by LF AI & Data Singapore on 7/11.
*   **[2025/06]** verl with Megatron backend enables large MoE models such as [DeepSeek-671b and Qwen3-236b](https://verl.readthedocs.io/en/latest/perf/dpsk.html).
*   **[2025/06]** verl team will provide latest project updates at [PyTorch Day China](https://www.lfasiallc.com/pytorch-day-china/) on June 7th. Meet our dev team in Beijing!
*   **[2025/04]** [Seed-Thinking-v1.5](https://github.com/ByteDance-Seed/Seed-Thinking-v1.5/blob/main/seed-thinking-v1.5.pdf) tech report is released! Trained with verl, Seed-Thinking-v1.5 achieves 86.7 on AIME 2024, 55.0 on Codeforces and 77.3 on GPQA, demonstrating excellent reasoning abilities in STEM and coding. Beyond reasoning tasks, the method demonstrates notable generalization across diverse domains.
*   **[2025/03]** [DAPO](https://dapo-sia.github.io/) is the open-sourced SOTA RL algorithm that achieves 50 points on AIME 2024 based on the Qwen2.5-32B pre-trained model, surpassing the previous SOTA achieved by DeepSeek's GRPO (DeepSeek-R1-Zero-Qwen-32B). DAPO's training is fully powered by verl and the reproduction code is available in `recipe/dapo` now.
<details><summary> more...</summary>
<ul>
  <li> [2025/04] [VAPO](https://arxiv.org/pdf/2504.05118) (value-based augmented PPO) paper covers our latest RL method for reasoning models. Trained from Qwen-32B-base model, VAPO achieves 60.4 on AIME 2024, outperforming DAPO-32B.</li>
  <li>[2025/05] [PF-PPO](https://arxiv.org/abs/2409.06957), accepted to ICML 2025, is now supported in verl! PF-PPO enhances policy learning efficiency and robustness by filtering potentially noisy reward signals and reusing high-quality experiences via a replay buffer.</li>
  <li>[2025/04] We will give a tutorial about latest post-training techniques and programming guide for verl at [ICLR 2025 Expo](https://iclr.cc/virtual/2025/calendar?filter_events=Expo+Talk+Panel&filter_rooms=), [SCI-FM workshop](https://open-foundation-model.github.io/) and [LMSys afterparty](https://lu.ma/d23nyynm). Talk materials available [here](https://github.com/eric-haibin-lin/verl-community/tree/main/iclr25). </li>
  <li>[2025/03] verl v0.3.0.post1 is released! See [release note](https://github.com/volcengine/verl/releases/) for details. It achieves [~1.4x speedup](https://tongyx361.github.io/blogs/posts/verl-intro/#/verl-flexible-and-efficient-rl-for-llms) compared to prev versions.</li>
  <li>[2025/05] verl will be presented at [A2M Shanghai](https://a2m.msup.com.cn/home/?aid=4488&city=shanghai) on 5/16 - 5/17.</li>
  <li>[2025/05] verl will be presented at [GOSIM x PyTorch Day 2025](https://paris2025.gosim.org/). See you in Paris! </li>
  <li>[2025/03] We introduced the programming model of verl at the [vLLM Beijing Meetup](https://mp.weixin.qq.com/s/n77GibL2corAtQHtVEAzfg) and [verl intro and updates](https://github.com/eric-haibin-lin/verl-community/blob/main/slides/verl-lmsys-meetup.pdf) at the [SGLang-LMSYS Org Meetup](https://lu.ma/ntjrr7ig) in Sunnyvale mid-March.</li>
  <li>[2025/03] We will present verl(HybridFlow) at EuroSys 2025. See you in Rotterdam!</li>
  <li>[2025/02] verl v0.2.0.post2 is released!</li>
  <li>[2025/02] We presented verl in the <a href="https://lu.ma/ji7atxux">Bytedance/NVIDIA/Anyscale Ray Meetup</a>. See you in San Jose!</li>
  <li>[2025/01] [Doubao-1.5-pro](https://team.doubao.com/zh/special/doubao_1_5_pro) is released with SOTA-level performance on LLM & VLM. The RL scaling preview model is trained using verl, reaching OpenAI O1-level performance on math benchmarks (70.0 pass@1 on AIME).</li>
  <li>[2024/12] verl is presented at Ray Forward 2024. Slides available <a href="https://github.com/eric-haibin-lin/verl-community/blob/main/slides/Ray_Forward_2024_%E5%B7%AB%E9%94%A1%E6%96%8C.pdf">here</a></li>
  <li>[2024/12] The team presented <a href="https://neurips.cc/Expo/Conferences/2024/workshop/100677">Post-training LLMs: From Algorithms to Infrastructure</a> at NeurIPS 2024. <a href="https://github.com/eric-haibin-lin/verl-data/tree/neurips">Slides</a> and <a href="https://neurips.cc/Expo/Conferences/2024/workshop/100677">video</a> available.</li>
  <li>[2024/10] verl is presented at Ray Summit. <a href="https://www.youtube.com/watch?v=MrhMcXkXvJU&list=PLzTswPQNepXntmT8jr9WaNfqQ60QwW7-U&index=37">Youtube video</a> available.</li>
  <li>[2024/08] HybridFlow (verl) is accepted to EuroSys 2025.</li>
</ul>   
</details>

## Getting Started
Find detailed information on how to get started with verl.

*   **[Documentation](https://verl.readthedocs.io/en/latest/index.html)**
*   **Quickstart:**
    *   [Installation](https://verl.readthedocs.io/en/latest/start/install.html)
    *   [Quickstart](https://verl.readthedocs.io/en/latest/start/quickstart.html)
    *   [Programming Guide](https://verl.readthedocs.io/en/latest/hybrid_flow.html) & [Tech Talk](https://hcqnc.xetlk.com/sl/3vACOK) (in Chinese)
    *   [PPO in verl](https://verl.readthedocs.io/en/latest/algo/ppo.html)
    *   [GRPO in verl](https://verl.readthedocs.io/en/latest/algo/grpo.html)

*   **PPO Example:**
    *   [Prepare Data for Post-Training](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html)
    *   [Implement Reward Function for Dataset](https://verl.readthedocs.io/en/latest/preparation/reward_function.html)
    *   [PPO Example Architecture](https://verl.readthedocs.io/en/latest/examples/ppo_code_architecture.html)
    *   [Config Explanation](https://verl.readthedocs.io/en/latest/examples/config.html)

*   **Reproducible algorithm baselines:**
    *   [RL performance on coding, math](https://verl.readthedocs.io/en/latest/algo/baseline.html)

*   **Code Explanation and Advance Usage**
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

*   **Blogs from the community**
    *   [When Reasoning Models Break Tokenization: The Hidden Complexity of Multiturn Training](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/fast_tokenization/multiturn_tokenization_and_masking.md)
    *   [verl deployment on AWS SageMaker](https://medium.com/@kaige.yang0110/run-verl-on-sagemaker-using-4x8-l40s-gpus-8e6d5c3c61d3)
    *   [verl x SGLang Multi-turn Code Walkthrough](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/code-walk-through/readme_EN.md)
    *   [Optimizing SGLang Memory Usage in verl](https://hebiao064.github.io/rl-memory-management)
    *   [SGLang, verl, OpenBMB and Tsinghua University: Pioneering End-to-End Multi-Turn RLHF](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/verl-multiturn-rollout-Release.md)
    *   [Reinforcement Learning from Human Feedback on AMD GPUs with verl and ROCm Integration](https://rocm.blogs.amd.com/artificial-intelligence/verl-large-scale/README.html)
    *   [veMLP x verl ：玩转强化学习训练](https://mp.weixin.qq.com/s/7nbqxk4knMGd-hQE9ls2tA)
    *   [使用 verl 进行 GRPO 分布式强化学习训练最佳实践](https://www.volcengine.com/docs/6459/1463942)
    *   [HybridFlow verl 原文浅析](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/readme.md)
    *   [最高提升 20 倍吞吐量！豆包大模型团队发布全新 RLHF 框架，现已开源！](https://team.doubao.com/en/blog/%E6%9C%80%E9%AB%98%E6%8F%90%E5%8D%8720%E5%80%8D%E5%90%9E%E5%90%90%E9%87%8F-%E8%B1%86%E5%8C%85%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%9B%A2%E9%98%9F%E5%8F%91%E5%B8%83%E5%85%A8%E6%96%B0-rlhf-%E6%A1%86%E6%9E%B6-%E7%8E%B0%E5%B7%B2%E5%BC%80%E6%BA%90)

## Performance Tuning Guide

Optimize your RL training with our detailed [performance tuning guide](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html)

## Upgrade Considerations
*   **vLLM:** verl supports vLLM>=0.8.2.  Avoid vllm 0.7.x to prevent potential issues. See  [this document](https://github.com/volcengine/verl/blob/main/docs/README_vllm0.8.md) for the installation guide and more information.
*   **SGLang:**  Leverage the latest SGLang version for unique features, including multi-turn agentic RL. Please refer to [this document](https://verl.readthedocs.io/en/latest/workers/sglang_worker.html) for the installation guide and more information.
*   **FSDP2:** Enable FSDP2 for improved throughput and memory usage.  Set `actor_rollout_ref.ref.strategy=fsdp2`,  `actor_rollout_ref.actor.strategy=fsdp2`, `critic.strategy=fsdp2`, and `reward_model.strategy=fsdp2`. CPU offloading can be enabled with `actor_rollout_ref.actor.fsdp_config.offload_policy=True`. See https://github.com/volcengine/verl/pull/1026

## AMD Support (ROCm Kernel)
verl supports FSDP and integrates with vLLM and SGLang. Please refer to [this document](https://github.com/volcengine/verl/blob/main/docs/amd_tutorial/amd_build_dockerfile_page.rst) and [this document](https://github.com/volcengine/verl/blob/main/docs/amd_tutorial/amd_vllm_page.rst) for instructions.

## Citation and Acknowledgements
If you find verl valuable, please cite the following:

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

verl is inspired by Nemo-Aligner, Deepspeed-chat and OpenRLHF, and is adopted and contributed by Bytedance, Anyscale, LMSys.org, [Alibaba Qwen team](https://github.com/QwenLM/), Shanghai AI Lab, Tsinghua University, UC Berkeley, UCLA, UIUC, University of Hong Kong, ke.com, [All Hands AI](https://www.all-hands.dev/), [ModelBest](http://modelbest.cn/), JD AI Lab, Microsoft Research, [StepFun](https://www.stepfun.com/), Amazon, LinkedIn, Meituan, [Camel-AI](https://www.camel-ai.org/), [OpenManus](https://github.com/OpenManus), Xiaomi, NVIDIA research, [Baichuan](https://www.baichuan-ai.com/home), [RedNote](https://www.xiaohongshu.com/), [SwissAI](https://www.swiss-ai.org/), [Moonshot AI (Kimi)](https://www.moonshot-ai.com/), Baidu, Snowflake, Skywork.ai, JetBrains, [IceSword Lab](https://www.iceswordlab.com), and many more.

## Awesome Work using verl

*   [TinyZero](https://github.com/Jiayi-Pan/TinyZero): a reproduction of **DeepSeek R1 Zero** recipe for reasoning tasks ![GitHub Repo stars](https://img.shields.io/github/stars/Jiayi-Pan/TinyZero)
*   [SkyThought](https://github.com/NovaSky-AI/SkyThought): RL training for Sky-T1-7B by NovaSky AI team. ![GitHub Repo stars](https://img.shields.io/github/stars/NovaSky-AI/SkyThought)
*   [simpleRL-reason](https://github.com/hkust-nlp/simpleRL-reason): SimpleRL-Zoo: Investigating and Taming Zero Reinforcement Learning for Open Base Models in the Wild ![GitHub Repo stars](https://img.shields.io/github/stars/hkust-nlp/simpleRL-reason)
*   [Easy-R1](https://github.com/hiyouga/EasyR1): **Multi-modal** RL training framework ![GitHub Repo stars](https://img.shields.io/github/stars/hiyouga/EasyR1)
*   [OpenManus-RL](https://github.com/OpenManus/OpenManus-RL): LLM Agents RL tunning framework for multiple agent environments. ![GitHub Repo stars](https://img.shields.io/github/stars/OpenManus/OpenManus-RL)
*   [rllm](https://github.com/agentica-project/rllm): async RL training with [verl-pipeline](https://github.com/agentica-project/verl-pipeline) ![GitHub Repo stars](https://img.shields.io/github/stars/agentica-project/rllm)
*   [RAGEN](https://github.com/ZihanWang314/ragen): a general-purpose reasoning **agent** training framework ![GitHub Repo stars](https://img.shields.io/github/stars/ZihanWang314/ragen)
*   [Search-R1](https://github.com/PeterGriffinJin/Search-R1): RL with reasoning and **searching (tool-call)** interleaved LLMs ![GitHub Repo stars](https://img.shields.io/github/stars/PeterGriffinJin/Search-R1)
*   [ReSearch](https://github.com/Agent-RL/ReSearch): Learning to **Re**ason with **Search** for LLMs via Reinforcement Learning ![GitHub Repo stars](https://img.shields.io/github/stars/Agent-RL/ReSearch)
*   [Skywork-OR1](https://github.com/SkyworkAI/Skywork-OR1): Skywork open reaonser series ![GitHub Repo stars](https://img.shields.io/github/stars/SkyworkAI/Skywork-OR1)
*   [ToRL](https://github.com/GAIR-NLP/ToRL): Scaling tool-integrated RL ![GitHub Repo stars](https://img.shields.io/github/stars/GAIR-NLP/ToRL)
*   [Absolute Zero Reasoner](https://github.com/LeapLabTHU/Absolute-Zero-Reasoner): [A no human curated data self-play framework for reasoning](https://arxiv.org/abs/2505.03335) ![GitHub Repo stars](https://img.shields.io/github/stars/LeapLabTHU/Absolute-Zero-Reasoner)
*   [verl-agent](https://github.com/langfengQ/verl-agent): A scalable training framework for **long-horizon LLM/VLM agents**, along with a new algorithm **GiGPO** ![GitHub Repo stars](https://img.shields.io/github/stars/langfengQ/verl-agent)
*   [RL-Factory](https://github.com/Simple-Efficient/RL-Factory): An easy and efficient RL post-training framework for Agentic Learning ![GitHub Repo stars](https://img.shields.io/github/stars/Simple-Efficient/RL-Factory)
*   [ReTool](https://retool-rl.github.io/): ReTool: reinforcement learning for strategic tool use in LLMs. Code release is in progress...
*   [verl-tool](https://github.com/TIGER-AI-Lab/verl-tool): An unified and easy-to-extend tool-agent training framework based on verl![GitHub Repo stars](https://img.shields.io/github/stars/TIGER-AI-Lab/verl-tool)
*   [PRIME](https://github.com/PRIME-RL/PRIME): Process reinforcement through implicit rewards ![GitHub Repo stars](https://img.shields.io/github/stars/PRIME-RL/PRIME)
*   [MemAgent](https://github.com/BytedTsinghua-SIA/MemAgent): MemAgent: Reshaping Long-Context LLM with Multi-Conv RL based Memory Agent ![GitHub Repo stars](https://img.shields.io/github/stars/BytedTsinghua-SIA/MemAgent)
*   [POLARIS](https://github.com/ChenxinAn-fdu/POLARIS): A Post-training recipe for scaling RL on Advanced Reasoning models ![GitHub Repo stars](https://img.shields.io/github/stars/ChenxinAn-fdu/POLARIS)
*   [GUI-R1](https://github.com/ritzz-ai/GUI-R1): **GUI-R1**: A Generalist R1-style Vision-Language Action Model For **GUI Agents** ![GitHub Repo stars](https://img.shields.io/github/stars/ritzz-ai/GUI-R1)
*   [DeepRetrieval](https://github.com/pat-jj/DeepRetrieval): RL Training of **Search Agent** with **Search/Retrieval Outcome** ![GitHub Repo stars](https://img.shields.io/github/stars/pat-jj/DeepRetrieval)
*   [Code-R1](https://github.com/ganler/code-r1): Reproducing R1 for **Code** with Reliable Rewards ![GitHub Repo stars](https://img.shields.io/github/stars/ganler/code-r1)
*   [DeepResearcher](https://github.com/GAIR-NLP/DeepResearcher): Scaling deep research via reinforcement learning in real-world environments ![GitHub Repo stars](https://img.shields.io/github/stars/GAIR-NLP/DeepResearcher)
*   [VAGEN](https://github.com/RAGEN-AI/VAGEN): Training VLM agents with multi-turn reinforcement learning ![GitHub Repo stars](https://img.shields.io/github/stars/RAGEN-AI/VAGEN)
*   [RM-R1](https://arxiv.org/abs/2505.02387): RL training of reasoning reward models ![GitHub Repo stars](https://img.shields.io/github/stars/RM-R1-UIUC/RM-R1)
*   [LUFFY](https://arxiv.org/pdf/2504.14945): Learning to Reason under Off-Policy Guidance![GitHub Repo stars](https://img.shields.io/github/stars/ElliottYan/LUFFY)
*   [DeepMath](https://github.com/zwhe99/DeepMath): DeepMath-103K data and series models for math reasoning![GitHub Repo stars](https://img.shields.io/github/stars/zwhe99/DeepMath)
*   [Entropy Mechanism of RL](https://github.com/PRIME-RL/Entropy-Mechanism-of-RL): The Entropy Mechanism of Reinforcement Learning for Large Language Model Reasoning![GitHub Repo stars](https://img.shields.io/github/stars/PRIME-RL/Entropy-Mechanism-of-RL)
*   [LLaSA-TTS-GRPO](https://github.com/channel-io/ch-tts-llasa-rl-grpo): TTS fine-tuning with GRPO optimization based on LLASA models ![GitHub Repo stars](https://img.shields.io/github/stars/channel-io/ch-tts-llasa-rl-grpo)
*   [PF-PPO](https://arxiv.org/abs/2409.06957): Policy Filtration for PPO based on the reliability of reward signals for more efficient and robust RLHF.
*   [RACRO](https://github.com/gyhdog99/RACRO2): Build multi-modal reasoning models via decoupling it into query-conditioned captioning and text-only reasoning ![GitHub Repo stars](https://img.shields.io/github/stars/gyhdog99/RACRO2)
*   [Agent Lightning](https://github.com/microsoft/agent-lightning): A flexible and extensible framework that enables seamless agent optimization for any existing agent framework. ![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/agent-lightning)

and many more awesome work listed in [recipe](recipe/README.md).

## Contribution Guide

See [contributions guide](CONTRIBUTING.md)

## About [ByteDance Seed Team](https://team.doubao.com/)

The ByteDance Seed Team is a world-class research team dedicated to crafting advanced AI foundation models. 
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

We are HIRING! Send us an [email](mailto:haibin.lin@bytedance.com) if you are interested in internship/FTE opportunities in RL for agents.