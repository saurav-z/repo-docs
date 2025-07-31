<!-- Improved README.md -->
<div align="center">
  <!-- Title and Badges -->
  <h1>verl: Production-Ready Reinforcement Learning for LLMs</h1>
  <p>verl is a flexible and efficient RL training library for large language models (LLMs), enabling state-of-the-art performance.</p>
  
  <br>
  <br>

  <!-- Badges -->
  <a href="https://github.com/volcengine/verl">
    <img src="https://img.shields.io/github/stars/volcengine/verl?style=social" alt="GitHub Stars">
  </a>
  <a href="https://twitter.com/verl_project">
    <img src="https://img.shields.io/twitter/follow/verl_project?style=social" alt="Follow on Twitter">
  </a>
  <a href="https://join.slack.com/t/verlgroup/shared_invite/zt-2w5p9o4c3-yy0x2Q56s_VlGLsJ93A6vA">
    <img src="https://img.shields.io/badge/Slack-verl-blueviolet?logo=slack&amp" alt="Join Slack">
  </a>
  <a href="https://verl.readthedocs.io/en/latest/">
    <img src="https://img.shields.io/badge/Documentation-blue" alt="Documentation">
  </a>
  <a href="https://arxiv.org/pdf/2409.19256">
      <img src="https://img.shields.io/static/v1?label=EuroSys&message=Paper&color=red" alt="EuroSys Paper">
  </a>
  <a href="https://raw.githubusercontent.com/eric-haibin-lin/verl-community/refs/heads/main/WeChat.JPG">
      <img src="https://img.shields.io/badge/微信-green?logo=wechat&amp" alt="WeChat">
  </a>
  <a href="https://deepwiki.com/volcengine/verl">
        <img src="https://devin.ai/assets/deepwiki-badge.png" alt="Ask DeepWiki.com" style="height:20px;">
  </a>
</div>

<div align="center">
  <img src="https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216" alt="Seed Logo" width="100">
</div>

## Key Features

*   **Flexible and Extensible:** Easily integrate and extend diverse RL algorithms, like PPO and GRPO, with a modular hybrid-controller programming model.
*   **Seamless Integration:** Compatible with existing LLM infrastructure, including FSDP, Megatron-LM, vLLM, and SGLang.
*   **Efficient Device Mapping:** Flexible GPU allocation for efficient resource utilization and scalability.
*   **Hugging Face Integration:** Ready-to-use with popular Hugging Face models.
*   **State-of-the-Art Throughput:** Optimized for SOTA LLM training and inference, achieving high RL throughput.
*   **3D-HybridEngine**: Efficient actor model resharding, eliminating memory redundancy.

## Getting Started

verl is a powerful library that empowers you to train and deploy state-of-the-art RL models for LLMs.  Explore the [verl documentation](https://verl.readthedocs.io/en/latest/index.html) to get started.

*   **Installation:** [Installation Guide](https://verl.readthedocs.io/en/latest/start/install.html)
*   **Quickstart:** [Quickstart Guide](https://verl.readthedocs.io/en/latest/start/quickstart.html)
*   **Programming Guide:** [Programming Guide](https://verl.readthedocs.io/en/latest/hybrid_flow.html)
*   **PPO Example:** [PPO in verl](https://verl.readthedocs.io/en/latest/algo/ppo.html)
*   **GRPO Example:** [GRPO in verl](https://verl.readthedocs.io/en/latest/algo/grpo.html)

For code explanation and advance usage (extension):

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

##  Performance Tuning Guide
The performance is essential for on-policy RL algorithm. We have written a detailed [performance tuning guide](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html) to help you optimize performance.

## News and Updates

*   **[2025/07]** ReTool Recipe Open Sourced and Verl Meetup at ICML Vancouver.
*   **[2025/06]** verl enables large MoE models like DeepSeek-671b and Qwen3-236b.
*   **[2025/04]** DAPO is the open-sourced SOTA RL algorithm that achieves 50 points on AIME 2024.
*   **[2025/04]**  VAPO paper covers our latest RL method for reasoning models.
*   **[2025/05]** [PF-PPO](https://arxiv.org/abs/2409.06957), accepted to ICML 2025, is now supported in verl!
*   **[2025/03]** verl v0.3.0.post1 is released.

For more information, check the [full list of updates](https://github.com/volcengine/verl#news).

## Ecosystem and Community

### Blogs from the community
- [When Reasoning Models Break Tokenization: The Hidden Complexity of Multiturn Training](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/fast_tokenization/multiturn_tokenization_and_masking.md)
- [verl deployment on AWS SageMaker](https://medium.com/@kaige.yang0110/run-verl-on-sagemaker-using-4x8-l40s-gpus-8e6d5c3c61d3)
- [verl x SGLang Multi-turn Code Walkthrough](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/code-walk-through/readme_EN.md)
- [Optimizing SGLang Memory Usage in verl](https://hebiao064.github.io/rl-memory-management)
- [SGLang, verl, OpenBMB and Tsinghua University: Pioneering End-to-End Multi-Turn RLHF](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/verl-multiturn-rollout-Release.md)
- [Reinforcement Learning from Human Feedback on AMD GPUs with verl and ROCm Integration](https://rocm.blogs.amd.com/artificial-intelligence/verl-large-scale/README.html)
- [veMLP x verl ：玩转强化学习训练](https://mp.weixin.qq.com/s/7nbqxk4knMGd-hQE9ls2tA)
- [使用 verl 进行 GRPO 分布式强化学习训练最佳实践](https://www.volcengine.com/docs/6459/1463942)
- [HybridFlow verl 原文浅析](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/readme.md)
- [最高提升 20 倍吞吐量！豆包大模型团队发布全新 RLHF 框架，现已开源！](https://team.doubao.com/en/blog/%E6%9C%80%E9%AB%98%E6%8F%90%E5%8D%8720%E5%80%8D%E5%90%9E%E5%90%90%E9%87%8F-%E8%B1%86%E5%8C%85%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%9B%A2%E9%98%9F%E5%8F%91%E5%B8%83%E5%85%A8%E6%96%B0-rlhf-%E6%A1%86%E6%9E%B6-%E7%8E%B0%E5%B7%B2%E5%BC%80%E6%BA%90)

### Awesome work using verl

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

### Community

verl is an open-source project initiated by the ByteDance Seed team and supported by a vibrant community.

*   **[GitHub Repository](https://github.com/volcengine/verl):** Find the source code, report issues, and contribute.
*   **[Slack](https://join.slack.com/t/verlgroup/shared_invite/zt-2w5p9o4c3-yy0x2Q56s_VlGLsJ93A6vA):** Connect with other users and developers.
*   **[Twitter](https://twitter.com/verl_project):** Stay updated on the latest news and announcements.

## Technical Details

*   **Modular Design:** Built on a flexible architecture for easy customization and extension.
*   **Optimized Performance:** Designed for high throughput and efficient training.
*   **Integration with Existing Frameworks:** Seamlessly integrates with popular LLM tools and libraries.

### Upgrades
*   **vLLM:** verl now supports vLLM>=0.8.2 when using FSDP as the training backend. Please refer to [this document](https://github.com/volcengine/verl/blob/main/docs/README_vllm0.8.md) for the installation guide and more information. Please avoid vllm 0.7.x, which contains bugs that may lead to OOMs and unexpected errors.
*   **SGLang:** SGLang is fully supported with verl, and SGLang RL Group is working extensively on building unique features, including multi-turn agentic RL, VLM RLHF, server-based RL, and partial rollout. Please refer to [this document](https://verl.readthedocs.io/en/latest/workers/sglang_worker.html) for the installation guide and more information.
*   **FSDP2:** verl is fully embracing FSDP2! FSDP2 is recommended by torch distributed team, providing better throughput and memory usage, and is composible with other features (e.g. torch.compile). To enable FSDP2, simply use verl main and set the following options:
```
actor_rollout_ref.ref.strategy=fsdp2
actor_rollout_ref.actor.strategy=fsdp2
critic.strategy=fsdp2 
reward_model.strategy=fsdp2 
```
Furthermore, FSDP2 cpu offloading is compatible with gradient accumulation. You can turn it on to save memory with `actor_rollout_ref.actor.fsdp_config.offload_policy=True`. For more details, see https://github.com/volcengine/verl/pull/1026

*   **AMD Support:** verl supports FSDP as the training engine (Megatron support coming soon) and both integrates with vLLM and SGLang as inference engines. Please refer to [this document](https://github.com/volcengine/verl/blob/main/docs/amd_tutorial/amd_build_dockerfile_page.rst) for the installation guide and more information, and [this document](https://github.com/volcengine/verl/blob/main/docs/amd_tutorial/amd_vllm_page.rst) for the vLLM performance tuning for ROCm.

## Citation and Acknowledgements

If you use verl in your research, please cite the following paper:

*   [HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256v2)
```bibtex
@article{sheng2024hybridflow,
  title   = {HybridFlow: A Flexible and Efficient RLHF Framework},
  author  = {Guangming Sheng and Chi Zhang and Zilingfeng Ye and Xibin Wu and Wang Zhang and Ru Zhang and Yanghua Peng and Haibin Lin and Chuan Wu},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2409.19256}
}
```

verl is inspired by Nemo-Aligner, Deepspeed-chat and OpenRLHF. The project is adopted and contributed by Bytedance, Anyscale, LMSys.org, [Alibaba Qwen team](https://github.com/QwenLM/), Shanghai AI Lab, Tsinghua University, UC Berkeley, UCLA, UIUC, University of Hong Kong, ke.com, [All Hands AI](https://www.all-hands.dev/), [ModelBest](http://modelbest.cn/), JD AI Lab, Microsoft Research, [StepFun](https://www.stepfun.com/), Amazon, LinkedIn, Meituan, [Camel-AI](https://www.camel-ai.org/), [OpenManus](https://github.com/OpenManus), Xiaomi, NVIDIA research, [Baichuan](https://www.baichuan-ai.com/home), [RedNote](https://www.xiaohongshu.com/), [SwissAI](https://www.swiss-ai.org/), [Moonshot AI (Kimi)](https://www.moonshot-ai.com/), Baidu, Snowflake, Skywork.ai, JetBrains, [IceSword Lab](https://www.iceswordlab.com), and many more.

## Contributing

We welcome contributions!  See the [contributions guide](CONTRIBUTING.md) for details.

## About [ByteDance Seed Team](https://team.doubao.com/)

The ByteDance Seed Team is dedicated to advancing AI foundation models and is committed to making significant contributions to the field. Learn more about us on our [website](https://team.doubao.com/).

<div style="display: flex; align-items: center;">
  <a href="https://team.doubao.com/">
    <img src="https://img.shields.io/badge/Website-%231e37ff?style=for-the-badge&logo=bytedance&logoColor=white"></a>
  <a href="https://github.com/user-attachments/assets/469535a8-42f2-4797-acdf-4f7a1d4a0c3e">
    <img src="https://img.shields.io/badge/WeChat-07C160?style=for-the-badge&logo=wechat&logoColor=white"></a>
  <a href="https://www.xiaohongshu.com/user/profile/668e7e15000000000303157d?xsec_token=ABl2-aqekpytY6A8TuxjrwnZskU-6BsMRE_ufQQaSAvjc%3D&xsec_source=pc_search">
    <img src="https://img.shields.io/badge/Xiaohongshu-%23FF2442?style=for-the-badge&logo=xiaohongshu&logoColor=white"></a>
  <a href="https://www.zhihu.com/org/dou-bao-da-mo-xing-tuan-dui/">
    <img src="https://img.shields.io/badge/zhihu-%230084FF?style=for-the-badge&logo=zhihu&logoColor=white"></a>
  <p>
    We are HIRING! Send us an [email](mailto:haibin.lin@bytedance.com) if you are interested in internship/FTE opportunities in RL for agents.
  </p>
</div>
```

Key improvements and SEO considerations:

*   **Concise Title:**  The main title is clear and keyword-rich: "verl: Production-Ready Reinforcement Learning for LLMs."
*   **One-Sentence Hook:** Immediately establishes the library's purpose and value.
*   **Keyword Optimization:**  Uses relevant keywords like "Reinforcement Learning," "LLMs," "RLHF", "Training," "Large Language Models" throughout the text, and within the descriptions of features.
*   **Clear Headings and Structure:**  Organized with well-defined sections for easy navigation and readability (Getting Started, Key Features, News & Updates, etc.).
*   **Bullet Points:** Uses bullet points to highlight key features, which is great for skimming and SEO.
*   **Emphasis on Benefits:** Focuses on the *benefits* of using verl (flexible, efficient, production-ready) rather than just listing features.
*   **Links to Documentation and Examples:** Directs users to the resources they need.
*   **Community Section:** Encourages engagement and provides links to the community.
*   **Call to Action:** Encourages users to reach out for opportunities
*   **Clear and Concise Language:** Uses simple and direct language, making it accessible to a wider audience.
*   **Updated for Relevance:** Includes the latest news and features, keeping the README current.
*   **Citation:** Includes the citation and acknowledgements
*   **Images:** The readme includes images and badges for better engagement.