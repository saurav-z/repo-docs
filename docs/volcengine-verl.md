# verl: Revolutionizing Reinforcement Learning for LLMs

verl is an open-source RL training library from ByteDance Seed team, designed to make training and deploying large language models (LLMs) easier and more efficient than ever before. [Explore verl on GitHub](https://github.com/volcengine/verl).

**Key Features:**

*   🚀 **Flexible and Extensible:** Easily integrate and extend diverse RL algorithms with verl's hybrid-controller programming model. Build RL dataflows like GRPO and PPO in just a few lines of code.
*   🔗 **Seamless Integration:** Effortlessly integrates with existing LLM frameworks, including FSDP, Megatron-LM, vLLM, and SGLang, through modular APIs.
*   ⚙️ **Flexible Device Mapping:** Supports optimized model placement across GPUs for efficient resource utilization and scalability, regardless of cluster size.
*   ✅ **Ready-to-Use:** Comes with pre-built integrations for popular Hugging Face models, simplifying your workflow.
*   ⚡ **High Performance:** Benefit from state-of-the-art LLM training and inference integrations, alongside superior RL throughput.
*   ✨ **3D-HybridEngine:** Resharding of actor models eliminates memory redundancy and significantly lowers communication overhead.

**News & Updates:**

*   **[2025/07]** Join us at the first verl meetup at ICML Vancouver on July 16th!
*   **[2025/07]** verl keynote at AWS AI Hours Singapore and updates at the Agent for SWE meetup by LF AI & Data Singapore.
*   **[2025/06]** verl with Megatron backend enables large MoE models such as [DeepSeek-671b and Qwen3-236b](https://verl.readthedocs.io/en/latest/perf/dpsk.html).
*   **[2025/06]** verl team will provide latest project updates at [PyTorch Day China](https://www.lfasiallc.com/pytorch-day-china/) on June 7th.
*   **[2025/04]** Seed-Thinking-v1.5 tech report is released! Trained with verl, Seed-Thinking-v1.5 achieves 86.7 on AIME 2024, 55.0 on Codeforces and 77.3 on GPQA, demonstrating excellent reasoning abilities in STEM and coding.
*   **[2025/03]** [DAPO](https://dapo-sia.github.io/) is the open-sourced SOTA RL algorithm that achieves 50 points on AIME 2024 based on the Qwen2.5-32B pre-trained model, surpassing the previous SOTA achieved by DeepSeek's GRPO (DeepSeek-R1-Zero-Qwen-32B). DAPO's training is fully powered by verl and the reproduction code is available in `recipe/dapo` now.

<!-- Add more news updates here as needed -->

**Getting Started:**

*   [Documentation](https://verl.readthedocs.io/en/latest/index.html)
*   [Installation Guide](https://verl.readthedocs.io/en/latest/start/install.html)
*   [Quickstart Guide](https://verl.readthedocs.io/en/latest/start/quickstart.html)
*   [Programming Guide](https://verl.readthedocs.io/en/latest/hybrid_flow.html)
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

**Blogs from the community:**
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

**Performance Tuning Guide:**

*   [Performance Tuning Guide](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html)

**Upgrading to vLLM >= v0.8.2:**

*   [vLLM >= 0.8.2 Installation Guide](https://github.com/volcengine/verl/blob/main/docs/README_vllm0.8.md)

**Using the Latest SGLang:**

*   [SGLang Installation Guide](https://verl.readthedocs.io/en/latest/workers/sglang_worker.html)

**Upgrading to FSDP2:**
```
actor_rollout_ref.ref.strategy=fsdp2
actor_rollout_ref.actor.strategy=fsdp2
critic.strategy=fsdp2 
reward_model.strategy=fsdp2 
```

For more details, see https://github.com/volcengine/verl/pull/1026

**AMD Support (ROCm Kernel):**

*   [AMD Installation Guide](https://github.com/volcengine/verl/blob/main/docs/amd_tutorial/amd_build_dockerfile_page.rst)
*   [vLLM Performance Tuning for ROCm](https://github.com/volcengine/verl/blob/main/docs/amd_tutorial/amd_vllm_page.rst)

**Citation and Acknowledgements:**

If you use verl, please cite the following papers:

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

verl is inspired by Nemo-Aligner, Deepspeed-chat, and OpenRLHF, and is supported by Bytedance, Anyscale, LMSys.org, [Alibaba Qwen team](https://github.com/QwenLM/), Shanghai AI Lab, Tsinghua University, UC Berkeley, UCLA, UIUC, University of Hong Kong, ke.com, [All Hands AI](https://www.all-hands.dev/), [ModelBest](http://modelbest.cn/), JD AI Lab, Microsoft Research, [StepFun](https://www.stepfun.com/), Amazon, LinkedIn, Meituan, [Camel-AI](https://www.camel-ai.org/), [OpenManus](https://github.com/OpenManus), Xiaomi, NVIDIA research, [Baichuan](https://www.baichuan-ai.com/home), [RedNote](https://www.xiaohongshu.com/), [SwissAI](https://www.swiss-ai.org/), [Moonshot AI (Kimi)](https://www.moonshot-ai.com/), Baidu, Snowflake, Skywork.ai, JetBrains, [IceSword Lab](https://www.iceswordlab.com), and many more.

**Awesome work using verl:**
*   [TinyZero](https://github.com/Jiayi-Pan/TinyZero)
*   [SkyThought](https://github.com/NovaSky-AI/SkyThought)
*   [simpleRL-reason](https://github.com/hkust-nlp/simpleRL-reason)
*   [Easy-R1](https://github.com/hiyouga/EasyR1)
*   [OpenManus-RL](https://github.com/OpenManus/OpenManus-RL)
*   [rllm](https://github.com/agentica-project/rllm)
*   [RAGEN](https://github.com/ZihanWang314/ragen)
*   [Search-R1](https://github.com/PeterGriffinJin/Search-R1)
*   [ReSearch](https://github.com/Agent-RL/ReSearch)
*   [Skywork-OR1](https://github.com/SkyworkAI/Skywork-OR1)
*   [ToRL](https://github.com/GAIR-NLP/ToRL)
*   [Absolute Zero Reasoner](https://github.com/LeapLabTHU/Absolute-Zero-Reasoner)
*   [verl-agent](https://github.com/langfengQ/verl-agent)
*   [RL-Factory](https://github.com/Simple-Efficient/RL-Factory)
*   [ReTool](https://retool-rl.github.io/)
*   [verl-tool](https://github.com/TIGER-AI-Lab/verl-tool)
*   [PRIME](https://github.com/PRIME-RL/PRIME)
*   [MemAgent](https://github.com/BytedTsinghua-SIA/MemAgent)
*   [POLARIS](https://github.com/ChenxinAn-fdu/POLARIS)
*   [GUI-R1](https://github.com/ritzz-ai/GUI-R1)
*   [DeepRetrieval](https://github.com/pat-jj/DeepRetrieval)
*   [Code-R1](https://github.com/ganler/code-r1)
*   [DeepResearcher](https://github.com/GAIR-NLP/DeepResearcher)
*   [VAGEN](https://github.com/RAGEN-AI/VAGEN)
*   [RM-R1](https://arxiv.org/abs/2505.02387)
*   [LUFFY](https://arxiv.org/pdf/2504.14945)
*   [DeepMath](https://github.com/zwhe99/DeepMath)
*   [Entropy Mechanism of RL](https://github.com/PRIME-RL/Entropy-Mechanism-of-RL)
*   [LLaSA-TTS-GRPO](https://github.com/channel-io/ch-tts-llasa-rl-grpo)
*   [PF-PPO](https://arxiv.org/abs/2409.06957)
*   [RACRO](https://github.com/gyhdog99/RACRO2)
*   [Agent Lightning](https://github.com/microsoft/agent-lightning)

See [recipe](recipe/README.md) for more projects.

**Contribution Guide:**

*   See [contributions guide](CONTRIBUTING.md)

**About [ByteDance Seed Team](https://team.doubao.com/)**:

ByteDance Seed Team is focused on building advanced AI foundation models.

We are HIRING! Email [haibin.lin@bytedance.com](mailto:haibin.lin@bytedance.com) if you're interested in internship/FTE opportunities.