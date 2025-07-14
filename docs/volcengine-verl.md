<div align="center">
  <img src="https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216" alt="Seed Logo" width="150">
</div>

<h1 align="center">verl: Revolutionizing LLM Training with Reinforcement Learning</h1>

verl is an open-source, production-ready reinforcement learning (RL) training library for Large Language Models (LLMs), offering a flexible and efficient solution for advanced LLM training.  **[Explore verl on GitHub](https://github.com/volcengine/verl)** to supercharge your LLM development!

<div align="center">
  <a href="https://deepwiki.com/volcengine/verl"><img src="https://devin.ai/assets/deepwiki-badge.png" alt="Ask DeepWiki.com" style="height:20px;"></a>
  [![GitHub Repo stars](https://img.shields.io/github/stars/volcengine/verl)](https://github.com/volcengine/verl/stargazers)
  [![Twitter](https://img.shields.io/twitter/follow/verl_project)](https://twitter.com/verl_project)
  <a href="https://join.slack.com/t/verlgroup/shared_invite/zt-2w5p9o4c3-yy0x2Q56s_VlGLsJ93A6vA"><img src="https://img.shields.io/badge/Slack-verl-blueviolet?logo=slack&amp"></a>
  <a href="https://arxiv.org/pdf/2409.19256"><img src="https://img.shields.io/static/v1?label=EuroSys&message=Paper&color=red"></a>
  [![Documentation](https://img.shields.io/badge/documentation-blue)](https://verl.readthedocs.io/en/latest/)
  <a href="https://raw.githubusercontent.com/eric-haibin-lin/verl-community/refs/heads/main/WeChat.JPG"><img src="https://img.shields.io/badge/微信-green?logo=wechat&amp"></a>
</div>

## Key Features

*   **Flexible and Extensible**: Easily integrate diverse RL algorithms and build custom dataflows (e.g., GRPO, PPO) with a hybrid-controller programming model.
*   **Seamless Integration**: Works seamlessly with existing LLM infrastructure and frameworks, including FSDP, Megatron-LM, vLLM, and SGLang.
*   **Optimized Resource Utilization**: Offers flexible device mapping and efficient model placement across GPUs for scalability.
*   **Hugging Face Compatibility**:  Ready-to-use with popular Hugging Face models.
*   **State-of-the-Art Performance**: Achieve SOTA throughput with optimized LLM training and inference engine integrations.
*   **3D-HybridEngine**:  Efficient actor model resharding to eliminate memory redundancy and reduce communication overhead.
*   **Diverse RL Algorithms**:  Supports PPO, GRPO, ReMax, REINFORCE++, RLOO, PRIME, DAPO, DrGRPO, and more.
*   **Advanced Features**: Multi-GPU LoRA RL, experiment tracking (wandb, swanlab, mlflow, tensorboard), flash attention 2, sequence packing, and more.

## Recent News & Updates

*   **[ICML Vancouver Meetup]**: Join us at ICML Vancouver on July 16th! ([join us](https://lu.ma/0ek2nyao))
*   **[AWS AI Hours Singapore]**: verl keynote on 7/8. ([Agenda](https://pages.awscloud.com/aws-ai-hours-sg.html#agenda))
*   **[Agent for SWE meetup]**: verl & verl-agent project updates on 7/11. ([Meetup](https://lu.ma/e498qhsi))
*   **[verl with DeepSeek and Qwen3]:** Now supports large MoE models such as DeepSeek-671b and Qwen3-236b.
*   **[PyTorch Day China]:** verl team will provide project updates on June 7th.
*   **[Seed-Thinking-v1.5]:** Release of Seed-Thinking-v1.5 tech report, achieving impressive results.
*   **[DAPO]:** Open-sourced DAPO, a SOTA RL algorithm trained with verl.
*   **[More Updates]:** Ongoing paper releases, tutorials, and meetups. See the [original README](https://github.com/volcengine/verl) for a complete list.

## Getting Started

*   **[Documentation](https://verl.readthedocs.io/en/latest/index.html)**
*   **Quickstart**: [Installation](https://verl.readthedocs.io/en/latest/start/install.html), [Quickstart](https://verl.readthedocs.io/en/latest/start/quickstart.html)
*   **Guides**: [Programming Guide](https://verl.readthedocs.io/en/latest/hybrid_flow.html), [PPO in verl](https://verl.readthedocs.io/en/latest/algo/ppo.html), [GRPO in verl](https://verl.readthedocs.io/en/latest/algo/grpo.html)

## Performance Tuning and Advanced Usage

*   **[Performance Tuning Guide](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html)**
*   **Advanced Usage**:  See the original README for information on [Adding Models](https://verl.readthedocs.io/en/latest/advance/fsdp_extension.html), [Multi-turn Rollout](https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html), and more.

## Citation

```bibtex
@article{sheng2024hybridflow,
  title   = {HybridFlow: A Flexible and Efficient RLHF Framework},
  author  = {Guangming Sheng and Chi Zhang and Zilingfeng Ye and Xibin Wu and Wang Zhang and Ru Zhang and Yanghua Peng and Haibin Lin and Chuan Wu},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2409.19256}
}
```

verl is a collaborative effort by ByteDance, Anyscale, LMSys.org, and the wider AI community.

## Awesome Projects Using verl

verl is used in many open source projects, please check out [recipe](recipe/README.md) for a more complete list.

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

## Contribute

See [contributions guide](CONTRIBUTING.md) for how to contribute to the project.

## About ByteDance Seed Team

verl is initiated by ByteDance Seed Team. For more information, visit the [ByteDance Seed Team website](https://team.doubao.com/).

---

We are hiring! Contact [haibin.lin@bytedance.com](mailto:haibin.lin@bytedance.com) if you are interested in internship or FTE opportunities in RL for agents.