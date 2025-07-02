<div align="center">
    <h1>verl: Unleash the Power of Reinforcement Learning for LLMs</h1>
    <p>verl is your gateway to efficient and flexible reinforcement learning for Large Language Models, empowering you to train and optimize LLMs for a wide range of applications.</p>
</div>

<div align="center">
  <a href="https://github.com/volcengine/verl">
    <img src="https://img.shields.io/github/stars/volcengine/verl?style=social" alt="GitHub stars">
  </a>
  <a href="https://twitter.com/verl_project">
    <img src="https://img.shields.io/twitter/follow/verl_project?style=social" alt="Follow on Twitter">
  </a>
  <a href="https://join.slack.com/t/verlgroup/shared_invite/zt-2w5p9o4c3-yy0x2Q56s_VlGLsJ93A6vA">
    <img src="https://img.shields.io/badge/Slack-verl-blueviolet?logo=slack&amp" alt="Join Slack">
  </a>
  <a href="https://verl.readthedocs.io/en/latest/">
    <img src="https://img.shields.io/badge/documentation-blue" alt="Documentation">
  </a>
  <a href="https://arxiv.org/pdf/2409.19256">
    <img src="https://img.shields.io/static/v1?label=EuroSys&message=Paper&color=red" alt="EuroSys Paper">
  </a>
  <a href="https://deepwiki.com/volcengine/verl">
    <img src="https://devin.ai/assets/deepwiki-badge.png" alt="Ask DeepWiki.com" height="20"/>
  </a>
  <a href="https://raw.githubusercontent.com/eric-haibin-lin/verl-community/refs/heads/main/WeChat.JPG">
    <img src="https://img.shields.io/badge/微信-green?logo=wechat&amp" alt="WeChat">
  </a>
</div>

<div align="center">
  <img src="https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216" alt="Seed Logo" width="100">
</div>

## Key Features

*   **Flexible RL Algorithms:** Easily implement diverse RL algorithms like PPO, GRPO, and more with a modular programming model.
*   **Seamless LLM Integration:** Works with popular LLM frameworks such as FSDP, Megatron-LM, and vLLM via modular APIs.
*   **Efficient Resource Utilization:** Supports flexible device mapping for optimal GPU usage and scalability across various cluster sizes.
*   **Hugging Face Compatibility:** Ready-to-use with popular Hugging Face models.
*   **State-of-the-Art Performance:** Achieves SOTA throughput in LLM training.
*   **3D-HybridEngine:** Efficient actor model resharding and significantly reduced communication overhead.
*   **Multi-Modal RL** VLMs (vision-language models) such as Qwen2.5-vl, Kimi-VL
*   **Multi-turn with tool calling** Supports for LLM tools

## News
*   [2025/06] verl with Megatron backend enables large MoE models such as [DeepSeek-671b and Qwen3-236b](https://verl.readthedocs.io/en/latest/perf/dpsk.html).
*   [2025/06] verl team will provide latest project updates at [PyTorch Day China](https://www.lfasiallc.com/pytorch-day-china/) on June 7th. Meet our dev team in Beijing!
*   [2025/05] [PF-PPO](https://arxiv.org/abs/2409.06957), accepted to ICML 2025, is now supported in verl! PF-PPO enhances policy learning efficiency and robustness by filtering potentially noisy reward signals and reusing high-quality experiences via a replay buffer.
*   [2025/04] [Seed-Thinking-v1.5](https://github.com/ByteDance-Seed/Seed-Thinking-v1.5/blob/main/seed-thinking-v1.5.pdf) tech report is released! Trained with verl, Seed-Thinking-v1.5 achieves 86.7 on AIME 2024, 55.0 on Codeforces and 77.3 on GPQA, demonstrating excellent reasoning abilities in STEM and coding. Beyond reasoning tasks, the method demonstrates notable generalization across diverse domains.
*   [2025/04] [VAPO](https://arxiv.org/pdf/2504.05118) (value-based augmented PPO) paper covers our latest RL method for reasoning models. Trained from Qwen-32B-base model, VAPO achieves 60.4 on AIME 2024, outperforming DAPO-32B.
*   [2025/03] [DAPO](https://dapo-sia.github.io/) is the open-sourced SOTA RL algorithm that achieves 50 points on AIME 2024 based on the Qwen2.5-32B pre-trained model, surpassing the previous SOTA achieved by DeepSeek's GRPO (DeepSeek-R1-Zero-Qwen-32B). DAPO's training is fully powered by verl and the reproduction code is available in `recipe/dapo` now.
<details><summary> more... </summary>
<ul>

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

*   **[Documentation](https://verl.readthedocs.io/en/latest/index.html)**
*   **Quickstart:**
    *   [Installation](https://verl.readthedocs.io/en/latest/start/install.html)
    *   [Quickstart](https://verl.readthedocs.io/en/latest/start/quickstart.html)
    *   [Programming Guide](https://verl.readthedocs.io/en/latest/hybrid_flow.html) & [Tech Talk](https://hcqnc.xetlk.com/sl/3vACOK) (in Chinese)
    *   [PPO in verl](https://verl.readthedocs.io/en/latest/algo/ppo.html)
    *   [GRPO in verl](https://verl.readthedocs.io/en/latest/algo/grpo.html)
*   **Running a PPO example step-by-step:**
    *   [Prepare Data for Post-Training](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html)
    *   [Implement Reward Function for Dataset](https://verl.readthedocs.io/en/latest/preparation/reward_function.html)
    *   [PPO Example Architecture](https://verl.readthedocs.io/en/latest/examples/ppo_code_architecture.html)
    *   [Config Explanation](https://verl.readthedocs.io/en/latest/examples/config.html)
*   **Reproducible algorithm baselines:**
    *   [RL performance on coding, math](https://verl.readthedocs.io/en/latest/algo/baseline.html)
*   **For code explanation and advance usage (extension):**
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

## Performance Tuning Guide

Optimize your RL training with our detailed [performance tuning guide](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html).

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

verl now supports FSDP as the training engine (Megatron support coming soon) and both integrates with vLLM and SGLang as inference engines. Please refer to [this document](https://verl.readthedocs.io/blob/main/docs/amd_tutorial/amd_build_dockerfile_page.rst) for the installation guide and more information, and [this document](https://verl.readthedocs.io/blob/main/docs/amd_tutorial/amd_vllm_page.rst) for the vLLM performance tuning for ROCm.

## Citation

If you find this project useful, please cite our work:

```bibtex
@article{sheng2024hybridflow,
  title   = {HybridFlow: A Flexible and Efficient RLHF Framework},
  author  = {Guangming Sheng and Chi Zhang and Zilingfeng Ye and Xibin Wu and Wang Zhang and Ru Zhang and Yanghua Peng and Haibin Lin and Chuan Wu},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2409.19256}
}
```

## Acknowledgements

verl is inspired by Nemo-Aligner, Deepspeed-chat, and OpenRLHF and is supported by Bytedance, Anyscale, LMSys.org and many others.

## Awesome Projects Using verl

*   [TinyZero](https://github.com/Jiayi-Pan/TinyZero): DeepSeek R1 Zero reproduction for reasoning tasks.
*   [SkyThought](https://github.com/NovaSky-AI/SkyThought): RL training for Sky-T1-7B by NovaSky AI team.
*   [simpleRL-reason](https://github.com/hkust-nlp/simpleRL-reason): Investigating and Taming Zero Reinforcement Learning for Open Base Models in the Wild
*   [Easy-R1](https://github.com/hiyouga/EasyR1): Multi-modal RL training framework.
*   [OpenManus-RL](https://github.com/OpenManus/OpenManus-RL): LLM Agents RL tunning framework for multiple agent environments.
*   [rllm](https://github.com/agentica-project/rllm): async RL training with [verl-pipeline](https://github.com/agentica-project/verl-pipeline)
*   [RAGEN](https://github.com/ZihanWang314/ragen): general-purpose reasoning agent training framework
*   [Search-R1](https://github.com/PeterGriffinJin/Search-R1): RL with reasoning and searching (tool-call) interleaved LLMs.
*   [ReSearch](https://github.com/Agent-RL/ReSearch): Learning to Reason with Search for LLMs via Reinforcement Learning.
*   [Skywork-OR1](https://github.com/SkyworkAI/Skywork-OR1): Skywork open reaonser series
*   [ToRL](https://github.com/GAIR-NLP/ToRL): Scaling tool-integrated RL.
*   [Absolute Zero Reasoner](https://github.com/LeapLabTHU/Absolute-Zero-Reasoner): A no human curated data self-play framework for reasoning
*   [verl-agent](https://github.com/langfengQ/verl-agent): A scalable training framework for long-horizon LLM/VLM agents, along with a new algorithm GiGPO
*   [RL-Factory](https://github.com/Simple-Efficient/RL-Factory): An easy and efficient RL post-training framework for Agentic Learning
*   [ReTool](https://retool-rl.github.io/): ReTool: reinforcement learning for strategic tool use in LLMs. Code release is in progress...
*   [verl-tool](https://github.com/TIGER-AI-Lab/verl-tool): An unified and easy-to-extend tool-agent training framework based on verl
*   [PRIME](https://github.com/PRIME-RL/PRIME): Process reinforcement through implicit rewards
*   [GUI-R1](https://github.com/ritzz-ai/GUI-R1): GUI-R1: A Generalist R1-style Vision-Language Action Model For GUI Agents
*   [DeepRetrieval](https://github.com/pat-jj/DeepRetrieval): RL Training of Search Agent with Search/Retrieval Outcome
*   [Code-R1](https://github.com/ganler/code-r1): Reproducing R1 for Code with Reliable Rewards
*   [DeepResearcher](https://github.com/GAIR-NLP/DeepResearcher): Scaling deep research via reinforcement learning in real-world environments
*   [VAGEN](https://github.com/RAGEN-AI/VAGEN): Training VLM agents with multi-turn reinforcement learning
*   [RM-R1](https://arxiv.org/abs/2505.02387): RL training of reasoning reward models
*   [LUFFY](https://arxiv.org/pdf/2504.14945): Learning to Reason under Off-Policy Guidance
*   [DeepMath](https://github.com/zwhe99/DeepMath): DeepMath-103K data and series models for math reasoning
*   [Entropy Mechanism of RL](https://github.com/PRIME-RL/Entropy-Mechanism-of-RL): The Entropy Mechanism of Reinforcement Learning for Large Language Model Reasoning
*   [LLaSA-TTS-GRPO](https://github.com/channel-io/ch-tts-llasa-rl-grpo): TTS fine-tuning with GRPO optimization based on LLASA models
*   [PF-PPO](https://arxiv.org/abs/2409.06957): Policy Filtration for PPO based on the reliability of reward signals for more efficient and robust RLHF.
*   [RACRO](https://github.com/gyhdog99/RACRO2): Build multi-modal reasoning models via decoupling it into query-conditioned captioning and text-only reasoning
and many more awesome work listed in [recipe](recipe/README.md).

## Contribution

See our [CONTRIBUTING guide](CONTRIBUTING.md) for more information.

## About ByteDance Seed Team

verl is initiated by the ByteDance Seed team. Learn more about the team on [ByteDance Seed Team](https://team.doubao.com/).

---

Interested in joining us?  We're hiring!  Send us an [email](mailto:haibin.lin@bytedance.com) to inquire about internship and full-time opportunities in RL for agents.

[Back to Top](#)