# verl: Revolutionizing Reinforcement Learning for LLMs

verl is a powerful and flexible RL training library, enabling efficient and scalable reinforcement learning for Large Language Models (LLMs).

[![GitHub Repo stars](https://img.shields.io/github/stars/volcengine/verl)](https://github.com/volcengine/verl/stargazers)
[![Twitter](https://img.shields.io/twitter/follow/verl_project)](https://twitter.com/verl_project)
[![Slack](https://img.shields.io/badge/Slack-verl-blueviolet?logo=slack&amp)](https://join.slack.com/t/verlgroup/shared_invite/zt-2w5p9o4c3-yy0x2Q56s_VlGLsJ93A6vA)
[![Documentation](https://img.shields.io/badge/documentation-blue)](https://verl.readthedocs.io/en/latest/)
[![EuroSys Paper](https://img.shields.io/static/v1?label=EuroSys&message=Paper&color=red)](https://arxiv.org/pdf/2409.19256)
[![WeChat](https://img.shields.io/badge/微信-green?logo=wechat&amp)](https://raw.githubusercontent.com/eric-haibin-lin/verl-community/refs/heads/main/WeChat.JPG)

verl, initiated by the ByteDance Seed team, offers a production-ready solution for RL training, built upon the foundation of the "[HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256v2)" paper.

## Key Features:

*   **Flexible RL Algorithms:** Easily implement and customize RL algorithms like PPO and GRPO using our hybrid-controller programming model.
*   **Seamless LLM Integration:** Integrate with popular LLM frameworks (FSDP, Megatron-LM, vLLM, SGLang, etc.) through modular APIs.
*   **Efficient Device Mapping:** Optimize resource utilization and scalability with flexible device placement across various GPU setups.
*   **Hugging Face Compatibility:** Ready to use with popular Hugging Face models for rapid prototyping.
*   **State-of-the-Art Performance:** Achieve SOTA throughput with integrations for top LLM training and inference engines.
*   **3D-HybridEngine:** Minimize communication overhead and memory redundancy during training and generation phases.
*   **Comprehensive Algorithm Support:** Includes PPO, GRPO, ReMax, REINFORCE++, RLOO, PRIME, DAPO, DrGRPO, and more.
*   **Multi-Modality and Multi-Turn Support:** Supports VLMs, multi-modal RL, and tool calling via SGLang.
*   **Alignment Recipes and Optimization:** Includes recipes like SPPO, FlashAttention 2, Sequence Packing/Parallelism, LoRA, and expert parallelism.

## News & Updates

*   **Meetup at ICML Vancouver:** Join us on July 16th!
*   **AWS AI Hours Singapore:** Verl keynote on 7/8.
*   **Agent for SWE Meetup:** Verl & verl-agent project updates on 7/11.
*   **Megatron Integration:** Enables large MoE models like DeepSeek-671b and Qwen3-236b.
*   **PyTorch Day China:** Verl team updates on June 7th.
*   **Key Publications:**
    *   [Seed-Thinking-v1.5](https://github.com/ByteDance-Seed/Seed-Thinking-v1.5/blob/main/seed-thinking-v1.5.pdf)
    *   [DAPO](https://dapo-sia.github.io/)
    *   [VAPO](https://arxiv.org/pdf/2504.05118)
    *   [PF-PPO](https://arxiv.org/abs/2409.06957)
    *   [EuroSys 2025 Paper: HybridFlow](https://arxiv.org/abs/2409.19256v2)

  See the original [README](https://github.com/volcengine/verl) for more details.

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

## Performance Tuning & Advanced Usage

*   **[Performance Tuning Guide](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html)**
*   **Advanced Usage and Extension:**
    *   [Add Models with the FSDP Backend](https://verl.readthedocs.io/en/latest/advance/fsdp_extension.html)
    *   [Add Models with the Megatron-LM Backend](https://verl.readthedocs.io/en/latest/advance/megatron_extension.html)
    *   [Multi-turn Rollout Support](https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html)
    *   [Search Tool Integration](https://verl.readthedocs.io/en/latest/sglang_multiturn/search_tool_example.html)
    *   [Sandbox Fusion Integration](https://verl.readthedocs.io/en/latest/examples/sandbox_fusion_example.html)
    *   [Deployment using Separate GPU Resources](https://github.com/volcengine/verl/tree/main/examples/split_placement)
    *   [Extend to Other RL(HF) algorithms](https://verl.readthedocs.io/en/latest/advance/dpo_extension.html)
    *   [Ray API design tutorial](https://verl.readthedocs.io/en/latest/advance/placement.html)

## Key Technologies & Integrations

*   **Training Backends:** FSDP, FSDP2, and Megatron-LM.
*   **Inference Engines:** vLLM, SGLang, and Hugging Face Transformers.
*   **Model Support:** Compatible with various models including Qwen-3, Qwen-2.5, Llama3.1, Gemma2, and DeepSeek-LLM.
*   **Other:** Support LLM alignment recipes such as Self-play preference optimization (SPPO), Flash attention 2, sequence packing, LoRA, Liger-kernel, and more.
*   **AMD Support (ROCm Kernel):** verl now supports FSDP as the training engine (Megatron support coming soon) and both integrates with vLLM and SGLang as inference engines. Please refer to [this document](https://github.com/volcengine/verl/blob/main/docs/amd_tutorial/amd_build_dockerfile_page.rst) for the installation guide and more information, and [this document](https://github.com/volcengine/verl/blob/main/docs/amd_tutorial/amd_vllm_page.rst) for the vLLM performance tuning for ROCm.

## Community and Related Work

*   [Awesome work using verl](#awesome-work-using-verl)
*   [Contribution Guide](CONTRIBUTING.md)
*   [About ByteDance Seed Team](https://team.doubao.com/)
*   [Citation and acknowledgement](#citation-and-acknowledgement)

---

*   **We are HIRING!** Send us an [email](mailto:haibin.lin@bytedance.com) if you are interested in internship/FTE opportunities in RL for agents.