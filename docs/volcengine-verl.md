# verl: Revolutionizing LLM Training with Reinforcement Learning

verl is a cutting-edge open-source library from the ByteDance Seed team, designed to make Reinforcement Learning (RL) for Large Language Models (LLMs) more flexible, efficient, and production-ready.  Explore the original repo on [GitHub](https://github.com/volcengine/verl).

[![GitHub Repo stars](https://img.shields.io/github/stars/volcengine/verl?style=social)](https://github.com/volcengine/verl/stargazers)
[![Twitter](https://img.shields.io/twitter/follow/verl_project?style=social)](https://twitter.com/verl_project)

**Key Features:**

*   **Flexible RL Algorithms:** Easily implement diverse RL algorithms like PPO, GRPO, and more.
*   **Seamless LLM Integration:** Works seamlessly with existing LLM frameworks (FSDP, Megatron-LM, vLLM, etc.).
*   **Efficient Resource Utilization:** Supports flexible device mapping for efficient use of GPUs.
*   **Hugging Face Compatibility:** Ready integration with popular Hugging Face models.
*   **State-of-the-Art Performance:**  Achieves state-of-the-art throughput for LLM training and RL.
*   **Optimized Communication:** Efficient actor model resharding with 3D-HybridEngine to reduce overhead.
*   **Multi-turn and Tool Calling Support:** With SGLang integration.
*   **LoRA and Expert Parallelism:** For efficient training.
*   **Experiment Tracking:**  Integrations with Wandb, SwanLab, MLflow, and TensorBoard.

**What's New:**

*   **[EuroSys 2025]** verl will be presented at EuroSys 2025, with a presentation of [HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256v2).
*   **[ICML Vancouver 2025]** Join us at the first verl meetup on July 16th.
*   **[DeepSeek and Qwen3 Support]** verl enables large MoE models such as [DeepSeek-671b and Qwen3-236b](https://verl.readthedocs.io/en/latest/perf/dpsk.html).
*   **[DAPO Algorithm]** DAPO achieves SOTA RL results using verl.
*   **[PF-PPO Algorithm]** PF-PPO, accepted to ICML 2025, is now supported.

**Getting Started:**

*   [Documentation](https://verl.readthedocs.io/en/latest/index.html)
*   [Installation Guide](https://verl.readthedocs.io/en/latest/start/install.html)
*   [Quickstart](https://verl.readthedocs.io/en/latest/start/quickstart.html)
*   [Programming Guide](https://verl.readthedocs.io/en/latest/hybrid_flow.html)

**Explore Key Examples:**

*   PPO: [PPO in verl](https://verl.readthedocs.io/en/latest/algo/ppo.html)
*   GRPO: [GRPO in verl](https://verl.readthedocs.io/en/latest/algo/grpo.html)
*   [PPO Example Architecture](https://verl.readthedocs.io/en/latest/examples/ppo_code_architecture.html)
*   [Config Explanation](https://verl.readthedocs.io/en/latest/examples/config.html)

**Further Resources:**

*   [Performance Tuning Guide](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html)
*   [AMD Support (ROCm Kernel)](https://verl.readthedocs.io/en/latest/docs/amd_tutorial/amd_build_dockerfile_page.rst)
*   [Upgrade to vLLM >= v0.8.2](https://github.com/volcengine/verl/blob/main/docs/README_vllm0.8.md)
*   [Use Latest SGLang](https://verl.readthedocs.io/en/latest/workers/sglang_worker.html)
*   [Upgrade to FSDP2](https://github.com/volcengine/verl/pull/1026)
*   [Blogs from the community](https://verl.readthedocs.io/en/latest/blogs.html)
*   [Awesome work using verl](https://github.com/volcengine/verl/blob/main/README.md#awesome-work-using-verl)

**Citation:**

If you use verl, please cite the [HybridFlow paper](https://arxiv.org/abs/2409.19256v2).

**Join the Community:**

*   [Slack](https://join.slack.com/t/verlgroup/shared_invite/zt-2w5p9o4c3-yy0x2Q56s_VlGLsJ93A6vA)
*   [Twitter](https://twitter.com/verl_project)
*   [WeChat](https://raw.githubusercontent.com/eric-haibin-lin/verl-community/refs/heads/main/WeChat.JPG)

---