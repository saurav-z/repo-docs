# verl: Unleash the Power of Reinforcement Learning for LLMs

**verl** is a cutting-edge reinforcement learning (RL) library, enabling you to train and fine-tune Large Language Models (LLMs) with unprecedented flexibility and efficiency. **[Explore the verl repository on GitHub](https://github.com/volcengine/verl)** and discover how to revolutionize your LLM training workflows!

<div align="center">
  <a href="https://deepwiki.com/volcengine/verl"><img src="https://devin.ai/assets/deepwiki-badge.png" alt="Ask DeepWiki.com" style="height:20px;"></a>
  [![GitHub Repo stars](https://img.shields.io/github/stars/volcengine/verl)](https://github.com/volcengine/verl/stargazers)
  [![Twitter](https://img.shields.io/twitter/follow/verl_project)](https://twitter.com/verl_project)
  <a href="https://join.slack.com/t/verlgroup/shared_invite/zt-2w5p9o4c3-yy0x2Q56s_VlGLsJ93A6vA"><img src="https://img.shields.io/badge/Slack-verl-blueviolet?logo=slack&amp"></a>
  <a href="https://arxiv.org/pdf/2409.19256"><img src="https://img.shields.io/static/v1?label=EuroSys&message=Paper&color=red"></a>
  [![Documentation](https://img.shields.io/badge/documentation-blue)](https://verl.readthedocs.io/en/latest/)
  <a href="https://raw.githubusercontent.com/eric-haibin-lin/verl-community/refs/heads/main/WeChat.JPG"><img src="https://img.shields.io/badge/微信-green?logo=wechat&amp"></a>
</div>

![seed logo](https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216)

## Key Features:

*   **Flexible RL Algorithms:** Easily implement and extend various RL algorithms like PPO, GRPO, and more.
*   **Seamless LLM Integration:** Compatible with popular LLM frameworks like FSDP, Megatron-LM, vLLM, and SGLang.
*   **Efficient Device Mapping:** Supports flexible GPU configurations for optimal resource utilization.
*   **Hugging Face Integration:** Ready-to-use with popular Hugging Face models.
*   **State-of-the-Art Performance:** Achieve SOTA LLM training and inference with efficient integrations.
*   **Model Compatibility:** Supports a wide range of models, including Qwen-3, Qwen-2.5, Llama3.1, Gemma2, and DeepSeek-LLM, with integration for supervised fine-tuning.
*   **Advanced RL Techniques:** Supports a range of RL algorithms including PPO, GRPO, ReMax, REINFORCE++, RLOO, PRIME, DAPO, DrGRPO, KL_Cov & Clip_Cov, with support for model-based and function-based reward for math, coding, etc. Also supports multi-modal RL.
*   **Advanced Optimization:** Features like Flash attention 2, sequence packing, sequence parallelism, LoRA, and expert parallelism for optimized performance.
*   **Experiment Tracking:** Integrates with tools like wandb, swanlab, mlflow, and tensorboard for effective experiment management.
*   **LoRA-RL:** Multi-gpu LoRA RL Support

## News

*   **(Upcoming Events):** Keynote at AWS AI Hours Singapore, updates at Agent for SWE meetup and ICML @ Vancouver (July 2025)
*   **(Recent Model Support):** Support for large MoE models such as DeepSeek-671b and Qwen3-236b using Megatron backend (June 2025).
*   **(Recent Research):** The team has contributed to research with recent projects. DAPO is trained with verl for Qwen2.5-32B, with a paper available on the project's GitHub repository and documentation
*   **(Recent Releases):** verl v0.3.0.post1 and verl v0.2.0.post2 are released!

## Getting Started

*   **Documentation:** [https://verl.readthedocs.io/en/latest/index.html](https://verl.readthedocs.io/en/latest/index.html)
*   **Installation:** [https://verl.readthedocs.io/en/latest/start/install.html](https://verl.readthedocs.io/en/latest/start/install.html)
*   **Quickstart:** [https://verl.readthedocs.io/en/latest/start/quickstart.html](https://verl.readthedocs.io/en/latest/start/quickstart.html)
*   **Programming Guide:** [https://verl.readthedocs.io/en/latest/hybrid_flow.html](https://verl.readthedocs.io/en/latest/hybrid_flow.html)
*   **PPO in verl:** [https://verl.readthedocs.io/en/latest/algo/ppo.html](https://verl.readthedocs.io/en/latest/algo/ppo.html)
*   **GRPO in verl:** [https://verl.readthedocs.io/en/latest/algo/grpo.html](https://verl.readthedocs.io/en/latest/algo/grpo.html)

## Performance Tuning Guide

Optimize your verl performance with our comprehensive [performance tuning guide](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html).

## Community Resources

Access a wealth of information and resources, including:

*   **Blogs from the community** - Find community resources, and information for deployments.
*   **Citation and acknowledgement** - Learn how verl is inspired, and is cited in relevant documents.

## Contribution Guide

Contribute to the verl project by following our [contributions guide](CONTRIBUTING.md).

## About ByteDance Seed Team

Learn more about the team behind verl:

*   [ByteDance Seed Team Website](https://team.doubao.com/)

**Interested in joining us?**  Contact us at [haibin.lin@bytedance.com] if you're interested in RL for agents.