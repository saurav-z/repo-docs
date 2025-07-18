# verl: Revolutionizing LLM Training with Reinforcement Learning

**verl is an open-source, production-ready reinforcement learning (RL) library that empowers you to train and fine-tune large language models (LLMs) efficiently.**  Developed by the ByteDance Seed team and maintained by the verl community, verl offers a flexible and powerful framework for cutting-edge LLM training.  [Explore verl on GitHub](https://github.com/volcengine/verl)

[![GitHub Repo stars](https://img.shields.io/github/stars/volcengine/verl?style=social)](https://github.com/volcengine/verl/stargazers)
[![Twitter](https://img.shields.io/twitter/follow/verl_project?style=social)](https://twitter.com/verl_project)
[![Documentation](https://img.shields.io/badge/documentation-blue)](https://verl.readthedocs.io/en/latest/)
[![Slack](https://img.shields.io/badge/Slack-verl-blueviolet?logo=slack&amp)](https://join.slack.com/t/verlgroup/shared_invite/zt-2w5p9o4c3-yy0x2Q56s_VlGLsJ93A6vA)
[![EuroSys Paper](https://img.shields.io/static/v1?label=EuroSys&message=Paper&color=red)](https://arxiv.org/pdf/2409.19256)

## Key Features

*   **Flexible RL Algorithms:** Easily implement and extend diverse RL algorithms like PPO, GRPO, and more. Build RL dataflows with our hybrid-controller programming model in a few lines of code.
*   **Seamless Integration:** Integrate with existing LLM infrastructure such as FSDP, Megatron-LM, vLLM, SGLang, and Hugging Face models using our modular APIs.
*   **Efficient Resource Utilization:** Utilize various GPU placements for optimized resource allocation and scalability across different cluster sizes.
*   **State-of-the-Art Throughput:** Benefit from SOTA LLM training and inference engine integrations and achieve SOTA RL throughput.
*   **3D-HybridEngine for Optimized Communication:**  Reduce memory redundancy and communication overhead with efficient actor model resharding.
*   **Hugging Face Compatibility**:  Ready integration with popular HuggingFace models, including Qwen, Llama, and Gemma.

## Recent Updates and News

*   **[2025/07]** The first verl meetup will be held at ICML Vancouver on July 16th! Please [join us](https://lu.ma/0ek2nyao) if you are at ICML! (onsite only)
*   **[2025/07]** verl keynote at [AWS AI Hours Singapore](https://pages.awscloud.com/aws-ai-hours-sg.html#agenda) on 7/8, verl & verl-agent project updates at [Agent for SWE meetup](https://lu.ma/e498qhsi) by LF AI & Data Singapore on 7/11.
*   **[2025/06]** verl with Megatron backend enables large MoE models such as [DeepSeek-671b and Qwen3-236b](https://verl.readthedocs.io/en/latest/perf/dpsk.html).
*   **[2025/06]** verl team will provide latest project updates at [PyTorch Day China](https://www.lfasiallc.com/pytorch-day-china/) on June 7th. Meet our dev team in Beijing!
*   **[2025/04]** [Seed-Thinking-v1.5](https://github.com/ByteDance-Seed/Seed-Thinking-v1.5/blob/main/seed-thinking-v1.5.pdf) tech report is released!
*   **[2025/03]** [DAPO](https://dapo-sia.github.io/) is the open-sourced SOTA RL algorithm.

*(See full list and more details in the original README)*

## Getting Started

*   **[Documentation](https://verl.readthedocs.io/en/latest/index.html)**
*   **Quickstart:**
    *   [Installation](https://verl.readthedocs.io/en/latest/start/install.html)
    *   [Quickstart](https://verl.readthedocs.io/en/latest/start/quickstart.html)
    *   [Programming Guide](https://verl.readthedocs.io/en/latest/hybrid_flow.html) & [Tech Talk](https://hcqnc.xetlk.com/sl/3vACOK) (in Chinese)
    *   [PPO in verl](https://verl.readthedocs.io/en/latest/algo/ppo.html)
    *   [GRPO in verl](https://verl.readthedocs.io/en/latest/algo/grpo.html)
*   **Examples:**  Follow the steps for running a PPO example step-by-step:
    *   [Prepare Data for Post-Training](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html)
    *   [Implement Reward Function for Dataset](https://verl.readthedocs.io/en/latest/preparation/reward_function.html)
    *   [PPO Example Architecture](https://verl.readthedocs.io/en/latest/examples/ppo_code_architecture.html)
    *   [Config Explanation](https://verl.readthedocs.io/en/latest/examples/config.html)
*   **Algorithm Baselines**: Explore [RL performance on coding, math](https://verl.readthedocs.io/en/latest/algo/baseline.html)

## Performance Tuning

Optimize your RL training with our detailed [performance tuning guide](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html).

## Framework Integration Updates

*   **vLLM >= v0.8.2:** Supports integration. Refer to [this document](https://github.com/volcengine/verl/blob/main/docs/README_vllm0.8.md).
*   **Latest SGLang:** Fully supported and offers unique features with multi-turn agentic RL. Refer to [this document](https://verl.readthedocs.io/en/latest/workers/sglang_worker.html).
*   **FSDP2**: Fully embrace FSDP2. See https://github.com/volcengine/verl/pull/1026 for more details.
*   **AMD Support (ROCm Kernel):** Integrates with vLLM and SGLang. Refer to [this document](https://verl.readthedocs.io/blob/main/docs/amd_tutorial/amd_vllm_page.rst) for vLLM performance tuning for ROCm.

## Contributions and Community

We welcome contributions! See our [contributions guide](CONTRIBUTING.md).

## Related Projects and Papers

*   [HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256v2)
*   [A Framework for Training Large Language Models for Code Generation via Proximal Policy Optimization](https://i.cs.hku.hk/~cwu/papers/gmsheng-NL2Code24.pdf)

## About ByteDance Seed Team

Learn more about the team and their work.
*   [Website](https://team.doubao.com/)
*   [WeChat](https://github.com/user-attachments/assets/469535a8-42f2-4797-acdf-4f7a1d4a0c3e)
*   [Xiaohongshu](https://www.xiaohongshu.com/user/profile/668e7e15000000000303157d?xsec_token=ABl2-aqekpytY6A8TuxjrwnZskU-6BsMRE_ufQQaSAvjc%3D&xsec_source=pc_search)
*   [Zhihu](https://www.zhihu.com/org/dou-bao-da-mo-xing-tuan-dui/)

**Hire:** We are HIRING! Send us an [email](mailto:haibin.lin@bytedance.com) if you are interested in internship/FTE opportunities in RL for agents.
```

Key improvements and SEO considerations:

*   **Clear and Concise Hook:** The one-sentence hook immediately conveys the library's value proposition.
*   **Keyword Optimization:** Includes relevant keywords like "Reinforcement Learning," "LLM training," and specific algorithm names (PPO, GRPO).
*   **Strategic Headings:** Uses clear, descriptive headings to organize information, improving readability and SEO.
*   **Bulleted Key Features:** Highlights the most important features in an easily digestible format.
*   **Internal Linking:** Links to specific sections within the documentation.
*   **External Linking:**  Links to the original repo, relevant papers, and the ByteDance Seed Team's resources.
*   **Actionable Language:** Uses calls to action like "Explore," and "Getting Started" to encourage user engagement.
*   **Concise Summarization:** Condenses the original README while retaining essential information.
*   **Social Media Integration**: Added social media links and badges to promote user engagement and awareness.
*   **Clear Structure**:  Improved the visual layout with bullet points.
*   **Focused Content:** Maintains a focus on the user benefits.