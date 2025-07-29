<div align="center">
    <!--  Logo (optional, replace with a relevant image) -->
    <img src="https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216" alt="verl Logo" width="150">
    <br>
    <h1>verl: Train Cutting-Edge LLMs with Reinforcement Learning</h1>
    <p>verl, initiated by ByteDance Seed and maintained by the community, is your go-to library for flexible, efficient, and production-ready reinforcement learning (RL) training of large language models (LLMs).</p>
</div>

<div align="center">
    <!-- Badges -->
    <a href="https://deepwiki.com/volcengine/verl"><img src="https://devin.ai/assets/deepwiki-badge.png" alt="Ask DeepWiki.com" style="height:20px;"></a>
    [![GitHub Repo stars](https://img.shields.io/github/stars/volcengine/verl)](https://github.com/volcengine/verl/stargazers)
    [![Twitter](https://img.shields.io/twitter/follow/verl_project)](https://twitter.com/verl_project)
    <a href="https://join.slack.com/t/verlgroup/shared_invite/zt-2w5p9o4c3-yy0x2Q56s_VlGLsJ93A6vA"><img src="https://img.shields.io/badge/Slack-verl-blueviolet?logo=slack&amp"></a>
    <a href="https://arxiv.org/pdf/2409.19256"><img src="https://img.shields.io/static/v1?label=EuroSys&message=Paper&color=red"></a>
    [![Documentation](https://img.shields.io/badge/documentation-blue)](https://verl.readthedocs.io/en/latest/)
    <a href="https://raw.githubusercontent.com/eric-haibin-lin/verl-community/refs/heads/main/WeChat.JPG"><img src="https://img.shields.io/badge/微信-green?logo=wechat&amp"></a>
</div>

## Key Features

*   **Flexible RL Algorithms:** Easily extend and implement diverse RL algorithms like PPO, GRPO, and more.
*   **Seamless LLM Integration:** Integrates with existing LLM frameworks such as FSDP, Megatron-LM, vLLM, and SGLang.
*   **Efficient Device Mapping:** Supports flexible model placement across GPUs for optimal resource utilization.
*   **Hugging Face Compatibility:** Ready-to-use integration with popular Hugging Face models.
*   **State-of-the-Art Performance:** Achieve SOTA LLM training and inference speeds.
*   **3D-HybridEngine for Efficiency:** Reduces memory redundancy and communication overhead during training.

## What's New

*   **[2025/07]** ReTool Recipe is fully open-sourced. [Blog](https://www.notion.so/verl-reTool-recipe-Using-multi-round-conversations-and-code-sandboxing-to-improve-the-math-of-large-23a8b5b7feba80b386b2e5b5e3c1cde0)
*   **[2025/07]** First verl meetup at ICML Vancouver on July 16th.
*   **[2025/07]** verl keynote at [AWS AI Hours Singapore](https://pages.awscloud.com/aws-ai-hours-sg.html#agenda) on 7/8, verl & verl-agent project updates at [Agent for SWE meetup](https://lu.ma/e498qhsi) by LF AI & Data Singapore on 7/11.
*   **[2025/06]** verl with Megatron backend enables large MoE models such as [DeepSeek-671b and Qwen3-236b](https://verl.readthedocs.io/en/latest/perf/dpsk.html).
*   **[2025/06]** verl team will provide latest project updates at [PyTorch Day China](https://www.lfasiallc.com/pytorch-day-china/) on June 7th. Meet our dev team in Beijing!
*   **[2025/04]** [Seed-Thinking-v1.5](https://github.com/ByteDance-Seed/Seed-Thinking-v1.5/blob/main/seed-thinking-v1.5.pdf) tech report is released!
*   **[2025/03]** [DAPO](https://dapo-sia.github.io/) is the open-sourced SOTA RL algorithm.
*   **[2025/04]** [VAPO](https://arxiv.org/pdf/2504.05118) (value-based augmented PPO) paper covers our latest RL method.
*   **[2025/05]** [PF-PPO](https://arxiv.org/abs/2409.06957), accepted to ICML 2025, is now supported in verl!
*   **[2025/04]** Tutorial at ICLR 2025 Expo, SCI-FM workshop, and LMSys afterparty.
*   **[2025/03]** verl v0.3.0.post1 is released!
*   **[2025/05]** verl will be presented at [A2M Shanghai](https://a2m.msup.com.cn/home/?aid=4488&city=shanghai) on 5/16 - 5/17.
*   **[2025/05]** verl will be presented at [GOSIM x PyTorch Day 2025](https://paris2025.gosim.org/).
*   **[2025/03]** We introduced the programming model of verl at the [vLLM Beijing Meetup](https://mp.weixin.qq.com/s/n77GibL2corAtQHtVEAzfg) and [verl intro and updates](https://github.com/eric-haibin-lin/verl-community/blob/main/slides/verl-lmsys-meetup.pdf) at the [SGLang-LMSYS Org Meetup](https://lu.ma/ntjrr7ig) in Sunnyvale mid-March.
*   **[2025/03]** We will present verl(HybridFlow) at EuroSys 2025.
*   **[2025/02]** verl v0.2.0.post2 is released!
*   **[2025/02]** We presented verl in the <a href="https://lu.ma/ji7atxux">Bytedance/NVIDIA/Anyscale Ray Meetup</a>.
*   **[2025/01]** [Doubao-1.5-pro](https://team.doubao.com/zh/special/doubao_1_5_pro) is released with SOTA-level performance on LLM & VLM.
*   **[2024/12]** verl is presented at Ray Forward 2024.
*   **[2024/12]** The team presented <a href="https://neurips.cc/Expo/Conferences/2024/workshop/100677">Post-training LLMs: From Algorithms to Infrastructure</a> at NeurIPS 2024.
*   **[2024/10]** verl is presented at Ray Summit.
*   **[2024/08]** HybridFlow (verl) is accepted to EuroSys 2025.

## Getting Started

*   [Documentation](https://verl.readthedocs.io/en/latest/index.html)
*   [Installation](https://verl.readthedocs.io/en/latest/start/install.html)
*   [Quickstart](https://verl.readthedocs.io/en/latest/start/quickstart.html)
*   [Programming Guide](https://verl.readthedocs.io/en/latest/hybrid_flow.html) & [Tech Talk](https://hcqnc.xetlk.com/sl/3vACOK) (in Chinese)
*   [PPO in verl](https://verl.readthedocs.io/en/latest/algo/ppo.html)
*   [GRPO in verl](https://verl.readthedocs.io/en/latest/algo/grpo.html)

Explore the documentation for detailed guides on:

*   Preparing data
*   Implementing reward functions
*   Understanding PPO architecture in verl
*   Reproducing algorithm baselines

## Key Technologies & Integrations

*   **Training Backends:** FSDP, FSDP2, Megatron-LM
*   **Inference Engines:** vLLM, SGLang, HF Transformers
*   **Supported Models:** Hugging Face Transformers, Modelscope Hub, Qwen-3, Qwen-2.5, Llama3.1, Gemma2, DeepSeek-LLM, etc.
*   **RL Algorithms:** PPO, GRPO, ReMax, REINFORCE++, RLOO, PRIME, DAPO, DrGRPO, KL_Cov & Clip_Cov, and others.
*   **Features:** SFT, Multi-modal RL, multi-turn with tool calling, alignment recipes, Flash attention 2, sequence packing, sequence parallelism, LoRA, and expert parallelism.

## Performance & Optimization

*   [Performance Tuning Guide](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html)
*   Upgrade to vLLM >= v0.8.2
*   Use Latest SGLang
*   Upgrade to FSDP2
*   AMD Support (ROCm Kernel)

## Explore Community Resources

*   **Blogs:** Access community-contributed blogs for insights and practical examples.
*   **Awesome work using verl:** Review projects leveraging verl for LLM and VLM agent development.

## Citation

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

## Contribution

See the [contributions guide](CONTRIBUTING.md).

## About ByteDance Seed Team

[ByteDance Seed Team](https://team.doubao.com/) is committed to advancing AI foundation models.

```
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
```

---

**Interested in joining the team?** Contact us at [haibin.lin@bytedance.com](mailto:haibin.lin@bytedance.com) for internship/FTE opportunities.