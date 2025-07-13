# verl: Accelerate LLM Training with Reinforcement Learning 🚀

**verl is an open-source reinforcement learning (RL) library designed for training and fine-tuning large language models (LLMs) efficiently, flexibly, and production-ready.** Built by the ByteDance Seed team, verl empowers researchers and developers to unlock the full potential of LLMs through cutting-edge RL techniques.

[![GitHub Repo stars](https://img.shields.io/github/stars/volcengine/verl)](https://github.com/volcengine/verl)
[![Twitter](https://img.shields.io/twitter/follow/verl_project)](https://twitter.com/verl_project)
[![Documentation](https://img.shields.io/badge/documentation-blue)](https://verl.readthedocs.io/en/latest/)
<a href="https://join.slack.com/t/verlgroup/shared_invite/zt-2w5p9o4c3-yy0x2Q56s_VlGLsJ93A6vA"><img src="https://img.shields.io/badge/Slack-verl-blueviolet?logo=slack&amp"></a>
<a href="https://arxiv.org/pdf/2409.19256"><img src="https://img.shields.io/static/v1?label=EuroSys&message=Paper&color=red"></a>
<a href="https://raw.githubusercontent.com/eric-haibin-lin/verl-community/refs/heads/main/WeChat.JPG"><img src="https://img.shields.io/badge/微信-green?logo=wechat&amp"></a>

## Key Features

*   **Flexible RL Algorithms:** Easily extend and implement diverse RL algorithms like PPO, GRPO, and more with a modular programming model.
*   **Seamless LLM Integration:** Compatible with popular LLM frameworks (FSDP, Megatron-LM, vLLM, SGLang, etc.) through modular APIs, streamlining your workflow.
*   **Efficient Resource Utilization:** Flexible device mapping and 3D-HybridEngine technology for optimized resource use and communication, improving performance and reducing memory overhead.
*   **Hugging Face Integration:** Ready-to-use with a wide range of Hugging Face models, including Qwen, Llama, and Gemma, for rapid prototyping and experimentation.
*   **State-of-the-Art Throughput:** Experience SOTA LLM training with optimized integrations, delivering superior performance compared to previous versions.
*   **LoRA Support:** Multi-GPU LoRA RL to save memory.
*   **Experiment Tracking:** Integration with wandb, swanlab, mlflow and tensorboard.
*   **Multi-modal RL:** Support vision-language models (VLMs) and multi-modal RL with Qwen2.5-vl, Kimi-VL.

## News and Updates

Stay informed about the latest verl developments:

*   **[July 2025]** The first verl meetup will be held at ICML Vancouver on July 16th!
*   **[July 2025]** verl keynote at [AWS AI Hours Singapore](https://pages.awscloud.com/aws-ai-hours-sg.html#agenda) on 7/8, verl & verl-agent project updates at [Agent for SWE meetup](https://lu.ma/e498qhsi) by LF AI & Data Singapore on 7/11.
*   **[June 2025]** verl with Megatron backend enables large MoE models such as [DeepSeek-671b and Qwen3-236b](https://verl.readthedocs.io/en/latest/perf/dpsk.html).
*   **[June 2025]** verl team will provide latest project updates at [PyTorch Day China](https://www.lfasiallc.com/pytorch-day-china/) on June 7th. Meet our dev team in Beijing!
*   **[April 2025]** [Seed-Thinking-v1.5](https://github.com/ByteDance-Seed/Seed-Thinking-v1.5/blob/main/seed-thinking-v1.5.pdf) tech report is released! Trained with verl, Seed-Thinking-v1.5 achieves 86.7 on AIME 2024, 55.0 on Codeforces and 77.3 on GPQA, demonstrating excellent reasoning abilities in STEM and coding. Beyond reasoning tasks, the method demonstrates notable generalization across diverse domains.
*   **[March 2025]** [DAPO](https://dapo-sia.github.io/) is the open-sourced SOTA RL algorithm that achieves 50 points on AIME 2024 based on the Qwen2.5-32B pre-trained model, surpassing the previous SOTA achieved by DeepSeek's GRPO (DeepSeek-R1-Zero-Qwen-32B). DAPO's training is fully powered by verl and the reproduction code is available in `recipe/dapo` now.

See more updates in the original [README](https://github.com/volcengine/verl).

## Getting Started

Get up and running with verl quickly:

*   **[Documentation](https://verl.readthedocs.io/en/latest/index.html)**
*   [Installation](https://verl.readthedocs.io/en/latest/start/install.html)
*   [Quickstart](https://verl.readthedocs.io/en/latest/start/quickstart.html)
*   [Programming Guide](https://verl.readthedocs.io/en/latest/hybrid_flow.html)
*   [PPO in verl](https://verl.readthedocs.io/en/latest/algo/ppo.html)
*   [GRPO in verl](https://verl.readthedocs.io/en/latest/algo/grpo.html)

## Performance & Tuning

Maximize your RL training efficiency:

*   [Performance Tuning Guide](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html)

## Advanced Features

*   **AMD Support (ROCm Kernel)**
*   **Upgrade to vLLM >= v0.8.2**
*   **Use Latest SGLang**
*   **Upgrade to FSDP2**

## Contribution

See [contributions guide](CONTRIBUTING.md)

## Acknowledgements

verl is inspired by the design of Nemo-Aligner, Deepspeed-chat and OpenRLHF. The project is adopted and contributed by Bytedance, Anyscale, LMSys.org, [Alibaba Qwen team](https://github.com/QwenLM/), Shanghai AI Lab, Tsinghua University, UC Berkeley, UCLA, UIUC, University of Hong Kong, ke.com, [All Hands AI](https://www.all-hands.dev/), [ModelBest](http://modelbest.cn/), JD AI Lab, Microsoft Research, [StepFun](https://www.stepfun.com/), Amazon, LinkedIn, Meituan, [Camel-AI](https://www.camel-ai.org/), [OpenManus](https://github.com/OpenManus), Xiaomi, NVIDIA research, [Baichuan](https://www.baichuan-ai.com/home), [RedNote](https://www.xiaohongshu.com/), [SwissAI](https://www.swiss-ai.org/), [Moonshot AI (Kimi)](https://www.moonshot-ai.com/), Baidu, Snowflake, Skywork.ai, JetBrains, [IceSword Lab](https://www.iceswordlab.com), and many more.

## About ByteDance Seed Team

Founded in 2023, ByteDance Seed Team is dedicated to crafting the industry's most advanced AI foundation models. The team aspires to become a world-class research team and make significant contributions to the advancement of science and society. You can get to know Bytedance Seed better through the following channels👇
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