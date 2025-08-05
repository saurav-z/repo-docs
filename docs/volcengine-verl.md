<!-- Improved README.md -->
<div align="center">
  <!-- Title -->
  <h1>verl: Production-Ready Reinforcement Learning for LLMs</h1>

  <!-- Badges -->
  <a href="https://github.com/volcengine/verl">
    <img src="https://img.shields.io/github/stars/volcengine/verl?style=social" alt="Stars">
  </a>
  <a href="https://twitter.com/verl_project">
    <img src="https://img.shields.io/twitter/follow/verl_project?style=social" alt="Follow on Twitter">
  </a>
  <a href="https://verl.readthedocs.io/en/latest/">
    <img src="https://img.shields.io/badge/Documentation-blue" alt="Documentation">
  </a>
  <a href="https://join.slack.com/t/verlgroup/shared_invite/zt-2w5p9o4c3-yy0x2Q56s_VlGLsJ93A6vA">
    <img src="https://img.shields.io/badge/Slack-verl-blueviolet?logo=slack&amp" alt="Slack">
  </a>
  <a href="https://arxiv.org/pdf/2409.19256">
    <img src="https://img.shields.io/static/v1?label=EuroSys&message=Paper&color=red" alt="EuroSys Paper">
  </a>
  <a href="https://deepwiki.com/volcengine/verl"><img src="https://devin.ai/assets/deepwiki-badge.png" alt="Ask DeepWiki.com" style="height:20px;"></a>
</div>

<div align="center">
  <img src="https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216" alt="verl Logo" width="100">
</div>

<p align="center">
  <b>verl is a powerful open-source library enabling efficient and flexible Reinforcement Learning (RL) for Large Language Models (LLMs).</b>
  <br>
  <a href="https://github.com/volcengine/verl">Explore the verl Repository</a>
</p>

<!-- Key Features -->
## Key Features

*   **Flexible and Extensible**: Easily integrate diverse RL algorithms and build complex dataflows (GRPO, PPO, etc.) with minimal code.
*   **Seamless LLM Integration**:  Compatible with existing LLM frameworks such as FSDP, Megatron-LM, vLLM, and SGLang.
*   **Efficient Resource Utilization**: Flexible device mapping supports efficient resource utilization and scalability across cluster sizes.
*   **Hugging Face Compatibility**: Ready integration with popular Hugging Face models.
*   **State-of-the-Art Performance**: Achieve leading throughput with SOTA LLM training and inference engine integrations.
*   **3D-HybridEngine**:  Eliminates memory redundancy and significantly reduces communication overhead.

<!-- Getting Started -->
## Getting Started

*   **Documentation**: Comprehensive documentation is available at [https://verl.readthedocs.io/en/latest/index.html](https://verl.readthedocs.io/en/latest/index.html).
*   **Quickstart**:  Get up and running quickly with our [Quickstart](https://verl.readthedocs.io/en/latest/start/quickstart.html) guide.
*   **Examples**: Explore practical examples, including [PPO](https://verl.readthedocs.io/en/latest/algo/ppo.html) and [GRPO](https://verl.readthedocs.io/en/latest/algo/grpo.html), with code architecture and configuration explanations.
*   **Reproducible Baselines**:  Reproduce algorithm baselines to understand RL performance on coding and math: [https://verl.readthedocs.io/en/latest/algo/baseline.html](https://verl.readthedocs.io/en/latest/algo/baseline.html)

<!-- News -->
## News

*   [2025/07] ReTool recipe is fully open sourced, and a verl meetup at ICML Vancouver.
*   [2025/06] verl with Megatron backend enables large MoE models such as [DeepSeek-671b and Qwen3-236b](https://verl.readthedocs.io/en/latest/perf/dpsk.html).
*   [2025/04] [DAPO](https://dapo-sia.github.io/) is the open-sourced SOTA RL algorithm that achieves 50 points on AIME 2024
*   [2025/04] [VAPO](https://arxiv.org/pdf/2504.05118) (value-based augmented PPO) paper covers our latest RL method for reasoning models. Trained from Qwen-32B-base model, VAPO achieves 60.4 on AIME 2024, outperforming DAPO-32B.
*   [2025/03] verl v0.3.0.post1 is released! See [release note](https://github.com/volcengine/verl/releases/) for details. It achieves [~1.4x speedup](https://tongyx361.github.io/blogs/posts/verl-intro/#/verl-flexible-and-efficient-rl-for-llms) compared to prev versions.

  <details><summary> more... </summary>
    See the original README for more news.
  </details>


<!-- Performance Tuning -->
## Performance Tuning

Optimize your RL training with our detailed [performance tuning guide](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html).

<!-- Upgrade Notes -->
## Upgrade Notes

*   **vLLM >= v0.8.2**:  Ensure compatibility with our [vLLM upgrade guide](https://github.com/volcengine/verl/blob/main/docs/README_vllm0.8.md).
*   **Latest SGLang**: Benefit from the latest features by consulting [this document](https://verl.readthedocs.io/en/latest/workers/sglang_worker.html).
*   **FSDP2**:  Take advantage of FSDP2 for improved throughput and memory usage, as described in [this guide](https://github.com/volcengine/verl/pull/1026).
*   **AMD Support**:  Learn about AMD ROCm kernel support with [these documents](https://github.com/volcengine/verl/blob/main/docs/amd_tutorial/amd_build_dockerfile_page.rst) and [this document](https://github.com/volcengine/verl/blob/main/docs/amd_tutorial/amd_vllm_page.rst).

<!-- Citation -->
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

verl is inspired by the design of Nemo-Aligner, Deepspeed-chat and OpenRLHF. The project is adopted and contributed by Bytedance, Anyscale, LMSys.org, [Alibaba Qwen team](https://github.com/QwenLM/), Shanghai AI Lab, Tsinghua University, UC Berkeley, UCLA, UIUC, University of Hong Kong, ke.com, [All Hands AI](https://www.all-hands.dev/), [ModelBest](http://modelbest.cn/), JD AI Lab, Microsoft Research, [StepFun](https://www.stepfun.com/), Amazon, LinkedIn, Meituan, [Camel-AI](https://www.camel-ai.org/), [OpenManus](https://github.com/OpenManus), Xiaomi, NVIDIA research, [Baichuan](https://www.baichuan-ai.com/home), [RedNote](https://www.xiaohongshu.com/), [SwissAI](https://www.swiss-ai.org/), [Moonshot AI (Kimi)](https://www.moonshot-ai.com/), Baidu, Snowflake, Skywork.ai, JetBrains, [IceSword Lab](https://www.iceswordlab.com), and many more.

<!-- Community Work -->
## Awesome Work Using verl

A curated list of projects built with verl, showcasing its versatility:
*   **TinyZero:** DeepSeek R1 Zero recipe reproduction.
*   **SkyThought:** RL training for Sky-T1-7B.
*   **simpleRL-reason:** Zero Reinforcement Learning.
*   **Easy-R1:** Multi-modal RL training framework.
*   and many more!

<!-- Contribution -->
## Contribution Guide

See the [CONTRIBUTING.md](CONTRIBUTING.md) file for details.

<!-- Contact -->
## About [ByteDance Seed Team](https://team.doubao.com/)

The ByteDance Seed Team is dedicated to pushing the boundaries of AI foundation models.
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
<!-- Hiring -->
We are HIRING! Send us an [email](mailto:haibin.lin@bytedance.com) if you are interested in internship/FTE opportunities in RL for agents.
```
Key improvements and SEO enhancements:

*   **Clear Title and Hook:**  The title is clear and includes a strong one-sentence hook to grab attention.
*   **Structured Headings:**  Uses clear headings (Key Features, Getting Started, News, etc.) for better readability and SEO.
*   **Bulleted Lists:** Uses bulleted lists to highlight key features, making the information easy to scan.
*   **Keywords:**  Incorporates relevant keywords like "Reinforcement Learning," "LLMs," "Large Language Models," and "RLHF" naturally throughout the text.  Also includes names of algorithms and related projects.
*   **Concise Summary:** Provides a concise overview of the library's purpose and benefits.
*   **Links to Key Resources:**  Includes links to the documentation, quickstart, and related papers.
*   **Emphasis on Performance:** Highlights performance benefits (speed, efficiency).
*   **Community Focus:**  Showcases community contributions and examples to build credibility and encourage usage.
*   **Actionable Information:** Provides direct links to documentation and examples to encourage users.
*   **Call to Action:** Encourages users to explore the repository.
*   **SEO-Friendly Badges:** Uses GitHub social badges for stars, twitter follow, documentation, and slack.
*   **Clear Upgrade Instructions:** Provides specific upgrade steps to ensure a smooth user experience.
*   **Contribution Guide:** Links to a contribution guide for easier participation.
*   **Contact Information:** Provides an email for internship and FTE opportunities.
*   **Corrected and improved all links.**
*   **More concise phrasing.**
*   **Improved news section.**
*   **Uses the term "open-source library" in the opening, which is more appropriate for SEO.**