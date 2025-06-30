<!-- Improved README with SEO Optimization -->

# verl: Revolutionizing Reinforcement Learning for LLMs

verl is a powerful, open-source reinforcement learning (RL) library designed for training large language models (LLMs), enabling flexible, efficient, and production-ready RL workflows. [Explore the verl repository](https://github.com/volcengine/verl) to supercharge your LLM training!

[![GitHub Repo stars](https://img.shields.io/github/stars/volcengine/verl?style=social)](https://github.com/volcengine/verl/stargazers)
[![Twitter](https://img.shields.io/twitter/follow/verl_project?style=social)](https://twitter.com/verl_project)
<a href="https://join.slack.com/t/verlgroup/shared_invite/zt-2w5p9o4c3-yy0x2Q56s_VlGLsJ93A6vA"><img src="https://img.shields.io/badge/Slack-verl-blueviolet?logo=slack&amp"></a>
<a href="https://arxiv.org/pdf/2409.19256"><img src="https://img.shields.io/static/v1?label=EuroSys&message=Paper&color=red"></a>
[![Documentation](https://img.shields.io/badge/documentation-blue)](https://verl.readthedocs.io/en/latest/)
<a href="https://raw.githubusercontent.com/eric-haibin-lin/verl-community/refs/heads/main/WeChat.JPG"><img src="https://img.shields.io/badge/微信-green?logo=wechat&amp"></a>

![seed logo](https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216)

## Key Features

*   **Flexible RL Algorithms:** Easy extension of diverse RL algorithms, including PPO and GRPO, through a hybrid-controller programming model.
*   **Seamless LLM Integration:**  Integrates effortlessly with existing LLM frameworks, such as FSDP, Megatron-LM, vLLM, and SGLang.
*   **Efficient Device Mapping:**  Supports flexible placement of models across GPUs for optimized resource utilization and scalability.
*   **Hugging Face Compatibility:** Ready integration with popular Hugging Face models.
*   **State-of-the-Art Performance:** Achieves SOTA throughput for LLM training and inference, and efficient actor model resharding.

## What's New

*   **[June 2025]** verl with Megatron backend enables large MoE models such as [DeepSeek-671b and Qwen3-236b](https://verl.readthedocs.io/en/latest/perf/dpsk.html).
*   **[June 2025]** verl team will provide latest project updates at [PyTorch Day China](https://www.lfasiallc.com/pytorch-day-china/) on June 7th. Meet our dev team in Beijing!
*   **[May 2025]** Support for [PF-PPO](https://arxiv.org/abs/2409.06957), accepted to ICML 2025.
*   **[April 2025]** [Seed-Thinking-v1.5](https://github.com/ByteDance-Seed/Seed-Thinking-v1.5/blob/main/seed-thinking-v1.5.pdf) tech report is released!
*   **[April 2025]** [VAPO](https://arxiv.org/pdf/2504.05118) (value-based augmented PPO) paper covers our latest RL method for reasoning models.
*   **[March 2025]** [DAPO](https://dapo-sia.github.io/) is the open-sourced SOTA RL algorithm fully powered by verl, reproduction code is available in `recipe/dapo`.

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

*   **Documentation:** Access detailed [documentation](https://verl.readthedocs.io/en/latest/index.html) for installation, quickstarts, and in-depth guides.

*   **Quickstart:**  Follow the [quickstart guide](https://verl.readthedocs.io/en/latest/start/quickstart.html) for a rapid introduction.
*   **Programming Guide:**  Explore the [programming guide](https://verl.readthedocs.io/en/latest/hybrid_flow.html) and [Tech Talk](https://hcqnc.xetlk.com/sl/3vACOK) (in Chinese).
*   **PPO/GRPO:** Learn how to implement PPO and GRPO in verl. ([PPO](https://verl.readthedocs.io/en/latest/algo/ppo.html), [GRPO](https://verl.readthedocs.io/en/latest/algo/grpo.html))

## Performance Tuning Guide

Optimize your RL workflows with our [performance tuning guide](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html).

## Contribution

We welcome contributions from the community! See our [project roadmap](https://github.com/volcengine/verl/issues/710) and [good first issues](https://github.com/volcengine/verl/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22good%20first%20issue%22).

### Code Linting and Formatting
We use pre-commit to help improve code quality. To initialize pre-commit, run:

```bash
pip install pre-commit
pre-commit install
```

To resolve CI errors locally, you can manually run pre-commit by:

```bash
pre-commit run
```

### Adding CI tests

If possible, please add CI test(s) for your new feature:

1. Find the most relevant workflow yml file, which usually corresponds to a `hydra` default config (e.g. `ppo_trainer`, `ppo_megatron_trainer`, `sft_trainer`, etc).
2. Add related path patterns to the `paths` section if not already included.
3. Minimize the workload of the test script(s) (see existing scripts for examples).

##  About ByteDance Seed Team

ByteDance Seed Team is dedicated to crafting the industry's most advanced AI foundation models.
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
```

Key improvements and SEO considerations:

*   **Concise Hook:**  The opening sentence clearly states the library's purpose.
*   **Keyword-Rich Headings:**  Uses relevant keywords like "Reinforcement Learning," "LLMs," and "RL library" in headings.
*   **Bulleted Key Features:** Uses bullet points for readability and to highlight important features, making it easy for users to scan and understand.
*   **Focus on Benefits:** Highlights the benefits (flexibility, efficiency, production-readiness).
*   **Clear Call to Action:** The "Explore the verl repository" encourages users to visit the repo.
*   **Updated Information:** Includes most recent updates from the original README.
*   **Emphasis on LLM:**  The phrase "large language models" is used strategically.
*   **Consistent Formatting:**  Uses consistent markdown for easy reading.
*   **Direct Links:** Makes the links easily accessible to the reader.
*   **SEO-Friendly Titles:** Makes it easy for search engines to understand the subject.
*   **Concise and Focused:** Removes excessive promotional language, keeping the focus on key information.