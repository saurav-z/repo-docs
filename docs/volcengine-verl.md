<div align="center">
    <h1 style="text-align: center;">verl: Train LLMs Efficiently with Reinforcement Learning</h1>

    <p>verl is a flexible and efficient reinforcement learning (RL) training library for large language models (LLMs), designed to optimize your models for real-world applications. <a href="https://github.com/volcengine/verl">Explore verl on GitHub</a></p>
    <br>
</div>

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

## Key Features

*   **Flexible RL Algorithms:** Easily implement and extend diverse RL algorithms like PPO, GRPO, and more.
*   **Seamless LLM Integration:** Compatible with popular LLM frameworks (FSDP, Megatron-LM, vLLM, SGLang) and Hugging Face models.
*   **Efficient Training:** Achieve state-of-the-art throughput for LLM training.
*   **Modular APIs:** Decoupled computation and data dependencies make verl easy to use.
*   **Flexible Device Mapping:** Supports various GPU configurations for efficient resource utilization.
*   **Advanced Features:** Support for Multi-GPU LoRA RL, Expert Parallelism, Sequence packing, and more.
*   **Comprehensive Support:** Compatible with  Qwen-3, Qwen-2.5, Llama3.1, Gemma2, DeepSeek-LLM, etc.

## Why Use verl?

verl empowers you to fine-tune LLMs efficiently with a focus on flexibility, efficiency, and production readiness. Built by the ByteDance Seed team, verl is the open-source implementation of the "HybridFlow: A Flexible and Efficient RLHF Framework" paper and is designed to accelerate your RLHF projects.

## Getting Started

*   **Documentation:** [verl Documentation](https://verl.readthedocs.io/en/latest/index.html)
*   **Quickstart:**
    *   [Installation Guide](https://verl.readthedocs.io/en/latest/start/install.html)
    *   [Quickstart Guide](https://verl.readthedocs.io/en/latest/start/quickstart.html)
    *   [Programming Guide](https://verl.readthedocs.io/en/latest/hybrid_flow.html) & [Tech Talk](https://hcqnc.xetlk.com/sl/3vACOK) (in Chinese)
    *   [PPO in verl](https://verl.readthedocs.io/en/latest/algo/ppo.html)
    *   [GRPO in verl](https://verl.readthedocs.io/en/latest/algo/grpo.html)
*   **Running a PPO example step-by-step:**
    *   [Prepare Data for Post-Training](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html)
    *   [Implement Reward Function for Dataset](https://verl.readthedocs.io/en/latest/preparation/reward_function.html)
    *   [PPO Example Architecture](https://verl.readthedocs.io/en/latest/examples/ppo_code_architecture.html)
    *   [Config Explanation](https://verl.readthedocs.io/en/latest/examples/config.html)

## News

*   [2025/07] The [ReTool](https://arxiv.org/pdf/2504.11536) recipe is fully open sourced. [Blog](https://www.notion.so/verl-reTool-recipe-Using-multi-round-conversations-and-code-sandboxing-to-improve-the-math-of-large-23a8b5b7feba80b386b2e5b5e3c1cde0)
*   [2025/07] The first verl meetup will be held at ICML Vancouver on July 16th! Please [join us](https://lu.ma/0ek2nyao) if you are at ICML! (onsite only)
*   [2025/07] verl keynote at [AWS AI Hours Singapore](https://pages.awscloud.com/aws-ai-hours-sg.html#agenda) on 7/8, verl & verl-agent project updates at [Agent for SWE meetup](https://lu.ma/e498qhsi) by LF AI & Data Singapore on 7/11.
*   [2025/06] verl with Megatron backend enables large MoE models such as [DeepSeek-671b and Qwen3-236b](https://verl.readthedocs.io/en/latest/perf/dpsk.html).
*   [2025/06] verl team will provide latest project updates at [PyTorch Day China](https://www.lfasiallc.com/pytorch-day-china/) on June 7th. Meet our dev team in Beijing!
*   [2025/04] [Seed-Thinking-v1.5](https://github.com/ByteDance-Seed/Seed-Thinking-v1.5/blob/main/seed-thinking-v1.5.pdf) tech report is released! Trained with verl, Seed-Thinking-v1.5 achieves 86.7 on AIME 2024, 55.0 on Codeforces and 77.3 on GPQA, demonstrating excellent reasoning abilities in STEM and coding. Beyond reasoning tasks, the method demonstrates notable generalization across diverse domains.
*   [2025/03] [DAPO](https://dapo-sia.github.io/) is the open-sourced SOTA RL algorithm that achieves 50 points on AIME 2024 based on the Qwen2.5-32B pre-trained model, surpassing the previous SOTA achieved by DeepSeek's GRPO (DeepSeek-R1-Zero-Qwen-32B). DAPO's training is fully powered by verl and the reproduction code is available in `recipe/dapo` now.
<details><summary> more... </summary>
<ul>
  <li> [2025/04] [VAPO](https://arxiv.org/pdf/2504.05118) (value-based augmented PPO) paper covers our latest RL method for reasoning models. Trained from Qwen-32B-base model, VAPO achieves 60.4 on AIME 2024, outperforming DAPO-32B.</li>
  <li>[2025/05] [PF-PPO](https://arxiv.org/abs/2409.06957), accepted to ICML 2025, is now supported in verl! PF-PPO enhances policy learning efficiency and robustness by filtering potentially noisy reward signals and reusing high-quality experiences via a replay buffer.</li>
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

## Performance Tuning Guide

For optimal performance, consult our [Performance Tuning Guide](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html).

## Upgrading and Compatibility

*   **vLLM:**  Ensure you use vLLM >= v0.8.2 for FSDP compatibility. See  [vLLM 0.8.2 Docs](https://github.com/volcengine/verl/blob/main/docs/README_vllm0.8.md).
*   **SGLang:** Leverage the latest SGLang version for multi-turn agentic RL, VLM RLHF, and more.  [SGLang worker guide](https://verl.readthedocs.io/en/latest/workers/sglang_worker.html).
*   **FSDP2:** Utilize FSDP2 for improved throughput and memory efficiency. Set `actor_rollout_ref.ref.strategy=fsdp2`, `actor_rollout_ref.actor.strategy=fsdp2`, `critic.strategy=fsdp2` and `reward_model.strategy=fsdp2` in your config.

## AMD Support

verl provides support for AMD GPUs with FSDP (and soon Megatron) and integrates with vLLM and SGLang. See the following guides:

*   [AMD Installation Guide](https://github.com/volcengine/verl/blob/main/docs/amd_tutorial/amd_build_dockerfile_page.rst)
*   [vLLM Performance Tuning for ROCm](https://github.com/volcengine/verl/blob/main/docs/amd_tutorial/amd_vllm_page.rst)

## Citation

```bibtex
@article{sheng2024hybridflow,
  title   = {HybridFlow: A Flexible and Efficient RLHF Framework},
  author  = {Guangming Sheng and Chi Zhang and Zilingfeng Ye and Xibin Wu and Wang Zhang and Ru Zhang and Yanghua Peng and Haibin Lin and Chuan Wu},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2409.19256}
}
```

## Acknowledgements

verl is inspired by projects like Nemo-Aligner, Deepspeed-chat, and OpenRLHF. It is adopted and contributed to by various organizations and researchers.

## Awesome verl Projects

(List of awesome projects using verl from original)

## Contribution Guide

See [contributions guide](CONTRIBUTING.md)

## About [ByteDance Seed Team](https://team.doubao.com/)

(ByteDance Seed Team section from original)

---
We are HIRING! Send us an [email](mailto:haibin.lin@bytedance.com) if you are interested in internship/FTE opportunities in RL for agents.