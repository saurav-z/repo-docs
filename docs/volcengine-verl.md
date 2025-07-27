<div align="center">
  <h1>verl: Train Powerful LLMs with Flexible, Efficient Reinforcement Learning</h1>
  <p>Supercharge your large language models with verl, the open-source RL training library from the ByteDance Seed team.</p>
</div>

<div align="center">
  <a href="https://deepwiki.com/volcengine/verl"><img src="https://devin.ai/assets/deepwiki-badge.png" alt="Ask DeepWiki.com" style="height:20px;"></a>
  <a href="https://github.com/volcengine/verl/stargazers"><img src="https://img.shields.io/github/stars/volcengine/verl?style=social" alt="GitHub stars"></a>
  <a href="https://twitter.com/verl_project"><img src="https://img.shields.io/twitter/follow/verl_project?style=social" alt="Follow verl on Twitter"></a>
  <a href="https://join.slack.com/t/verlgroup/shared_invite/zt-2w5p9o4c3-yy0x2Q56s_VlGLsJ93A6vA"><img src="https://img.shields.io/badge/Slack-verl-blueviolet?logo=slack&amp" alt="Join verl Slack"></a>
  <a href="https://arxiv.org/pdf/2409.19256"><img src="https://img.shields.io/static/v1?label=EuroSys&message=Paper&color=red" alt="EuroSys Paper"></a>
  <a href="https://verl.readthedocs.io/en/latest/"><img src="https://img.shields.io/badge/documentation-blue" alt="Documentation"></a>
  <a href="https://raw.githubusercontent.com/eric-haibin-lin/verl-community/refs/heads/main/WeChat.JPG"><img src="https://img.shields.io/badge/微信-green?logo=wechat&amp" alt="WeChat"></a>
</div>

<img src="https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216" alt="Seed Logo" style="display: block; margin: 0 auto; max-width: 100px;">

<hr>

## Key Features of verl

*   **Flexible RL Algorithms:** Easily extend and implement diverse RL algorithms like PPO, GRPO, and DAPO with the hybrid-controller programming model.
*   **Seamless Integration:** Integrate with existing LLM infrastructure, including FSDP, Megatron-LM, vLLM, and SGLang.
*   **Efficient Resource Utilization:** Flexible device mapping for efficient resource utilization and scalability across various cluster sizes.
*   **Hugging Face Compatibility:** Ready integration with popular Hugging Face models.
*   **State-of-the-Art Performance:** Achieve SOTA LLM training and inference throughput.

## News and Updates

*   [2025/07] ReTool recipe fully open sourced, and first verl meetup at ICML Vancouver.
*   [2025/06] verl enables large MoE models such as [DeepSeek-671b and Qwen3-236b](https://verl.readthedocs.io/en/latest/perf/dpsk.html).
*   [2025/04]  [Seed-Thinking-v1.5](https://github.com/ByteDance-Seed/Seed-Thinking-v1.5/blob/main/seed-thinking-v1.5.pdf) tech report released demonstrating strong reasoning abilities.
*   [2025/03] [DAPO](https://dapo-sia.github.io/) is the open-sourced SOTA RL algorithm that achieves 50 points on AIME 2024 powered by verl.
*   ... (See the original README for more updates)

## Getting Started

Explore the power of verl with these resources:

*   [Documentation](https://verl.readthedocs.io/en/latest/index.html)
*   [Installation Guide](https://verl.readthedocs.io/en/latest/start/install.html)
*   [Quickstart Tutorial](https://verl.readthedocs.io/en/latest/start/quickstart.html)
*   [Programming Guide](https://verl.readthedocs.io/en/latest/hybrid_flow.html) & [Tech Talk](https://hcqnc.xetlk.com/sl/3vACOK) (in Chinese)
*   [PPO in verl](https://verl.readthedocs.io/en/latest/algo/ppo.html)
*   [GRPO in verl](https://verl.readthedocs.io/en/latest/algo/grpo.html)

Explore the full potential of verl by referencing [the original repository](https://github.com/volcengine/verl).

## Performance Tuning and Advanced Usage

*   [Performance Tuning Guide](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html)
*   [Advanced Usage and Extension](https://verl.readthedocs.io/en/latest/advance/fsdp_extension.html)
*   [Multi-turn Rollout Support](https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html)

## Citation and Acknowledgements

If you use verl in your research, please cite the following:

```bibtex
@article{sheng2024hybridflow,
  title   = {HybridFlow: A Flexible and Efficient RLHF Framework},
  author  = {Guangming Sheng and Chi Zhang and Zilingfeng Ye and Xibin Wu and Wang Zhang and Ru Zhang and Yanghua Peng and Haibin Lin and Chuan Wu},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2409.19256}
}
```

verl is inspired by and builds upon the work of many, including Nemo-Aligner, Deepspeed-chat, and OpenRLHF. This project is contributed by Bytedance, Anyscale, LMSys.org, and many more.

## Awesome Work Using verl

(See the original README for a list of projects)

## Contribution Guide

See the [contributions guide](CONTRIBUTING.md) for details.

## About ByteDance Seed Team

ByteDance Seed Team is dedicated to crafting the industry's most advanced AI foundation models. You can learn more at [ByteDance Seed's website](https://team.doubao.com/).

---

We are HIRING! Send us an [email](mailto:haibin.lin@bytedance.com) if you are interested in internship/FTE opportunities in RL for agents.