<div align="center">
  <h1>verl: Unleash the Power of RL for LLMs</h1>
  <p><b>verl</b>, initiated by the ByteDance Seed team, is a versatile and efficient reinforcement learning (RL) training library designed for large language models (LLMs), accelerating research and production.</p>
  <p>
    <a href="https://github.com/volcengine/verl">
      <img src="https://img.shields.io/github/stars/volcengine/verl?style=social" alt="GitHub stars">
    </a>
    <a href="https://twitter.com/verl_project">
      <img src="https://img.shields.io/twitter/follow/verl_project?style=social" alt="Follow on Twitter">
    </a>
    <a href="https://join.slack.com/t/verlgroup/shared_invite/zt-2w5p9o4c3-yy0x2Q56s_VlGLsJ93A6vA">
      <img src="https://img.shields.io/badge/Slack-verl-blueviolet?logo=slack&amp" alt="Join Slack">
    </a>
    <a href="https://verl.readthedocs.io/en/latest/">
      <img src="https://img.shields.io/badge/documentation-blue" alt="Documentation">
    </a>
    <a href="https://arxiv.org/pdf/2409.19256">
      <img src="https://img.shields.io/static/v1?label=EuroSys&message=Paper&color=red" alt="EuroSys Paper">
    </a>
    <a href="https://raw.githubusercontent.com/eric-haibin-lin/verl-community/refs/heads/main/WeChat.JPG">
      <img src="https://img.shields.io/badge/微信-green?logo=wechat&amp" alt="WeChat">
    </a>
  </p>
</div>

<div align="center">
  <a href="https://deepwiki.com/volcengine/verl">
    <img src="https://devin.ai/assets/deepwiki-badge.png" alt="Ask DeepWiki.com" height="20"/>
  </a>
</div>


<p align="center">
  <img src="https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216" alt="Seed Logo" width="100">
</p>

## Key Features

*   🚀 **Flexible and Extensible:** Easily integrate and extend RL algorithms, including PPO, GRPO, and more, with the hybrid-controller programming model.
*   🔗 **Seamless Integration:**  Integrates with popular LLM frameworks like FSDP, Megatron-LM, vLLM, and SGLang, decoupling computation and data dependencies.
*   ⚙️ **Efficient Resource Utilization:** Supports flexible device mapping for optimized resource utilization and scalability across various cluster sizes.
*   🤗 **Hugging Face Compatibility:** Ready-to-use with Hugging Face models for simplified training and deployment.
*   ⚡️ **State-of-the-Art Performance:** Achieves SOTA LLM training and inference throughput.
*   🔄 **Efficient Resharding:** 3D-HybridEngine eliminates memory redundancy and reduces communication overhead.

## News & Updates

*   **[2025/06]**: verl supports MoE models like DeepSeek-671b and Qwen3-236b using Megatron backend.
*   **[2025/06]**: verl team at PyTorch Day China.
*   **[2025/05]**: Support for PF-PPO, enhancing policy learning efficiency and robustness.
*   **[2025/04]**: Seed-Thinking-v1.5 tech report released, demonstrating strong reasoning abilities.
*   **[2025/04]**: VAPO paper released, introducing a new RL method for reasoning models.
*   **[2025/03]**: DAPO is supported in verl and the reproduction code is available in `recipe/dapo`.
*   **[2025/03]**: verl v0.3.0.post1 is released, achieving ~1.4x speedup.

<details><summary> more...</summary>
<ul>
    <li>[2025/04] verl tutorial at ICLR 2025 Expo, SCI-FM workshop, and LMSys afterparty.</li>
    <li>[2025/05] verl at A2M Shanghai.</li>
    <li>[2025/05] verl at GOSIM x PyTorch Day 2025.</li>
    <li>[2025/03] verl programming model at vLLM Beijing Meetup and SGLang-LMSYS Org Meetup.</li>
    <li>[2025/03] verl (HybridFlow) at EuroSys 2025.</li>
    <li>[2025/02] verl v0.2.0.post2 is released!</li>
    <li>[2025/02] verl at Bytedance/NVIDIA/Anyscale Ray Meetup.</li>
    <li>[2025/01] Doubao-1.5-pro released, RL scaling preview model trained using verl, reaching OpenAI O1-level performance.</li>
    <li>[2024/12] verl is presented at Ray Forward 2024.</li>
    <li>[2024/12] The team presented Post-training LLMs: From Algorithms to Infrastructure at NeurIPS 2024.</li>
    <li>[2024/10] verl is presented at Ray Summit.</li>
    <li>[2024/08] HybridFlow (verl) is accepted to EuroSys 2025.</li>
</ul>
</details>

## Getting Started

Access comprehensive resources to begin your RL journey with verl:

*   📚 **Documentation:** [verl Documentation](https://verl.readthedocs.io/en/latest/index.html)
*   🚀 **Quickstart:** [Installation](https://verl.readthedocs.io/en/latest/start/install.html), [Quickstart](https://verl.readthedocs.io/en/latest/start/quickstart.html)
*   💻 **Programming Guide:** [Hybrid Flow](https://verl.readthedocs.io/en/latest/hybrid_flow.html)
*   👨‍🏫 **Tutorials:** [PPO in verl](https://verl.readthedocs.io/en/latest/algo/ppo.html), [GRPO in verl](https://verl.readthedocs.io/en/latest/algo/grpo.html)
*   🔬 **Reproducible Baselines:** [RL performance on coding, math](https://verl.readthedocs.io/en/latest/algo/baseline.html)

## Performance Tuning

Optimize your RL training with our [performance tuning guide](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html).

## Upgrades & Compatibility

*   **vLLM:** Supports vLLM >= 0.8.2; avoid 0.7.x due to potential issues.  Refer to [this document](https://github.com/volcengine/verl/blob/main/docs/README_vllm0.8.md) for more details.
*   **SGLang:** Fully supported.  See [this document](https://verl.readthedocs.io/en/latest/workers/sglang_worker.html) for integration details.
*   **FSDP2:**  Embrace FSDP2 for improved throughput and memory efficiency. Enable with  `actor_rollout_ref.ref.strategy=fsdp2`, `actor_rollout_ref.actor.strategy=fsdp2`, `critic.strategy=fsdp2`, and `reward_model.strategy=fsdp2`.

## AMD Support

verl now supports FSDP as the training engine and integrates with vLLM and SGLang for inference on AMD GPUs.  See the installation guide [here](https://github.com/volcengine/verl/blob/main/docs/amd_tutorial/amd_build_dockerfile_page.rst) and vLLM tuning [here](https://github.com/volcengine/verl/blob/main/docs/amd_tutorial/amd_vllm_page.rst).

## Citation

If verl has been helpful, please cite the following papers:

```bibtex
@article{sheng2024hybridflow,
  title   = {HybridFlow: A Flexible and Efficient RLHF Framework},
  author  = {Guangming Sheng and Chi Zhang and Zilingfeng Ye and Xibin Wu and Wang Zhang and Ru Zhang and Yanghua Peng and Haibin Lin and Chuan Wu},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2409.19256}
}
```

## Acknowledgements

verl is inspired by Nemo-Aligner, Deepspeed-chat, and OpenRLHF and is supported by various contributors from Bytedance, Anyscale, LMSys.org, and many more.

## Awesome work using verl

*   [TinyZero](https://github.com/Jiayi-Pan/TinyZero)
*   [SkyThought](https://github.com/NovaSky-AI/SkyThought)
*   [simpleRL-reason](https://github.com/hkust-nlp/simpleRL-reason)
*   [Easy-R1](https://github.com/hiyouga/EasyR1)
*   [OpenManus-RL](https://github.com/OpenManus/OpenManus-RL)
*   [rllm](https://github.com/agentica-project/rllm)
*   [RAGEN](https://github.com/ZihanWang314/ragen)
*   [Search-R1](https://github.com/PeterGriffinJin/Search-R1)
*   [ReSearch](https://github.com/Agent-RL/ReSearch)
*   [Skywork-OR1](https://github.com/SkyworkAI/Skywork-OR1)
*   [ToRL](https://github.com/GAIR-NLP/ToRL)
*   [Absolute Zero Reasoner](https://github.com/LeapLabTHU/Absolute-Zero-Reasoner)
*   [verl-agent](https://github.com/langfengQ/verl-agent)
*   [RL-Factory](https://github.com/Simple-Efficient/RL-Factory)
*   [ReTool](https://retool-rl.github.io/)
*   [verl-tool](https://github.com/TIGER-AI-Lab/verl-tool)
*   [PRIME](https://github.com/PRIME-RL/PRIME)
*   [GUI-R1](https://github.com/ritzz-ai/GUI-R1)
*   [DeepRetrieval](https://github.com/pat-jj/DeepRetrieval)
*   [Code-R1](https://github.com/ganler/code-r1)
*   [DeepResearcher](https://github.com/GAIR-NLP/DeepResearcher)
*   [VAGEN](https://github.com/RAGEN-AI/VAGEN)
*   [RM-R1](https://arxiv.org/abs/2505.02387)
*   [LUFFY](https://arxiv.org/pdf/2504.14945)
*   [DeepMath](https://github.com/zwhe99/DeepMath)
*   [Entropy Mechanism of RL](https://github.com/PRIME-RL/Entropy-Mechanism-of-RL)
*   [LLaSA-TTS-GRPO](https://github.com/channel-io/ch-tts-llasa-rl-grpo)
*   [PF-PPO](https://arxiv.org/abs/2409.06957)
*   [RACRO](https://github.com/gyhdog99/RACRO2)

and many more awesome work listed in [recipe](recipe/README.md).

## Contribution

We welcome contributions! See our [contribution guide](CONTRIBUTING.md).

## About ByteDance Seed Team

[ByteDance Seed Team](https://team.doubao.com/) is dedicated to advancing AI foundation models.
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
  **Join our team!** We are hiring for RL positions. Contact us via [email](mailto:haibin.lin@bytedance.com).
```
Key improvements and SEO enhancements:

*   **Strong Hook:** The first sentence directly states the library's purpose and benefit, using keywords.
*   **Clear Headings:** Uses headings (H1, H2) for better organization and SEO.
*   **Bulleted Key Features:** Uses bullet points to highlight the most important features, making the information easy to scan and understand.
*   **Keyword Optimization:** Includes relevant keywords like "Reinforcement Learning," "LLMs," "RLHF," "large language models," and specific algorithm names.
*   **Community & Contributions:** Highlights the supportive community and welcomes contributions.
*   **Up-to-date News Section:** Keeps the information current to keep users engaged.
*   **Actionable:** Adds actionable call-to-action links to relevant sections.
*   **Contact Info:** Highlights ways to contact the team, including hiring information.
*   **Clear & Concise:** Improves the clarity and conciseness of the original README.
*   **Social Media Links:** Adds social media buttons to enable easy engagement.