<div align="center">
  <h1>verl: Your Go-To Library for Efficient RL Training of Large Language Models</h1>
  <p>verl is a powerful, open-source reinforcement learning (RL) training library designed by ByteDance Seed team for large language models (LLMs), built for flexibility, efficiency, and production readiness.</p>
  <br>
  <a href="https://github.com/volcengine/verl">
    <img src="https://img.shields.io/github/stars/volcengine/verl?style=social" alt="GitHub stars">
  </a>
  <a href="https://twitter.com/verl_project">
    <img src="https://img.shields.io/twitter/follow/verl_project?style=social" alt="Twitter">
  </a>
  <a href="https://join.slack.com/t/verlgroup/shared_invite/zt-2w5p9o4c3-yy0x2Q56s_VlGLsJ93A6vA">
    <img src="https://img.shields.io/badge/Slack-verl-blueviolet?logo=slack&amp" alt="Slack">
  </a>
  <a href="https://arxiv.org/pdf/2409.19256">
    <img src="https://img.shields.io/static/v1?label=EuroSys&message=Paper&color=red" alt="EuroSys Paper">
  </a>
  <a href="https://verl.readthedocs.io/en/latest/">
    <img src="https://img.shields.io/badge/documentation-blue" alt="Documentation">
  </a>
  <a href="https://raw.githubusercontent.com/eric-haibin-lin/verl-community/refs/heads/main/WeChat.JPG">
    <img src="https://img.shields.io/badge/微信-green?logo=wechat&amp" alt="WeChat">
  </a>
</div>

[<img src="https://devin.ai/assets/deepwiki-badge.png" alt="Ask DeepWiki.com" height="20"/>](https://deepwiki.com/volcengine/verl)

<img src="https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216" alt="Seed Team Logo" style="display: block; margin: 0 auto;">

**Key Features:**

*   **Flexible RL Algorithms:** Easily implement diverse RL algorithms like PPO, GRPO, and DAPO with verl's hybrid-controller programming model.
*   **Seamless LLM Integration:** Compatible with existing LLM frameworks such as FSDP, vLLM, and Hugging Face Transformers.
*   **Efficient Device Mapping:** Supports flexible model placement across GPUs for optimal resource utilization and scalability.
*   **Hugging Face Compatibility:** Ready-to-use with popular Hugging Face models.
*   **State-of-the-Art Performance:** Achieve SOTA LLM training and inference speeds.
*   **Advanced Features:** Includes LoRA, Flash Attention 2, and Sequence Packing/Parallelism for efficient training.
*   **Model Support:** Compatible with Qwen-3, Qwen-2.5, Llama3.1, Gemma2, DeepSeek-LLM, and more.
*   **Alignment Recipes:**  Supports LLM alignment methods such as Self-play preference optimization (SPPO).
*   **Expert Parallelism:** Scales to models up to 671B parameters and hundreds of GPUs.
*   **Comprehensive Support:** Experiment tracking with wandb, swanlab, mlflow and tensorboard.

**[Learn More and Contribute on GitHub](https://github.com/volcengine/verl)**

**News:**

*   [2025/06] verl with Megatron backend enables large MoE models such as [DeepSeek-671b and Qwen3-236b](https://verl.readthedocs.io/en/latest/perf/dpsk.html).
*   [2025/06] verl team will provide latest project updates at [PyTorch Day China](https://www.lfasiallc.com/pytorch-day-china/) on June 7th. Meet our dev team in Beijing!
*   [2025/05] [PF-PPO](https://arxiv.org/abs/2409.06957), accepted to ICML 2025, is now supported in verl! PF-PPO enhances policy learning efficiency and robustness by filtering potentially noisy reward signals and reusing high-quality experiences via a replay buffer.
*   [2025/04] [Seed-Thinking-v1.5](https://github.com/ByteDance-Seed/Seed-Thinking-v1.5/blob/main/seed-thinking-v1.5.pdf) tech report is released! Trained with verl, Seed-Thinking-v1.5 achieves 86.7 on AIME 2024, 55.0 on Codeforces and 77.3 on GPQA, demonstrating excellent reasoning abilities in STEM and coding. Beyond reasoning tasks, the method demonstrates notable generalization across diverse domains.
*   [2025/04] [VAPO](https://arxiv.org/pdf/2504.05118) (value-based augmented PPO) paper covers our latest RL method for reasoning models. Trained from Qwen-32B-base model, VAPO achieves 60.4 on AIME 2024, outperforming DAPO-32B.
*   [2025/03] [DAPO](https://dapo-sia.github.io/) is the open-sourced SOTA RL algorithm that achieves 50 points on AIME 2024 based on the Qwen2.5-32B pre-trained model, surpassing the previous SOTA achieved by DeepSeek's GRPO (DeepSeek-R1-Zero-Qwen-32B). DAPO's training is fully powered by verl and the reproduction code is available in `recipe/dapo` now.
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

**Getting Started:**

*   [Documentation](https://verl.readthedocs.io/en/latest/index.html)
*   [Installation](https://verl.readthedocs.io/en/latest/start/install.html)
*   [Quickstart](https://verl.readthedocs.io/en/latest/start/quickstart.html)
*   [Programming Guide](https://verl.readthedocs.io/en/latest/hybrid_flow.html)
*   [PPO in verl](https://verl.readthedocs.io/en/latest/algo/ppo.html)
*   [GRPO in verl](https://verl.readthedocs.io/en/latest/algo/grpo.html)

**Performance Tuning:**

*   [Performance Tuning Guide](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html)

**Key Updates:**

*   **vLLM Compatibility:** Supports vLLM >= v0.8.2.
*   **SGLang Integration:**  Full support for SGLang, enhancing multi-turn and agentic RL capabilities.
*   **FSDP2 Support:** Now fully embracing FSDP2 for enhanced performance and memory efficiency.
*   **AMD Support:** Includes ROCm Kernel for AMD GPU support.

**Citation:**

```bibtex
@article{sheng2024hybridflow,
  title   = {HybridFlow: A Flexible and Efficient RLHF Framework},
  author  = {Guangming Sheng and Chi Zhang and Zilingfeng Ye and Xibin Wu and Wang Zhang and Ru Zhang and Yanghua Peng and Haibin Lin and Chuan Wu},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2409.19256}
}
```

**Acknowledgements:**

verl is inspired by Nemo-Aligner, Deepspeed-chat, and OpenRLHF and supported by Bytedance, Anyscale, LMSys.org, [Alibaba Qwen team](https://github.com/QwenLM/), Shanghai AI Lab, Tsinghua University, UC Berkeley, UCLA, UIUC, University of Hong Kong, ke.com, [All Hands AI](https://www.all-hands.dev/), [ModelBest](http://modelbest.cn/), JD AI Lab, Microsoft Research, [StepFun](https://www.stepfun.com/), Amazon, LinkedIn, Meituan, [Camel-AI](https://www.camel-ai.org/), [OpenManus](https://github.com/OpenManus), Xiaomi, NVIDIA research, [Baichuan](https://www.baichuan-ai.com/home), [RedNote](https://www.xiaohongshu.com/), [SwissAI](https://www.swiss-ai.org/), [Moonshot AI (Kimi)](https://www.moonshot-ai.com/), Baidu, Snowflake, Skywork.ai, JetBrains, [IceSword Lab](https://www.iceswordlab.com), and many more.

**Awesome Work Using verl:**

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

**Contribution Guide:**

*   [Contribution Guide](CONTRIBUTING.md)

**About ByteDance Seed Team:**
The ByteDance Seed Team is dedicated to crafting the industry's most advanced AI foundation models.
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