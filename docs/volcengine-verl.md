# verl: Reinforcement Learning for LLMs - Train Your LLM to Excel

verl is a powerful, flexible, and efficient open-source library for Reinforcement Learning (RL) training of Large Language Models (LLMs), enabling you to fine-tune your models and unlock their full potential.

[<img src="https://img.shields.io/github/stars/volcengine/verl?style=social" alt="GitHub stars" />](https://github.com/volcengine/verl)
[![Twitter](https://img.shields.io/twitter/follow/verl_project)](https://twitter.com/verl_project)
[![Documentation](https://img.shields.io/badge/documentation-blue)](https://verl.readthedocs.io/en/latest/)

verl, initiated by the ByteDance Seed team, offers a comprehensive solution for RL training, built upon the foundation of the [HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256v2) paper.

**Key Features:**

*   **Flexible RL Algorithms:** Easily implement and customize various RL algorithms like PPO, GRPO, and more.
*   **Seamless Integration:** Works with existing LLM infrastructure, including FSDP, Megatron-LM, vLLM, and SGLang.
*   **Efficient Resource Utilization:** Supports flexible device mapping for optimal performance and scalability.
*   **Hugging Face Compatibility:** Ready to use with popular Hugging Face models.
*   **State-of-the-Art Performance:** Achieve high throughput for LLM training and inference.
*   **Model support:** Qwen-3, Qwen-2.5, Llama3.1, Gemma2, DeepSeek-LLM, etc

**What's New:**

*   **[verl meetup at ICML Vancouver on July 16th!](https://lu.ma/0ek2nyao) (onsite only)**
*   **verl keynote at [AWS AI Hours Singapore](https://pages.awscloud.com/aws-ai-hours-sg.html#agenda) on 7/8**
*   **verl with Megatron backend enables large MoE models such as [DeepSeek-671b and Qwen3-236b](https://verl.readthedocs.io/en/latest/perf/dpsk.html).**

**Getting Started:**

*   [Documentation](https://verl.readthedocs.io/en/latest/index.html)
*   [Installation Guide](https://verl.readthedocs.io/en/latest/start/install.html)
*   [Quickstart Guide](https://verl.readthedocs.io/en/latest/start/quickstart.html)
*   [PPO Example](https://verl.readthedocs.io/en/latest/examples/ppo_trainer/)
*   [GRPO Example](https://verl.readthedocs.io/en/latest/examples/grpo_trainer/)

**Community Blogs and Resources:**

*   [When Reasoning Models Break Tokenization: The Hidden Complexity of Multiturn Training](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/fast_tokenization/multiturn_tokenization_and_masking.md)
*   [verl deployment on AWS SageMaker](https://medium.com/@kaige.yang0110/run-verl-on-sagemaker-using-4x8-l40s-gpus-8e6d5c3c61d3)
*   [verl x SGLang Multi-turn Code Walkthrough](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/code-walk-through/readme_EN.md)
*   [Optimizing SGLang Memory Usage in verl](https://hebiao064.github.io/rl-memory-management)

**Performance Tuning Guide:**
*   [performance tuning guide](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html)

**Citation:**

If you use verl in your research, please cite:

```bibtex
@article{sheng2024hybridflow,
  title   = {HybridFlow: A Flexible and Efficient RLHF Framework},
  author  = {Guangming Sheng and Chi Zhang and Zilingfeng Ye and Xibin Wu and Wang Zhang and Ru Zhang and Yanghua Peng and Haibin Lin and Chuan Wu},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2409.19256}
}
```

**[Contribute to verl](CONTRIBUTING.md)**

**About ByteDance Seed Team**

The ByteDance Seed Team is dedicated to advancing AI foundation models.

*   [Website](https://team.doubao.com/)
*   [WeChat](https://github.com/user-attachments/assets/469535a8-42f2-4797-acdf-4f7a1d4a0c3e)
*   [Xiaohongshu](https://www.xiaohongshu.com/user/profile/668e7e15000000000303157d?xsec_token=ABl2-aqekpytY6A8TuxjrwnZskU-6BsMRE_ufQQaSAvjc%3D&xsec_source=pc_search)
*   [Zhihu](https://www.zhihu.com/org/dou-bao-da-mo-xing-tuan-dui/)

---

**[Learn More and Contribute to verl on GitHub](https://github.com/volcengine/verl)**