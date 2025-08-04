<div align="center">
 <a href="https://github.com/volcengine/verl">
  <img src="https://raw.githubusercontent.com/volcengine/verl/main/seed.png" alt="verl Logo" width="200">
 </a>
  <h1>verl: Unleash the Power of Reinforcement Learning for LLMs</h1>
  <p>verl is a powerful, flexible, and production-ready RL training library for large language models (LLMs), empowering you to fine-tune and optimize your models with ease.</p>
</div>

[![GitHub Repo stars](https://img.shields.io/github/stars/volcengine/verl?style=social)](https://github.com/volcengine/verl/stargazers)
[![Twitter](https://img.shields.io/twitter/follow/verl_project?style=social)](https://twitter.com/verl_project)
[![Documentation](https://img.shields.io/badge/documentation-blue)](https://verl.readthedocs.io/en/latest/)

## Key Features

*   **Flexible RL Algorithms:** Easily implement diverse RL algorithms like PPO, GRPO, and more with verl's modular design.
*   **Seamless LLM Integration:** Works with popular LLM frameworks (FSDP, Megatron-LM, vLLM, SGLang, Hugging Face Transformers) for effortless integration.
*   **Efficient Resource Utilization:** Supports flexible device mapping for optimal GPU utilization and scalability.
*   **High Performance:** Achieve state-of-the-art throughput through SOTA LLM training integrations.
*   **Extensive Model Support:** Compatible with a wide range of Hugging Face models, including Qwen, Llama3, Gemma, and DeepSeek-LLM.
*   **Multi-Modal RL:** Supports vision-language models (VLMs) for training, including Qwen2.5-vl and Kimi-VL.
*   **Alignment Recipes**: Includes alignment recipes like Self-play preference optimization (SPPO).
*   **Advanced Features:** Supports Flash attention 2, sequence packing, and parallelism via DeepSpeed Ulysses and LoRA.
*   **Scalability:** Scales up to 671B models and hundreds of GPUs with expert parallelism.
*   **Experiment Tracking:** Integrates with popular tracking tools like WandB, SwanLab, MLflow, and TensorBoard.

## What's New

*   **ReTool Recipe Open Sourced**: The [ReTool](https://arxiv.org/pdf/2504.11536) recipe is fully open-sourced. [Blog](https://www.notion.so/verl-reTool-recipe-Using-multi-round-conversations-and-code-sandboxing-to-improve-the-math-of-large-23a8b5b7feba80b386b2e5b5e3c1cde0)
*   **First verl Meetup at ICML**: verl keynote at [AWS AI Hours Singapore](https://pages.awscloud.com/aws-ai-hours-sg.html#agenda) on 7/8, verl & verl-agent project updates at [Agent for SWE meetup](https://lu.ma/e498qhsi) by LF AI & Data Singapore on 7/11.
*   **Megatron Backend Support for Large MoE Models**:  Enables large MoE models such as [DeepSeek-671b and Qwen3-236b](https://verl.readthedocs.io/en/latest/perf/dpsk.html).
*   **DAPO Open Sourced**: The open-sourced SOTA RL algorithm that achieves 50 points on AIME 2024 based on the Qwen2.5-32B pre-trained model, surpassing the previous SOTA achieved by DeepSeek's GRPO (DeepSeek-R1-Zero-Qwen-32B). DAPO's training is fully powered by verl and the reproduction code is available in `recipe/dapo` now.
*   **VAPO Published**: [VAPO](https://arxiv.org/pdf/2504.05118) (value-based augmented PPO) paper covers our latest RL method for reasoning models. Trained from Qwen-32B-base model, VAPO achieves 60.4 on AIME 2024, outperforming DAPO-32B.
*   **PF-PPO Support**:  [PF-PPO](https://arxiv.org/abs/2409.06957), accepted to ICML 2025, is now supported in verl!

  *For more updates and news, see the original [README](https://github.com/volcengine/verl) or the [verl blog](https://www.notion.so/verl-reTool-recipe-Using-multi-round-conversations-and-code-sandboxing-to-improve-the-math-of-large-23a8b5b7feba80b386b2e5b5e3c1cde0).*

## Getting Started

*   **[Documentation](https://verl.readthedocs.io/en/latest/index.html)**: Comprehensive documentation with installation guides, quickstarts, and advanced usage examples.
*   **Quickstart:** Follow the [Quickstart guide](https://verl.readthedocs.io/en/latest/start/quickstart.html) to get up and running quickly.
*   **Programming Guide:** Explore the [Programming Guide](https://verl.readthedocs.io/en/latest/hybrid_flow.html) for a deep dive into verl's architecture and functionality.
*   **Examples**: Run the [PPO in verl](https://verl.readthedocs.io/en/latest/algo/ppo.html) and [GRPO in verl](https://verl.readthedocs.io/en/latest/algo/grpo.html) examples.
*   **Code Architecture & Configuration:** Learn the [PPO Example Architecture](https://verl.readthedocs.io/en/latest/examples/ppo_code_architecture.html) and [Config Explanation](https://verl.readthedocs.io/en/latest/examples/config.html) for in-depth understanding.

## Community & Support

verl is an open-source project initiated by ByteDance Seed team and supported by a growing community.

*   [Join the verl Slack](https://join.slack.com/t/verlgroup/shared_invite/zt-2w5p9o4c3-yy0x2Q56s_VlGLsJ93A6vA)
*   Follow us on [Twitter](https://twitter.com/verl_project)

## Cite Us

If you use verl in your research, please cite:

```bibtex
@article{sheng2024hybridflow,
  title   = {HybridFlow: A Flexible and Efficient RLHF Framework},
  author  = {Guangming Sheng and Chi Zhang and Zilingfeng Ye and Xibin Wu and Wang Zhang and Ru Zhang and Yanghua Peng and Haibin Lin and Chuan Wu},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2409.19256}
}
```

## Contributions

We welcome contributions! See our [contribution guide](CONTRIBUTING.md) to get started.

---
**[Visit the original repository for more details](https://github.com/volcengine/verl).**