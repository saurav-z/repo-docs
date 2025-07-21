# verl: Unleash the Power of Reinforcement Learning for LLMs

verl is an open-source RL training library, initiated by ByteDance Seed and the verl community, designed for efficient and flexible Reinforcement Learning for Large Language Models (LLMs). Learn more on the [original repo](https://github.com/volcengine/verl).

**Key Features:**

*   **Flexible RL Algorithms:** Easily extend and customize RL algorithms like PPO and GRPO with the hybrid-controller programming model.
*   **Seamless LLM Integration:** Integrate with existing LLM frameworks (FSDP, Megatron-LM, vLLM, SGLang, HuggingFace) using modular APIs.
*   **Efficient Resource Utilization:** Supports flexible device mapping for optimal GPU resource utilization and scalability.
*   **State-of-the-Art Performance:** Achieves SOTA LLM training and inference throughput.
*   **Extensive Model Support:** Ready integration with Hugging Face models and compatible with Qwen-3, Qwen-2.5, Llama3.1, Gemma2, and more.
*   **Comprehensive RL Support:** Includes support for various RL algorithms like PPO, GRPO, ReMax, and others, including model-based and function-based rewards.
*   **Multi-Modal and Multi-Turn Capabilities:** Supports vision-language models and multi-turn interactions with tools using SGLang.

## Key Highlights

*   **HybridFlow Paper:** verl is the open-source version of [HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256v2)
*   **Latest News & Updates:** Stay informed with recent announcements, including conference presentations, new feature releases, and community contributions. (See "News" section in original README)
*   **Thriving Community:** Join the growing community of researchers and developers using verl for cutting-edge LLM research and applications.
*   **Performance Tuning Guide:** Offers a detailed guide for optimizing the performance of your RL models.
*   **Extensive Tooling:** Supports expert parallelism, multi-GPU LoRA RL, and experiment tracking with various tools.

## Getting Started

*   **Documentation:** Access comprehensive documentation to guide you through installation, quickstarts, and advanced usage.
*   **Examples:** Explore practical examples, including how to run PPO step-by-step, and learn from reproducible algorithm baselines.
*   **Community Contributions:** Discover the work of many awesome projects built on verl including SkyThought, TinyZero, MemAgent and many more (See "Awesome work using verl" section in original README).

## Contributing

Contribute to verl and help improve the library by following the [contribution guide](CONTRIBUTING.md).

## Support

For support or to learn more about the project, visit the following resources:

*   Documentation:  [https://verl.readthedocs.io/en/latest/index.html](https://verl.readthedocs.io/en/latest/index.html)
*   Twitter:  [@verl\_project](https://twitter.com/verl_project)
*   Slack:  [verlgroup](https://join.slack.com/t/verlgroup/shared_invite/zt-2w5p9o4c3-yy0x2Q56s_VlGLsJ93A6vA)

## Citation

If you use verl in your research, please cite the following:

*   [HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256v2)

```bibtex
@article{sheng2024hybridflow,
  title   = {HybridFlow: A Flexible and Efficient RLHF Framework},
  author  = {Guangming Sheng and Chi Zhang and Zilingfeng Ye and Xibin Wu and Wang Zhang and Ru Zhang and Yanghua Peng and Haibin Lin and Chuan Wu},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2409.19256}
}