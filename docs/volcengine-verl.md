# verl: Train LLMs Efficiently with Reinforcement Learning

verl is an open-source, production-ready reinforcement learning (RL) library for training large language models (LLMs), offering flexibility, efficiency, and state-of-the-art performance.  Explore the original repository on GitHub: [volcengine/verl](https://github.com/volcengine/verl).

**Key Features:**

*   **Flexible RL Algorithms:** Easily implement and extend diverse RL algorithms like PPO, GRPO, and DAPO with the hybrid-controller programming model.
*   **Seamless Integration:** Integrate with existing LLM frameworks (FSDP, vLLM, SGLang, etc.) through modular APIs.
*   **Efficient Device Mapping:**  Supports flexible placement of models on different GPUs for optimized resource utilization and scalability.
*   **Hugging Face Compatibility:** Ready integration with popular Hugging Face models, including Qwen, Llama, and DeepSeek-LLM.
*   **State-of-the-Art Performance:** Achieve SOTA LLM training and inference throughput with efficient actor model resharding.

**Highlighted Features:**

*   **SOTA Integration:** Integrates with FSDP, FSDP2, and Megatron-LM for training, and vLLM, SGLang, and HF Transformers for rollout generation.
*   **Model Support:** Compatible with a wide range of models: Qwen-3, Qwen-2.5, Llama3.1, Gemma2, DeepSeek-LLM, and more.
*   **Training Capabilities:** Supports supervised fine-tuning, and various reinforcement learning algorithms, including PPO, GRPO, ReMax, REINFORCE++, RLOO, PRIME, DAPO, DrGRPO, KL_Cov & Clip_Cov, and Self-play preference optimization (SPPO).
*   **Multi-Modal RL:** Support vision-language models (VLMs) and multi-modal RL with Qwen2.5-vl, Kimi-VL.
*   **Advanced Optimizations:** Supports Flash attention 2, sequence packing, sequence parallelism (DeepSpeed Ulysses), LoRA, Liger-kernel, and expert parallelism.
*   **Memory Optimization:** Multi-gpu LoRA RL support for efficient memory usage.
*   **Experiment Tracking:** Integrated experiment tracking with wandb, swanlab, mlflow, and tensorboard.
*   **Community-Driven:** Features a comprehensive [Awesome work using verl](https://github.com/volcengine/verl#awesome-work-using-verl) showcase.

**Getting Started:**

*   [Documentation](https://verl.readthedocs.io/en/latest/index.html)
*   [Installation](https://verl.readthedocs.io/en/latest/start/install.html)
*   [Quickstart](https://verl.readthedocs.io/en/latest/start/quickstart.html)
*   [PPO in verl](https://verl.readthedocs.io/en/latest/algo/ppo.html)
*   [GRPO in verl](https://verl.readthedocs.io/en/latest/algo/grpo.html)

**Performance Tuning Guide:**

*   Optimize your RL training with our detailed [performance tuning guide](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html).

**Contribution:**

*   See our [contributions guide](CONTRIBUTING.md) to get involved.

---

**Note:** Please refer to the documentation and the repository for the most up-to-date information on features, installation, and usage.