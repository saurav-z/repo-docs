# verl: Revolutionizing Large Language Model (LLM) Training with Reinforcement Learning (RL)

verl is a powerful open-source RL training library designed for LLMs, offering flexibility, efficiency, and production-readiness. Dive into the world of advanced LLM training with [verl's comprehensive capabilities](https://github.com/volcengine/verl)!

## Key Features

*   **Flexible RL Algorithms:** Easily extend and implement diverse RL algorithms such as PPO, GRPO, and more. Build RL dataflows with ease using our hybrid-controller programming model.
*   **Seamless Integration:** Integrate with existing LLM infrastructures such as FSDP, Megatron-LM, vLLM, and SGLang, thanks to modular APIs that decouple computation and data dependencies.
*   **Efficient Resource Utilization:** Leverage flexible device mapping to optimize GPU usage across various cluster sizes.
*   **Hugging Face Compatibility:** Ready integration with popular HuggingFace models.
*   **State-of-the-Art Performance:** Experience high LLM training throughput with SOTA LLM training and inference engine integrations, and SOTA RL throughput.
*   **3D-HybridEngine for Efficiency:** Benefit from efficient actor model resharding, reducing memory redundancy and communication overhead.

### Highlighted Features:
*   **FSDP, FSDP2 and Megatron-LM** for training.
*   **vLLM, SGLang and HF Transformers** for rollout generation.
*   Compatible with popular models like **Qwen-3, Qwen-2.5, Llama3.1, Gemma2, DeepSeek-LLM**, and more.
*   Support for SFT, PPO, GRPO, ReMax, RLOO and other cutting-edge algorithms.
*   Integration of model-based and function-based rewards for math and coding tasks.
*   Multi-turn LLM with tool calling support, VLM models support, and LLM alignment recipes.
*   Flash attention 2, sequence packing, sequence parallelism, LoRA, and Liger-kernel support.
*   Experiment tracking with wandb, swanlab, mlflow and tensorboard.

## News and Updates

Stay informed on the latest verl developments:

*   [Recent News and Events, Tutorials and Technical Reports](https://github.com/volcengine/verl#news)
*   [Project roadmap](https://github.com/volcengine/verl/issues/710)
*   [verl-agent](https://github.com/langfengQ/verl-agent): A scalable training framework for **long-horizon LLM/VLM agents**, along with a new algorithm **GiGPO**
*   And more!

## Getting Started

Explore the possibilities with verl:

*   [Documentation](https://verl.readthedocs.io/en/latest/index.html)
*   [Installation](https://verl.readthedocs.io/en/latest/start/install.html)
*   [Quickstart](https://verl.readthedocs.io/en/latest/start/quickstart.html)
*   [Programming Guide](https://verl.readthedocs.io/en/latest/hybrid_flow.html)
*   [PPO in verl](https://verl.readthedocs.io/en/latest/algo/ppo.html)
*   [GRPO in verl](https://verl.readthedocs.io/en/latest/algo/grpo.html)
*   [Reproducible algorithm baselines](https://verl.readthedocs.io/en/latest/algo/baseline.html)

## Performance Tuning

Maximize your performance with our detailed [performance tuning guide](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html).

## Community Contributions

verl is a community-driven project, inspired by the design of Nemo-Aligner, Deepspeed-chat and OpenRLHF, adopted and contributed by Bytedance, Anyscale, LMSys.org, and more!

## Awesome Work Using verl
*   [A list of cutting-edge research and implementations using verl](https://github.com/volcengine/verl#awesome-work-using-verl)

## Contribution Guide

Contribute to verl's development: [contributions guide](CONTRIBUTING.md)

## About

verl is initiated by ByteDance Seed team.
[ByteDance Seed Team](https://team.doubao.com/) is dedicated to crafting the industry's most advanced AI foundation models.