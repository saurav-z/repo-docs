<div align="center">
  <img src="https://github.com/InternLM/lmdeploy/assets/36994684/0cf8d00f-e86b-40ba-9b54-dc8f1bc6c8d8" width="600"/>
  <br /><br />
</div>

[![GitHub Repo stars](https://img.shields.io/github/stars/InternLM/xtuner?style=social)](https://github.com/InternLM/xtuner/stargazers)
[![license](https://img.shields.io/github/license/InternLM/xtuner.svg)](https://github.com/InternLM/xtuner/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/xtuner)](https://pypi.org/project/xtuner/)
[![Downloads](https://static.pepy.tech/badge/xtuner)](https://pypi.org/project/xtuner/)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/InternLM/xtuner)](https://github.com/InternLM/xtuner/issues)
[![open issues](https://img.shields.io/github/issues-raw/InternLM/xtuner)](https://github.com/InternLM/xtuner/issues)

👋 Join the XTuner community on: [![Static Badge](https://img.shields.io/badge/-grey?style=social&logo=wechat&label=WeChat)](https://cdn.vansin.top/internlm/xtuner.jpg)
[![Static Badge](https://img.shields.io/badge/-grey?style=social&logo=twitter&label=Twitter)](https://twitter.com/intern_lm)
[![Static Badge](https://img.shields.io/badge/-grey?style=social&logo=discord&label=Discord)](https://discord.gg/xa29JuW87d)

🚀 **XTuner is a cutting-edge LLM training engine designed for efficient and scalable fine-tuning of large language models, especially MoE models.**

[View the XTuner Repository on GitHub](https://github.com/InternLM/xtuner)

## Key Features

*   **Dropless Training:** Train massive models without expert parallelism, simplifying the process and improving efficiency.
    *   Scalable training of ultra-large MoE models (up to 200B) without expert parallelism.
    *   Optimized parallelism strategy for efficient Dropless training.
*   **Long Sequence Support:** Train models with extended sequence lengths without performance compromises.
    *   Memory-efficient design enables training 200B MoE models on 64k sequence lengths.
    *   Full support for DeepSpeed Ulysses sequence parallelism for flexible scaling.
    *   Robust performance with consistent stability during long sequence training.
*   **Superior Efficiency:** Experience unparalleled training speed and scale.
    *   Supports MoE training up to 1T parameters.
    *   Achieves FSDP training throughput surpassing traditional 3D parallel schemes for MoE models above 200B scale.
    *   Hardware optimization enables top performance on Ascend A3 Supernode.

## Speed Benchmark

<div align=center>
  <img src="https://github.com/user-attachments/assets/fa42d587-068d-427b-b88c-25a164b3511c" style="width:80%">
</div>

## What's New

-   **\[2025/09]** XTuner V1 Released! A Next-Generation Training Engine Built for Ultra-Large MoE Models

## XTuner V1: The Future of LLM Training

XTuner V1 is a next-generation LLM training engine specifically designed for ultra-large-scale MoE models. Built for modern research, XTuner V1 is optimized to handle the complexity of today's MoE training scenarios.

## Roadmap

XTuner V1 is committed to continuously improving training efficiency for pre-training, instruction fine-tuning, and reinforcement learning of ultra-large MoE models, with special focus on Ascend NPU optimization.

### Training Engine

Our vision is to establish XTuner V1 as a versatile training backend that seamlessly integrates with the broader open-source ecosystem.

| Model         | GPU (FP8) | GPU (BF16) | NPU (BF16) |
| ------------- | --------- | ---------- | ---------- |
| Intern S1     | ✅        | ✅         | ✅         |
| Intern VL     | ✅        | ✅         | ✅         |
| Qwen3 Dense   | ✅        | ✅         | ✅         |
| Qwen3 MoE     | ✅        | ✅         | ✅         |
| GPT OSS       | ✅        | ✅         | 🚧        |
| Deepseek V3   | ✅        | ✅         | 🚧        |
| KIMI K2       | ✅        | ✅         | 🚧        |

### Algorithm

The algorithm component is actively evolving. We welcome community contributions - with XTuner V1, scale your algorithms to unprecedented sizes!

**Implemented**

*   ✅ **Multimodal Pre-training** - Full support for vision-language model training
*   ✅ **Multimodal Supervised Fine-tuning** - Optimized for instruction following
*   ✅ [GRPO](https://arxiv.org/pdf/2402.03300) - Group Relative Policy Optimization

**Coming Soon**

*   🔄 [MPO](https://arxiv.org/pdf/2411.10442) - Mixed Preference Optimization
*   🔄 [DAPO](https://arxiv.org/pdf/2503.14476) - Dynamic Sampling Policy Optimization
*   🔄 **Multi-turn Agentic RL** - Advanced agent training capabilities

### Inference Engine Integration

Seamless deployment with leading inference frameworks:

-   [x] LMDeploy
-   [ ] vLLM
-   [ ] SGLang

### Data Preparation

*   You can use [GraphGen](https://github.com/open-sciencelab/GraphGen) to create synthetic data for fine-tuning.

## Contributing

We appreciate all contributions to XTuner. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgements

The development of XTuner V1's training engine has been greatly inspired by and built upon the excellent work of the open-source community. We extend our sincere gratitude to the following pioneering projects:

**Training Engine:**

-   [Torchtitan](https://github.com/pytorch/torchtitan) - A PyTorch native platform for training generative AI models
-   [Deepspeed](https://github.com/deepspeedai/DeepSpeed) - Microsoft's deep learning optimization library
-   [MindSpeed](https://gitee.com/ascend/MindSpeed) - Ascend's high-performance training acceleration library
-   [Megatron](https://github.com/NVIDIA/Megatron-LM) - NVIDIA's large-scale transformer training framework

**Reinforcement Learning:**

XTuner V1's reinforcement learning capabilities have been enhanced through insights and best practices from:

-   [veRL](https://github.com/volcengine/verl) - Volcano Engine Reinforcement Learning for LLMs
-   [SLIME](https://github.com/THUDM/slime) - THU's scalable RLHF implementation
-   [AReal](https://github.com/inclusionAI/AReaL) - Ant Reasoning Reinforcement Learning for LLMs
-   [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) - An Easy-to-use, Scalable and High-performance RLHF Framework based on Ray

We are deeply grateful to all contributors and maintainers of these projects for advancing the field of large-scale model training.

## Citation

```bibtex
@misc{2023xtuner,
    title={XTuner: A Toolkit for Efficiently Fine-tuning LLM},
    author={XTuner Contributors},
    howpublished = {\url{https://github.com/InternLM/xtuner}},
    year={2023}
}
```

## License

This project is released under the [Apache License 2.0](LICENSE). Please also adhere to the Licenses of models and datasets being used.