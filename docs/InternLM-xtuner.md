<div align="center">
  <img src="https://github.com/InternLM/lmdeploy/assets/36994684/0cf8d00f-e86b-40ba-9b54-dc8f1bc6c8d8" width="600"/>
  <br /><br />

[![GitHub Repo stars](https://img.shields.io/github/stars/InternLM/xtuner?style=social)](https://github.com/InternLM/xtuner/stargazers)
[![license](https://img.shields.io/github/license/InternLM/xtuner.svg)](https://github.com/InternLM/xtuner/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/xtuner)](https://pypi.org/project/xtuner/)
[![Downloads](https://static.pepy.tech/badge/xtuner)](https://pypi.org/project/xtuner/)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/InternLM/xtuner)](https://github.com/InternLM/xtuner/issues)
[![open issues](https://img.shields.io/github/issues-raw/InternLM/xtuner)](https://github.com/InternLM/xtuner/issues)

👋 Join the XTuner community on:
[![Static Badge](https://img.shields.io/badge/-grey?style=social&logo=wechat&label=WeChat)](https://cdn.vansin.top/internlm/xtuner.jpg)
[![Static Badge](https://img.shields.io/badge/-grey?style=social&logo=twitter&label=Twitter)](https://twitter.com/intern_lm)
[![Static Badge](https://img.shields.io/badge/-grey?style=social&logo=discord&label=Discord)](https://discord.gg/xa29JuW87d)

🔍 Explore our models on:
[![Static Badge](https://img.shields.io/badge/-gery?style=social&label=🤗%20Huggingface)](https://huggingface.co/xtuner)
[![Static Badge](https://img.shields.io/badge/-gery?style=social&label=🤖%20ModelScope)](https://www.modelscope.cn/organization/xtuner)
[![Static Badge](https://img.shields.io/badge/-gery?style=social&label=🧰%20OpenXLab)](https://openxlab.org.cn/usercenter/xtuner)
[![Static Badge](https://img.shields.io/badge/-gery?style=social&label=🧠%20WiseModel)](https://www.wisemodel.cn/organization/xtuner)

English | [简体中文](README_zh-CN.md)

</div>

## XTuner: Your Toolkit for Efficient LLM Fine-tuning

XTuner is a cutting-edge training engine designed for efficient fine-tuning of large language models (LLMs), particularly for ultra-large-scale Mixture-of-Experts (MoE) models. [Explore the XTuner repository](https://github.com/InternLM/xtuner).

### Key Features

*   **Dropless Training:** Train massive MoE models (200B+ parameters) with remarkable efficiency, eliminating the need for expert parallelism in many cases.
    *   Scalable training without added complexity.
    *   Optimized parallelism strategy for efficient Dropless training.
*   **Long Sequence Support:** Train on exceptionally long sequences (up to 64k tokens for 200B MoE models) without sequence parallelism.
    *   Memory-efficient design for handling long sequences.
    *   Supports DeepSpeed Ulysses sequence parallelism for scalable lengths.
    *   Maintains stability even with expert load imbalances during training.
*   **Superior Efficiency:** Achieve unparalleled training performance.
    *   Supports MoE training up to 1 Trillion parameters.
    *   Achieves state-of-the-art training throughput for MoE models over 200B.
    *   Optimized for Ascend NPU hardware, surpassing traditional 3D parallel schemes.

### Speed Benchmark

<div align=center>
  <img src="https://github.com/user-attachments/assets/fa42d587-068d-427b-b88c-25a164b3511c" style="width:80%">
</div>

### News
-   **\[2025/09]** XTuner V1 Released!

### XTuner V1: Next-Generation LLM Training Engine

XTuner V1 represents a significant advancement in LLM training technology, focusing on the efficient training of ultra-large MoE models. It moves beyond conventional 3D parallel architectures, optimizing for the demands of modern research.

### Roadmap

XTuner V1 continuously improves training efficiency for pre-training, instruction fine-tuning, and reinforcement learning of ultra-large MoE models, with special focus on Ascend NPU optimization.

### Training Engine
Our vision is to establish XTuner V1 as a versatile training backend that seamlessly integrates with the broader open-source ecosystem.

| Model        | GPU(FP8) | GPU(BF16) | NPU(BF16) |
|--------------|----------|-----------|-----------|
| Intern S1    | ✅        | ✅         | ✅         |
| Intern VL    | ✅        | ✅         | ✅         |
| Qwen3 Dense  | ✅        | ✅         | ✅         |
| Qwen3 MoE    | ✅        | ✅         | ✅         |
| GPT OSS      | ✅        | ✅         | 🚧        |
| Deepseek V3  | ✅        | ✅         | 🚧        |
| KIMI K2      | ✅        | ✅         | 🚧        |

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

*   [x] LMDeploy
*   [ ] vLLM
*   [ ] SGLang

### Data Preparation

*   You can use [GraphGen](https://github.com/open-sciencelab/GraphGen) to create synthetic data for fine-tuning.

### Contributing

We encourage contributions to XTuner! Review the [CONTRIBUTING.md](.github/CONTRIBUTING.md) for guidelines.

### Acknowledgements

XTuner V1's development is inspired by and built upon the following open-source projects:

**Training Engine:**

*   [Torchtitan](https://github.com/pytorch/torchtitan)
*   [Deepspeed](https://github.com/deepspeedai/DeepSpeed)
*   [MindSpeed](https://gitee.com/ascend/MindSpeed)
*   [Megatron](https://github.com/NVIDIA/Megatron-LM)

**Reinforcement Learning:**

*   [veRL](https://github.com/volcengine/verl)
*   [SLIME](https://github.com/THUDM/slime)
*   [AReal](https://github.com/inclusionAI/AReaL)
*   [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)

### Citation

```bibtex
@misc{2023xtuner,
    title={XTuner: A Toolkit for Efficiently Fine-tuning LLM},
    author={XTuner Contributors},
    howpublished = {\url{https://github.com/InternLM/xtuner}},
    year={2023}
}
```

### License

This project is licensed under the [Apache License 2.0](LICENSE). Also adhere to the Licenses of the models and datasets being used.