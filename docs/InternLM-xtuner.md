<div align="center">
  <img src="https://github.com/InternLM/lmdeploy/assets/36994684/0cf8d00f-e86b-40ba-9b54-dc8f1bc6c8d8" width="600"/>
  <br /><br />

[![GitHub Repo stars](https://img.shields.io/github/stars/InternLM/xtuner?style=social)](https://github.com/InternLM/xtuner/stargazers)
[![license](https://img.shields.io/github/license/InternLM/xtuner.svg)](https://github.com/InternLM/xtuner/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/xtuner)](https://pypi.org/project/xtuner/)
[![Downloads](https://static.pepy.tech/badge/xtuner)](https://pypi.org/project/xtuner/)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/InternLM/xtuner)](https://github.com/InternLM/xtuner/issues)
[![open issues](https://img.shields.io/github/issues-raw/InternLM/xtuner)](https://github.com/InternLM/xtuner/issues)

👋 Join the XTuner community: [![Static Badge](https://img.shields.io/badge/-grey?style=social&logo=wechat&label=WeChat)](https://cdn.vansin.top/internlm/xtuner.jpg)
[![Static Badge](https://img.shields.io/badge/-grey?style=social&logo=twitter&label=Twitter)](https://twitter.com/intern_lm)
[![Static Badge](https://img.shields.io/badge/-grey?style=social&logo=discord&label=Discord)](https://discord.gg/xa29JuW87d)

🔍 Explore our models on: [![Static Badge](https://img.shields.io/badge/-grey?style=social&label=🤗%20Huggingface)](https://huggingface.co/xtuner)
[![Static Badge](https://img.shields.io/badge/-grey?style=social&label=🤖%20ModelScope)](https://www.modelscope.cn/organization/xtuner)
[![Static Badge](https://img.shields.io/badge/-grey?style=social&label=🧰%20OpenXLab)](https://openxlab.org.cn/usercenter/xtuner)
[![Static Badge](https://img.shields.io/badge/-grey?style=social&label=🧠%20WiseModel)](https://www.wisemodel.cn/organization/xtuner)

English | [简体中文](README_zh-CN.md)
</div>

## XTuner: Unleash the Power of Efficient LLM Training

XTuner is a next-generation training engine designed for fine-tuning and training ultra-large language models (LLMs), especially Mixture of Experts (MoE) models, offering superior efficiency and scalability. [Learn more about XTuner on GitHub](https://github.com/InternLM/xtuner).

## Key Features

*   **Dropless Training:** Train massive MoE models (up to 600B parameters) with optimized parallelism, eliminating the need for complex expert parallelism.
    *   Scalable without complexity: Train 200B-scale MoE models without expert parallelism; 600B models require only intra-node expert parallelism
    *   Optimized parallelism strategy: Smaller expert parallelism dimension compared to traditional 3D approaches, enabling more efficient Dropless training
*   **Long Sequence Support:** Train 200B MoE models on 64k sequence lengths without sequence parallelism.
    *   Memory-efficient design: Train 200B MoE models on 64k sequence lengths without sequence parallelism through advanced memory optimization techniques
    *   Flexible scaling: Full support for DeepSpeed Ulysses sequence parallelism with linearly scalable maximum sequence length
    *   Robust performance: Maintains stability despite expert load imbalance during long sequence training
*   **Superior Efficiency:** Experience groundbreaking performance and hardware optimization.
    *   Supports MoE training up to 1T parameters
    *   Breakthrough performance: First to achieve FSDP training throughput that surpasses traditional 3D parallel schemes for MoE models above 200B scale
    *   Hardware optimization: Achieves training efficiency on Ascend A3 Supernode that exceeds NVIDIA H800

## Speed Benchmark

<div align=center>
  <img src="https://github.com/user-attachments/assets/fa42d587-068d-427b-b88c-25a164b3511c" style="width:80%">
</div>

## Roadmap & Future Development

XTuner V1 is committed to continuously improving training efficiency for pre-training, instruction fine-tuning, and reinforcement learning of ultra-large MoE models, with special focus on Ascend NPU optimization.

### 🚀 Training Engine

Our vision is to establish XTuner V1 as a versatile training backend that seamlessly integrates with the broader open-source ecosystem.

| Model         | GPU (FP8) | GPU (BF16) | NPU (BF16) |
| :------------ | :-------- | :--------- | :--------- |
| Intern S1     | ✅        | ✅         | ✅         |
| Intern VL     | ✅        | ✅         | ✅         |
| Qwen3 Dense   | ✅        | ✅         | ✅         |
| Qwen3 MoE     | ✅        | ✅         | ✅         |
| GPT OSS       | ✅        | ✅         | 🚧         |
| Deepseek V3   | ✅        | ✅         | 🚧         |
| KIMI K2       | ✅        | ✅         | 🚧         |

### 🧠 Algorithm

The algorithm component is actively evolving. We welcome community contributions - with XTuner V1, scale your algorithms to unprecedented sizes!

**Implemented**

*   ✅ **Multimodal Pre-training** - Full support for vision-language model training
*   ✅ **Multimodal Supervised Fine-tuning** - Optimized for instruction following
*   ✅ [GRPO](https://arxiv.org/pdf/2402.03300) - Group Relative Policy Optimization

**Coming Soon**

*   🔄 [MPO](https://arxiv.org/pdf/2411.10442) - Mixed Preference Optimization
*   🔄 [DAPO](https://arxiv.org/pdf/2503.14476) - Dynamic Sampling Policy Optimization
*   🔄 **Multi-turn Agentic RL** - Advanced agent training capabilities

### ⚡ Inference Engine Integration

Seamless deployment with leading inference frameworks:

*   [x] LMDeploy
*   [ ] vLLM
*   [ ] SGLang

### Data Preparation

*   You can use [GraphGen](https://github.com/open-sciencelab/GraphGen) to create synthetic data for fine-tuning.

## Contributing

We welcome contributions!  See our [CONTRIBUTING.md](.github/CONTRIBUTING.md) for guidelines.

## Acknowledgements

XTuner V1's development draws inspiration from the following open-source projects:

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

Released under the [Apache License 2.0](LICENSE). Please also adhere to the licenses of any models and datasets you use.