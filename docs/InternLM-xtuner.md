<div align="center">
  <img src="https://github.com/InternLM/lmdeploy/assets/36994684/0cf8d00f-e86b-40ba-9b54-dc8f1bc6c8d8" width="600" alt="XTuner Logo">
  <br /><br />

[![GitHub Repo stars](https://img.shields.io/github/stars/InternLM/xtuner?style=social)](https://github.com/InternLM/xtuner/stargazers)
[![License](https://img.shields.io/github/license/InternLM/xtuner.svg)](https://github.com/InternLM/xtuner/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/xtuner)](https://pypi.org/project/xtuner/)
[![Downloads](https://static.pepy.tech/badge/xtuner)](https://pypi.org/project/xtuner/)
[![Issue Resolution](https://img.shields.io/github/issues-closed-raw/InternLM/xtuner)](https://github.com/InternLM/xtuner/issues)
[![Open Issues](https://img.shields.io/github/issues-raw/InternLM/xtuner)](https://github.com/InternLM/xtuner/issues)

👋 Join the XTuner community on: [![WeChat](https://img.shields.io/badge/-grey?style=social&logo=wechat&label=WeChat)](https://cdn.vansin.top/internlm/xtuner.jpg)
[![Twitter](https://img.shields.io/badge/-grey?style=social&logo=twitter&label=Twitter)](https://twitter.com/intern_lm)
[![Discord](https://img.shields.io/badge/-grey?style=social&logo=discord&label=Discord)](https://discord.gg/xa29JuW87d)

🔍 Explore our models on:
[![Hugging Face](https://img.shields.io/badge/-gery?style=social&label=%F0%9F%A4%97%20Huggingface)](https://huggingface.co/xtuner)
[![ModelScope](https://img.shields.io/badge/-gery?style=social&label=%F0%9F%A4%96%20ModelScope)](https://www.modelscope.cn/organization/xtuner)
[![OpenXLab](https://img.shields.io/badge/-gery?style=social&label=%F0%9F%A7%B0%20OpenXLab)](https://openxlab.org.cn/usercenter/xtuner)
[![WiseModel](https://img.shields.io/badge/-gery?style=social&label=%F0%9F%A7%A0%20WiseModel)](https://www.wisemodel.cn/organization/xtuner)

[English](README.md) | [简体中文](README_zh-CN.md)

</div>

## XTuner: The Cutting-Edge Toolkit for LLM Training and Fine-tuning

XTuner is a next-generation LLM training engine, empowering researchers and developers to efficiently train and fine-tune large language models, particularly ultra-large MoE models.  Learn more on the [XTuner GitHub Repository](https://github.com/InternLM/xtuner).

## Key Features of XTuner V1

*   **Dropless Training for Enhanced Scalability:**
    *   Train massive 200B+ parameter MoE models without expert parallelism and 600B models with only intra-node expert parallelism.
    *   Optimized parallelism strategies for more efficient Dropless training.

*   **Long Sequence Support:**
    *   Train 200B MoE models with 64k sequence lengths using memory optimization.
    *   Full support for DeepSpeed Ulysses sequence parallelism, linearly scaling maximum sequence length.
    *   Maintains stability even with expert load imbalances during long sequence training.

*   **Superior Efficiency:**
    *   Supports MoE training up to 1T parameters.
    *   Achieves FSDP training throughput that surpasses traditional 3D parallel schemes for MoE models above 200B scale.
    *   Optimized for hardware, achieving leading training efficiency on Ascend A3 Supernode.

## 🚀 Speed Benchmark

<div align=center>
  <img src="https://github.com/user-attachments/assets/fa42d587-068d-427b-b88c-25a164b3511c" style="width:80%" alt="Speed Benchmark">
</div>

## 🎉 News

*   **\[2025/09\]** XTuner V1 Released!

## Roadmap

XTuner V1 is continuously improving to deliver improved training efficiency for pre-training, instruction fine-tuning, and reinforcement learning of ultra-large MoE models, with an emphasis on Ascend NPU optimization.

### 🚀 Training Engine

XTuner V1 aims to be a versatile training backend that integrates into the open-source ecosystem.

|   Model    |  GPU(FP8) | GPU(BF16)| NPU(BF16) |
|------------|-----------|----------|-----------|
| Intern S1  |    ✅     |    ✅    |    ✅     |
| Intern VL  |    ✅     |    ✅    |    ✅     |
| Qwen3 Dense|    ✅     |    ✅    |    ✅     |
| Qwen3 MoE  |    ✅     |    ✅    |    ✅     |
| GPT OSS    |    ✅     |    ✅    |    🚧     |
| Deepseek V3|    ✅     |    ✅    |    🚧     |
| KIMI K2    |    ✅     |    ✅    |    🚧     |

### 🧠 Algorithm

The algorithm component is under active development, and we encourage community contributions.

**Implemented**

*   ✅ **Multimodal Pre-training:** Comprehensive support for vision-language model training.
*   ✅ **Multimodal Supervised Fine-tuning:** Optimized for instruction following.
*   ✅ [GRPO](https://arxiv.org/pdf/2402.03300) - Group Relative Policy Optimization

**Coming Soon**

*   🔄 [MPO](https://arxiv.org/pdf/2411.10442) - Mixed Preference Optimization
*   🔄 [DAPO](https://arxiv.org/pdf/2503.14476) - Dynamic Sampling Policy Optimization
*   🔄 **Multi-turn Agentic RL** - Advanced agent training capabilities

### ⚡ Inference Engine Integration

Seamless deployment with leading inference frameworks:

-   [x] LMDeploy
-   [ ] vLLM
-   [ ] SGLang

### Data Preparation

*   You can use [GraphGen](https://github.com/open-sciencelab/GraphGen) to generate synthetic data for fine-tuning.

## 🤝 Contributing

We welcome all contributions to XTuner.  Please review the [CONTRIBUTING.md](.github/CONTRIBUTING.md) file for guidelines.

## 🙏 Acknowledgements

XTuner V1's development draws inspiration and builds upon the work of the open-source community.  We are grateful to the following projects:

**Training Engine:**

*   [Torchtitan](https://github.com/pytorch/torchtitan)
*   [Deepspeed](https://github.com/deepspeedai/DeepSpeed)
*   [MindSpeed](https://gitee.com/ascend/MindSpeed)
*   [Megatron](https://github.com/NVIDIA/Megatron-LM)

**Reinforcement Learning:**

XTuner V1's reinforcement learning capabilities benefit from insights and best practices from:

*   [veRL](https://github.com/volcengine/verl)
*   [SLIME](https://github.com/THUDM/slime)
*   [AReal](https://github.com/inclusionAI/AReaL)
*   [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)

We thank all contributors to these projects for their contributions to the field of large-scale model training.

## 🖊️ Citation

```bibtex
@misc{2023xtuner,
    title={XTuner: A Toolkit for Efficiently Fine-tuning LLM},
    author={XTuner Contributors},
    howpublished = {\url{https://github.com/InternLM/xtuner}},
    year={2023}
}
```

## License

XTuner is released under the [Apache License 2.0](LICENSE).  Please adhere to the licenses of any models and datasets used.