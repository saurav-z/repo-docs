<div align="center">
  <img src="https://github.com/InternLM/lmdeploy/assets/36994684/0cf8d00f-e86b-40ba-9b54-dc8f1bc6c8d8" width="600"/>
  <br /><br />

[![GitHub Repo stars](https://img.shields.io/github/stars/InternLM/xtuner?style=social)](https://github.com/InternLM/xtuner/stargazers)
[![license](https://img.shields.io/github/license/InternLM/xtuner.svg)](https://github.com/InternLM/xtuner/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/xtuner)](https://pypi.org/project/xtuner/)
[![Downloads](https://static.pepy.tech/badge/xtuner)](https://pypi.org/project/xtuner/)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/InternLM/xtuner)](https://github.com/InternLM/xtuner/issues)
[![open issues](https://img.shields.io/github/issues-raw/InternLM/xtuner)](https://github.com/InternLM/xtuner/issues)

👋 Join us on
[![Static Badge](https://img.shields.io/badge/-grey?style=social&logo=wechat&label=WeChat)](https://cdn.vansin.top/internlm/xtuner.jpg)
[![Static Badge](https://img.shields.io/badge/-grey?style=social&logo=twitter&label=Twitter)](https://twitter.com/intern_lm)
[![Static Badge](https://img.shields.io/badge/-grey?style=social&logo=discord&label=Discord)](https://discord.gg/xa29JuW87d)

🔍 Explore our models on
[![Static Badge](https://img.shields.io/badge/-gery?style=social&label=🤗%20Huggingface)](https://huggingface.co/xtuner)
[![Static Badge](https://img.shields.io/badge/-gery?style=social&label=🤖%20ModelScope)](https://www.modelscope.cn/organization/xtuner)
[![Static Badge](https://img.shields.io/badge/-gery?style=social&label=🧰%20OpenXLab)](https://openxlab.org.cn/usercenter/xtuner)
[![Static Badge](https://img.shields.io/badge/-gery?style=social&label=🧠%20WiseModel)](https://www.wisemodel.cn/organization/xtuner)

English | [简体中文](README_zh-CN.md)
</div>

# XTuner: The Next-Generation LLM Training Engine for Ultra-Large MoE Models

XTuner is a powerful toolkit designed for efficient fine-tuning of Large Language Models (LLMs), enabling cutting-edge research and development in the field.  [Explore the XTuner repository on GitHub](https://github.com/InternLM/xtuner).

## Key Features

*   **Dropless Training:**

    *   Train massive MoE models (200B+ parameters) without the complexity of expert parallelism; 600B models require only intra-node expert parallelism.
    *   Optimized parallelism strategy for efficient Dropless training with smaller expert parallelism dimensions.

*   **Long Sequence Support:**

    *   Train 200B MoE models on 64k sequence lengths without sequence parallelism using advanced memory optimization.
    *   Full support for DeepSpeed Ulysses sequence parallelism for linearly scalable maximum sequence length.
    *   Maintains stability even with expert load imbalance during long sequence training.

*   **Superior Efficiency:**

    *   Supports MoE training up to 1T parameters.
    *   Achieves industry-leading FSDP training throughput, surpassing traditional 3D parallel schemes for MoE models above 200B scale.
    *   Optimized for Ascend A3 Supernode hardware, delivering exceptional training efficiency.

## Roadmap

XTuner V1 is continuously evolving to improve training efficiency for pre-training, instruction fine-tuning, and reinforcement learning of ultra-large MoE models, with a special focus on Ascend NPU optimization.

### Training Engine

XTuner V1 aims to become a versatile training backend, seamlessly integrating with the open-source ecosystem.

|   Model    |  GPU(FP8) | GPU(BF16)| NPU(BF16) |
|------------|-----------|----------|-----------|
| Intern S1  |    ✅     |    ✅    |    ✅     |
| Intern VL  |    ✅     |    ✅    |    ✅     |
| Qwen3 Dense|    ✅     |    ✅    |    ✅     |
| Qwen3 MoE  |    ✅     |    ✅    |    ✅     |
| GPT OSS    |    ✅     |    ✅    |    🚧     |
| Deepseek V3|    ✅     |    ✅    |    🚧     |
| KIMI K2    |    ✅     |    ✅    |    🚧     |

### Algorithm

The algorithm component is actively evolving, with community contributions welcome.

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

*   Use [GraphGen](https://github.com/open-sciencelab/GraphGen) to generate synthetic data for fine-tuning.

## Contributing

We welcome contributions!  See [CONTRIBUTING.md](.github/CONTRIBUTING.md) for guidelines.

## Acknowledgements

XTuner V1's development draws inspiration from the open-source community.  Special thanks to:

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

This project is licensed under the [Apache License 2.0](LICENSE).  Please adhere to the licenses of used models and datasets.