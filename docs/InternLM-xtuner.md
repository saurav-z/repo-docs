<div align="center">
  <img src="https://github.com/InternLM/lmdeploy/assets/36994684/0cf8d00f-e86b-40ba-9b54-dc8f1bc6c8d8" width="600"/>
  <br /><br />

[![GitHub Repo stars](https://img.shields.io/github/stars/InternLM/xtuner?style=social)](https://github.com/InternLM/xtuner/stargazers)
[![license](https://img.shields.io/github/license/InternLM/xtuner.svg)](https://github.com/InternLM/xtuner/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/xtuner)](https://pypi.org/project/xtuner/)
[![Downloads](https://static.pepy.tech/badge/xtuner)](https://pypi.org/project/xtuner/)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/InternLM/xtuner)](https://github.com/InternLM/xtuner/issues)
[![open issues](https://img.shields.io/github/issues-raw/InternLM/xtuner)](https://github.com/InternLM/xtuner/issues)

👋 Join the XTuner community!
[![Static Badge](https://img.shields.io/badge/-grey?style=social&logo=wechat&label=WeChat)](https://cdn.vansin.top/internlm/xtuner.jpg)
[![Static Badge](https://img.shields.io/badge/-grey?style=social&logo=twitter&label=Twitter)](https://twitter.com/intern_lm)
[![Static Badge](https://img.shields.io/badge/-grey?style=social&logo=discord&label=Discord)](https://discord.gg/xa29JuW87d)

🔍 Explore XTuner models on:
[![Static Badge](https://img.shields.io/badge/-gery?style=social&label=🤗%20Huggingface)](https://huggingface.co/xtuner)
[![Static Badge](https://img.shields.io/badge/-gery?style=social&label=🤖%20ModelScope)](https://www.modelscope.cn/organization/xtuner)
[![Static Badge](https://img.shields.io/badge/-gery?style=social&label=🧰%20OpenXLab)](https://openxlab.org.cn/usercenter/xtuner)
[![Static Badge](https://img.shields.io/badge/-gery?style=social&label=🧠%20WiseModel)](https://www.wisemodel.cn/organization/xtuner)

English | [简体中文](README_zh-CN.md)

</div>

## XTuner: Unleash the Power of Ultra-Large MoE Models with Unrivaled Training Efficiency

XTuner is a next-generation LLM training engine designed for efficient training of ultra-large Mixture of Experts (MoE) models. Find the original repository [here](https://github.com/InternLM/xtuner).

## Key Features

*   **Dropless Training:**
    *   Train massive MoE models (up to 200B parameters) without expert parallelism, and 600B models with only intra-node expert parallelism.
    *   Optimized parallelism strategy for efficient "Dropless" training.
*   **Long Sequence Support:**
    *   Train 200B MoE models on sequence lengths up to 64k without sequence parallelism using advanced memory optimization.
    *   Full support for DeepSpeed Ulysses sequence parallelism for linearly scalable sequence lengths.
    *   Maintains stability even with expert load imbalance during long sequence training.
*   **Superior Efficiency:**
    *   Supports MoE training up to 1 Trillion parameters.
    *   Achieves FSDP training throughput surpassing traditional 3D parallel schemes for MoE models above 200B scale.
    *   Optimized for Ascend A3 Supernode hardware, achieving exceptional training efficiency.

## Speed Benchmark

<div align=center>
  <img src="https://github.com/user-attachments/assets/fa42d587-068d-427b-b88c-25a164b3511c" style="width:80%">
</div>

## News

-   **\[2025/09]** XTuner V1 Released!

## XTuner V1

XTuner V1 is built specifically for ultra-large-scale MoE models, offering innovative features and optimized performance.

## Roadmap

XTuner V1 is committed to continuous improvement in training efficiency for pre-training, instruction fine-tuning, and reinforcement learning of ultra-large MoE models, particularly with a focus on Ascend NPU optimization.

### Training Engine

XTuner V1 aims to be a versatile training backend, seamlessly integrating with the open-source ecosystem.

| Model          | GPU(FP8) | GPU(BF16) | NPU(BF16) |
| -------------- | -------- | --------- | --------- |
| Intern S1      | ✅       | ✅        | ✅        |
| Intern VL      | ✅       | ✅        | ✅        |
| Qwen3 Dense    | ✅       | ✅        | ✅        |
| Qwen3 MoE      | ✅       | ✅        | ✅        |
| GPT OSS        | ✅       | ✅        | 🚧       |
| Deepseek V3    | ✅       | ✅        | 🚧       |
| KIMI K2        | ✅       | ✅        | 🚧       |

### Algorithm

The algorithm component is under active development, and community contributions are welcome!

**Implemented**

-   ✅ **Multimodal Pre-training** - Full support for vision-language model training
-   ✅ **Multimodal Supervised Fine-tuning** - Optimized for instruction following
-   ✅ [GRPO](https://arxiv.org/pdf/2402.03300) - Group Relative Policy Optimization

**Coming Soon**

-   🔄 [MPO](https://arxiv.org/pdf/2411.10442) - Mixed Preference Optimization
-   🔄 [DAPO](https://arxiv.org/pdf/2503.14476) - Dynamic Sampling Policy Optimization
-   🔄 **Multi-turn Agentic RL** - Advanced agent training capabilities

### Inference Engine Integration

Seamless deployment with leading inference frameworks:

-   [x] LMDeploy
-   [ ] vLLM
-   [ ] SGLang

### Data Preparation

-   You can use [GraphGen](https://github.com/open-sciencelab/GraphGen) to create synthetic data for fine-tuning.

## Contributing

Contributions to XTuner are highly appreciated. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for details.

## Acknowledgements

XTuner V1's development was inspired by the open-source community. We thank the following projects:

**Training Engine:**

-   [Torchtitan](https://github.com/pytorch/torchtitan) - A PyTorch native platform for training generative AI models
-   [Deepspeed](https://github.com/deepspeedai/DeepSpeed) - Microsoft's deep learning optimization library
-   [MindSpeed](https://gitee.com/ascend/MindSpeed) - Ascend's high-performance training acceleration library
-   [Megatron](https://github.com/NVIDIA/Megatron-LM) - NVIDIA's large-scale transformer training framework

**Reinforcement Learning:**

-   [veRL](https://github.com/volcengine/verl) - Volcano Engine Reinforcement Learning for LLMs
-   [SLIME](https://github.com/THUDM/slime) - THU's scalable RLHF implementation
-   [AReal](https://github.com/inclusionAI/AReaL) - Ant Reasoning Reinforcement Learning for LLMs
-   [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) - An Easy-to-use, Scalable and High-performance RLHF Framework based on Ray

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

This project is released under the [Apache License 2.0](LICENSE). Also, adhere to the licenses of the models and datasets used.