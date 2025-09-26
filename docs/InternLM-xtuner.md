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

# XTuner: The Next-Generation LLM Training Engine for Ultra-Large Models

XTuner is a powerful and efficient toolkit designed for fine-tuning large language models (LLMs), enabling researchers and developers to train cutting-edge models at scale.  [Visit the original repository](https://github.com/InternLM/xtuner) to learn more.

## Key Features

*   **Dropless Training:** Train massive MoE models without the complexities of traditional 3D parallelism.
    *   Scalable up to 200B-scale MoE models without expert parallelism; 600B models require only intra-node expert parallelism.
    *   Optimized parallelism strategy reduces expert parallelism dimension for efficient training.
*   **Long Sequence Support:**  Train on extended sequences with memory efficiency.
    *   Train 200B MoE models on 64k sequence lengths without sequence parallelism.
    *   Full support for DeepSpeed Ulysses sequence parallelism.
    *   Maintains stability even with expert load imbalances.
*   **Superior Efficiency:** Achieve breakthrough performance in training.
    *   Supports MoE training up to 1T parameters.
    *   Achieves FSDP training throughput exceeding traditional 3D parallel schemes for MoE models above 200B scale.
    *   Optimized for Ascend A3 Supernode hardware.

## Speed Benchmark

<div align=center>
  <img src="https://github.com/user-attachments/assets/fa42d587-068d-427b-b88c-25a164b3511c" style="width:80%">
</div>

## News

*   **\[2025/09]** XTuner V1 Released! A Next-Generation Training Engine Built for Ultra-Large MoE Models

## XTuner V1: Deep Dive

XTuner V1 is a cutting-edge training engine specifically designed for training ultra-large-scale Mixture of Experts (MoE) models. It departs from traditional 3D parallel training architectures, focusing on optimization for the current MoE training landscape.

### Training Engine Roadmap

XTuner V1 is focused on continuously improving training efficiency for pre-training, instruction fine-tuning, and reinforcement learning of ultra-large MoE models, particularly focusing on Ascend NPU optimization.

|   Model    |  GPU(FP8) | GPU(BF16)| NPU(BF16) |
|------------|-----------|----------|-----------|
| Intern S1  |    ✅     |    ✅    |    ✅     |
| Intern VL  |    ✅     |    ✅    |    ✅     |
| Qwen3 Dense|    ✅     |    ✅    |    ✅     |
| Qwen3 MoE  |    ✅     |    ✅    |    ✅     |
| GPT OSS    |    ✅     |    ✅    |    🚧     |
| Deepseek V3|    ✅     |    ✅    |    🚧     |
| KIMI K2    |    ✅     |    ✅    |    🚧     |

### Algorithm Component

The algorithm component is continuously evolving. Contributions are welcome.

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

*   You can use [GraphGen](https://github.com/open-sciencelab/GraphGen) to create synthetic data for fine-tuning.

## Contributing

We welcome contributions to XTuner!  Please review the [CONTRIBUTING.md](.github/CONTRIBUTING.md) file for guidelines.

## Acknowledgements

XTuner V1's development has been inspired by and built upon the work of the open-source community. We acknowledge the following projects:

**Training Engine:**

*   [Torchtitan](https://github.com/pytorch/torchtitan) - A PyTorch native platform for training generative AI models
*   [Deepspeed](https://github.com/deepspeedai/DeepSpeed) - Microsoft's deep learning optimization library
*   [MindSpeed](https://gitee.com/ascend/MindSpeed) - Ascend's high-performance training acceleration library
*   [Megatron](https://github.com/NVIDIA/Megatron-LM) - NVIDIA's large-scale transformer training framework

**Reinforcement Learning:**

*   [veRL](https://github.com/volcengine/verl) - Volcano Engine Reinforcement Learning for LLMs
*   [SLIME](https://github.com/THUDM/slime) - THU's scalable RLHF implementation
*   [AReal](https://github.com/inclusionAI/AReaL) - Ant Reasoning Reinforcement Learning for LLMs
*   [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) - An Easy-to-use, Scalable and High-performance RLHF Framework based on Ray

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