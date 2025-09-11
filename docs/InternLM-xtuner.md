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

# XTuner: The Next-Generation Training Engine for Ultra-Large MoE Models

**XTuner is a powerful, efficient, and scalable training engine designed for fine-tuning large language models, particularly MoE models.**  [Explore the XTuner repository](https://github.com/InternLM/xtuner).

## Key Features

*   **Dropless Training:** Train massive MoE models without complex expert parallelism, optimizing efficiency and scalability.
    *   Scales to 200B+ parameter MoE models with ease.
    *   Efficient parallelism strategies for improved performance.

*   **Long Sequence Support:**  Train with extended sequence lengths for improved performance on complex tasks.
    *   Supports 64k sequence lengths for 200B MoE models without sequence parallelism.
    *   Full DeepSpeed Ulysses sequence parallelism support.
    *   Maintains stability during long sequence training with expert load balancing.

*   **Superior Efficiency:**  Achieve groundbreaking training performance for MoE models.
    *   Supports MoE training up to 1 Trillion parameters.
    *   Outperforms traditional 3D parallel schemes for MoE models above 200B scale.
    *   Optimized for Ascend A3 Supernode hardware.

## Speed Benchmarks

<div align=center>
  <img src="https://github.com/user-attachments/assets/fa42d587-068d-427b-b88c-25a164b3511c" style="width:80%">
</div>

## What's New

*   **\[2025/09]**: XTuner V1 Released! A Next-Generation Training Engine Built for Ultra-Large MoE Models

## Roadmap

XTuner V1 is continuously improving training efficiency for pre-training, instruction fine-tuning, and reinforcement learning of ultra-large MoE models, with special focus on Ascend NPU optimization.

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

We are actively evolving the algorithm component.  Community contributions are welcome.

**Implemented**

*   ✅ **Multimodal Pre-training:** Full support for vision-language model training
*   ✅ **Multimodal Supervised Fine-tuning:** Optimized for instruction following
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

## Contributing

We welcome contributions to XTuner!  See [CONTRIBUTING.md](.github/CONTRIBUTING.md) for guidelines.

## Acknowledgements

XTuner V1 has been inspired by and built upon the great work of the open-source community. We are very grateful to the following pioneering projects:

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

We appreciate all the contributors and maintainers of these projects.

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

This project is released under the [Apache License 2.0](LICENSE).  Please also adhere to the licenses of any models and datasets you use.