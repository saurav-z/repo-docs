<div align="center">
  <img src="https://github.com/InternLM/lmdeploy/assets/36994684/0cf8d00f-e86b-40ba-9b54-dc8f1bc6c8d8" width="600"/>
  <br /><br />

[![GitHub Repo stars](https://img.shields.io/github/stars/InternLM/xtuner?style=social)](https://github.com/InternLM/xtuner/stargazers)
[![license](https://img.shields.io/github/license/InternLM/xtuner.svg)](https://github.com/InternLM/xtuner/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/xtuner)](https://pypi.org/project/xtuner/)
[![Downloads](https://static.pepy.tech/badge/xtuner)](https://pypi.org/project/xtuner/)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/InternLM/xtuner)](https://github.com/InternLM/xtuner/issues)
[![open issues](https://img.shields.io/github/issues-raw/InternLM/xtuner)](https://github.com/InternLM/xtuner/issues)

👋 Join the XTuner community on [![Static Badge](https://img.shields.io/badge/-grey?style=social&logo=wechat&label=WeChat)](https://cdn.vansin.top/internlm/xtuner.jpg)
[![Static Badge](https://img.shields.io/badge/-grey?style=social&logo=twitter&label=Twitter)](https://twitter.com/intern_lm)
[![Static Badge](https://img.shields.io/badge/-grey?style=social&logo=discord&label=Discord)](https://discord.gg/xa29JuW87d)

🔍 Explore our models on
[![Static Badge](https://img.shields.io/badge/-gery?style=social&label=🤗%20Huggingface)](https://huggingface.co/xtuner)
[![Static Badge](https://img.shields.io/badge/-gery?style=social&label=🤖%20ModelScope)](https://www.modelscope.cn/organization/xtuner)
[![Static Badge](https://img.shields.io/badge/-gery?style=social&label=🧰%20OpenXLab)](https://openxlab.org.cn/usercenter/xtuner)
[![Static Badge](https://img.shields.io/badge/-gery?style=social&label=🧠%20WiseModel)](https://www.wisemodel.cn/organization/xtuner)

English | [简体中文](README_zh-CN.md)

</div>

# XTuner: Unleash the Power of Ultra-Large MoE Models with Efficient Training

XTuner is a cutting-edge training engine designed for efficiently fine-tuning and training ultra-large Mixture-of-Experts (MoE) models; [explore the repo](https://github.com/InternLM/xtuner).

## Key Features

*   **Dropless Training:** Train massive MoE models (200B+ parameters) with simplified parallelism and optimized strategies.
*   **Long Sequence Support:** Efficiently handle 64k sequence lengths without sequence parallelism through advanced memory optimization techniques.
*   **Superior Efficiency:** Achieve breakthrough performance in MoE training, surpassing traditional 3D parallel schemes, with support for up to 1T parameters.
*   **Hardware Optimization:** Optimized for Ascend NPU, with efficiency exceeding NVIDIA H800.
*   **Comprehensive Support:** Supports various model architectures and training methods.
*   **Active Development:** Continuously improved for pre-training, instruction fine-tuning, and reinforcement learning.

## Speed Benchmark

<div align=center>
  <img src="https://github.com/user-attachments/assets/fa42d587-068d-427b-b88c-25a164b3511c" style="width:80%">
</div>

## XTuner V1: Next-Generation LLM Training

XTuner V1 is a next-generation training engine specifically designed for ultra-large-scale MoE models. It is optimized for the mainstream MoE training scenarios prevalent in today's academic research.

### Core Advantages

*   **Scalability Without Complexity:** Train massive MoE models without the complexities of expert parallelism
*   **Optimized Parallelism:** Smaller expert parallelism dimensions lead to more efficient Dropless training.
*   **Memory Efficiency:** Advanced techniques enable training on long sequences without sequence parallelism
*   **Flexible Scaling:** Full support for DeepSpeed Ulysses sequence parallelism.
*   **Robust Performance:** Maintains stability even with expert load imbalances during long sequence training
*   **Supports MoE Training:** Up to 1T parameters
*   **Performance Breakthrough:** Exceeds traditional 3D parallel schemes.
*   **Hardware Optimized:** Training efficiency exceeding NVIDIA H800

## Roadmap

XTuner V1 is committed to continuously improving training efficiency for pre-training, instruction fine-tuning, and reinforcement learning of ultra-large MoE models, with special focus on Ascend NPU optimization.

### 🚀 Training Engine

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

## 🤝 Contributing

We welcome contributions to XTuner! Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for guidelines.

## 🙏 Acknowledgements

We are deeply grateful to the open-source community, and especially to these pioneering projects:

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

This project is released under the [Apache License 2.0](LICENSE). Please also adhere to the Licenses of models and datasets being used.