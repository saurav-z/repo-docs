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

👋 Join the XTuner community on [![Static Badge](https://img.shields.io/badge/-grey?style=social&logo=wechat&label=WeChat)](https://cdn.vansin.top/internlm/xtuner.jpg)
[![Static Badge](https://img.shields.io/badge/-grey?style=social&logo=twitter&label=Twitter)](https://twitter.com/intern_lm)
[![Static Badge](https://img.shields.io/badge/-grey?style=social&logo=discord&label=Discord)](https://discord.gg/xa29JuW87d)

🔍 Explore our models on
[![Static Badge](https://img.shields.io/badge/-gery?style=social&label=🤗%20Huggingface)](https://huggingface.co/xtuner)
[![Static Badge](https://img.shields.io/badge/-gery?style=social&label=🤖%20ModelScope)](https://www.modelscope.cn/organization/xtuner)
[![Static Badge](https://img.shields.io/badge/-gery?style=social&label=🧰%20OpenXLab)](https://openxlab.org.cn/usercenter/xtuner)
[![Static Badge](https://img.shields.io/badge/-gery?style=social&label=🧠%20WiseModel)](https://www.wisemodel.cn/organization/xtuner)

English | [简体中文](README_zh-CN.md)

## **XTuner: Fine-tune and Train Large Language Models (LLMs) with Unprecedented Efficiency**

XTuner is a powerful, next-generation training engine designed to optimize the fine-tuning and training of ultra-large language models, especially Mixture of Experts (MoE) models.  Visit the original repository for more details: [https://github.com/InternLM/xtuner](https://github.com/InternLM/xtuner).

## **Key Features**

*   **Dropless Training:**
    *   Train massive MoE models (200B+ parameters) without the complexity of traditional 3D parallelism.
    *   Optimized parallelism strategies for efficient Dropless training.
*   **Long Sequence Support:**
    *   Train 200B MoE models on sequence lengths up to 64k without sequence parallelism, thanks to advanced memory optimization.
    *   Full DeepSpeed Ulysses support for linearly scalable sequence lengths.
    *   Maintains stable performance even with expert load imbalances during long sequence training.
*   **Superior Efficiency:**
    *   Supports MoE training up to 1 Trillion parameters.
    *   Achieves leading FSDP training throughput for MoE models exceeding 200B parameters.
    *   Optimized for Ascend A3 Supernode, exceeding NVIDIA H800 training efficiency.

## **Roadmap & Development**

XTuner V1 is dedicated to continuous improvement in the efficiency of pre-training, instruction fine-tuning, and reinforcement learning of ultra-large MoE models, with a focus on Ascend NPU optimization.

### **Training Engine**

XTuner V1 aims to be a versatile training backend, seamlessly integrating with the broader open-source ecosystem.

| Model        | GPU (FP8) | GPU (BF16) | NPU (BF16) |
|--------------|-----------|-----------|-----------|
| Intern S1    | ✅         | ✅          | ✅          |
| Intern VL    | ✅         | ✅          | ✅          |
| Qwen3 Dense  | ✅         | ✅          | ✅          |
| Qwen3 MoE    | ✅         | ✅          | ✅          |
| GPT OSS      | ✅         | ✅          | 🚧         |
| Deepseek V3  | ✅         | ✅          | 🚧         |
| KIMI K2      | ✅         | ✅          | 🚧         |

### **Algorithms**

The algorithm component is actively evolving. We welcome community contributions to scale your algorithms!

**Implemented**

*   ✅ **Multimodal Pre-training:** Full support for vision-language model training.
*   ✅ **Multimodal Supervised Fine-tuning:** Optimized for instruction following.
*   ✅ [GRPO](https://arxiv.org/pdf/2402.03300) - Group Relative Policy Optimization

**Coming Soon**

*   🔄 [MPO](https://arxiv.org/pdf/2411.10442) - Mixed Preference Optimization
*   🔄 [DAPO](https://arxiv.org/pdf/2503.14476) - Dynamic Sampling Policy Optimization
*   🔄 **Multi-turn Agentic RL:** Advanced agent training capabilities

### **Inference Engine Integration**

Seamless deployment with leading inference frameworks:

*   [x] LMDeploy
*   [ ] vLLM
*   [ ] SGLang

### **Data Preparation**

*   Use [GraphGen](https://github.com/open-sciencelab/GraphGen) to create synthetic data for fine-tuning.

## **Contributing**

We welcome contributions! Please refer to our [CONTRIBUTING.md](.github/CONTRIBUTING.md) for guidelines.

## **Acknowledgements**

XTuner V1's development has been inspired by the following projects:

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

## **Citation**

```bibtex
@misc{2023xtuner,
    title={XTuner: A Toolkit for Efficiently Fine-tuning LLM},
    author={XTuner Contributors},
    howpublished = {\url{https://github.com/InternLM/xtuner}},
    year={2023}
}
```

## **License**

This project is released under the [Apache License 2.0](LICENSE).  Please also adhere to the licenses of models and datasets you use.