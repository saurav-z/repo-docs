<!-- Improved & Summarized README with SEO Optimization -->

<div align="center">
  <img src="https://github.com/InternLM/lmdeploy/assets/36994684/0cf8d00f-e86b-40ba-9b54-dc8f1bc6c8d8" width="600" alt="XTuner Logo">
  <br /><br />

[![GitHub Repo stars](https://img.shields.io/github/stars/InternLM/xtuner?style=social)](https://github.com/InternLM/xtuner/stargazers)
[![license](https://img.shields.io/github/license/InternLM/xtuner.svg)](https://github.com/InternLM/xtuner/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/xtuner)](https://pypi.org/project/xtuner/)
[![Downloads](https://static.pepy.tech/badge/xtuner)](https://pypi.org/project/xtuner/)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/InternLM/xtuner)](https://github.com/InternLM/xtuner/issues)
[![open issues](https://img.shields.io/github/issues-raw/InternLM/xtuner)](https://github.com/InternLM/xtuner/issues)

👋 Join the XTuner community on: [![Static Badge](https://img.shields.io/badge/-grey?style=social&logo=wechat&label=WeChat)](https://cdn.vansin.top/internlm/xtuner.jpg)
[![Static Badge](https://img.shields.io/badge/-grey?style=social&logo=twitter&label=Twitter)](https://twitter.com/intern_lm)
[![Static Badge](https://img.shields.io/badge/-grey?style=social&logo=discord&label=Discord)](https://discord.gg/xa29JuW87d)

🔍 Explore models trained with XTuner on:
[![Static Badge](https://img.shields.io/badge/-gery?style=social&label=🤗%20Huggingface)](https://huggingface.co/xtuner)
[![Static Badge](https://img.shields/badge/-gery?style=social&label=🤖%20ModelScope)](https://www.modelscope.cn/organization/xtuner)
[![Static Badge](https://img.shields.io/badge/-gery?style=social&label=🧰%20OpenXLab)](https://openxlab.org.cn/usercenter/xtuner)
[![Static Badge](https://img.shields.io/badge/-gery?style=social&label=🧠%20WiseModel)](https://www.wisemodel.cn/organization/xtuner)

English | [简体中文](README_zh-CN.md)

</div>

## 🚀 **XTuner: The Next-Generation Training Engine for Ultra-Large MoE Models**

XTuner empowers researchers and developers to efficiently train and fine-tune massive Mixture-of-Experts (MoE) models, accelerating the path to cutting-edge AI.

## **Key Features**

*   **Dropless Training:**
    *   Scales efficiently to 200B+ parameter MoE models without complex expert parallelism.
    *   Optimized parallelism strategies enable efficient Dropless training.
*   **Long Sequence Support:**
    *   Trains 200B MoE models on 64k sequence lengths with memory optimization.
    *   Supports DeepSpeed Ulysses sequence parallelism.
    *   Maintains stability, even with expert load imbalances.
*   **Superior Efficiency:**
    *   Supports MoE training up to 1T parameters.
    *   Achieves leading FSDP training throughput.
    *   Optimized for Ascend NPU hardware.

## 🎉 **News**

*   **\[2025/09]** XTuner V1 Released!

## **XTuner V1: Deep Dive**

XTuner V1 is a state-of-the-art training engine tailored for ultra-large-scale MoE models.  It addresses the limitations of traditional 3D parallel training, offering optimized solutions for modern MoE training scenarios.

<div align=center>
  <img src="https://github.com/user-attachments/assets/98519a93-1ce8-49f0-a7ab-d7968c9d67a6" style="width:90%" alt="XTuner Performance Chart">
</div>

## 🔥 **Roadmap & Future Development**

XTuner V1 is dedicated to improving training efficiency for pre-training, instruction fine-tuning, and reinforcement learning (RL) of ultra-large MoE models, with a special focus on Ascend NPU optimization.

### **🚀 Training Engine: Model Support**

XTuner V1 provides comprehensive support for training various model architectures, including:

| Model           | GPU(FP8) | GPU(BF16) | NPU(BF16) |
|-----------------|-----------|-----------|-----------|
| Intern S1       | ✅        | ✅        | ✅        |
| Intern VL       | ✅        | ✅        | ✅        |
| Qwen3 Dense     | ✅        | ✅        | ✅        |
| Qwen3 MoE       | ✅        | ✅        | ✅        |
| GPT OSS         | ✅        | ✅        | 🚧        |
| Deepseek V3     | ✅        | ✅        | 🚧        |
| KIMI K2         | ✅        | ✅        | 🚧        |

### **🧠 Algorithm Development**

The XTuner V1 algorithm component is continuously evolving with community contributions welcome.

**Implemented Algorithms:**

*   ✅ Multimodal Pre-training (Vision-Language)
*   ✅ Multimodal Supervised Fine-tuning
*   ✅ GRPO (Group Relative Policy Optimization)

**Upcoming Algorithms:**

*   🔄 MPO (Mixed Preference Optimization)
*   🔄 DAPO (Dynamic Sampling Policy Optimization)
*   🔄 Multi-turn Agentic RL

### **⚡ Inference Engine Integration**

Future plans include seamless deployment with popular inference frameworks:

*   [x] LMDeploy
*   [ ] vLLM
*   [ ] SGLang

### **Data Preparation Resources**

*   You can use [GraphGen](https://github.com/open-sciencelab/GraphGen) to create synthetic data for fine-tuning.

## 🤝 **Contributing**

We encourage community contributions!  Please review the [CONTRIBUTING.md](.github/CONTRIBUTING.md) file for guidelines.

## 🙏 **Acknowledgements**

XTuner V1 development is inspired by and built upon the work of the open-source community.  We appreciate the following projects:

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

## 🖊️ **Citation**

```bibtex
@misc{2023xtuner,
    title={XTuner: A Toolkit for Efficiently Fine-tuning LLM},
    author={XTuner Contributors},
    howpublished = {\url{https://github.com/InternLM/xtuner}},
    year={2023}
}
```

## **License**

XTuner is released under the [Apache License 2.0](LICENSE). Adhere to the licenses of any models or datasets you use.

[Back to Top](#)  | [Original Repository](https://github.com/InternLM/xtuner)