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

# XTuner: Your Toolkit for Efficient Large Language Model (LLM) Fine-tuning

**XTuner is a next-generation LLM training engine designed to efficiently fine-tune ultra-large-scale models, especially MoE (Mixture of Experts) models.**  Learn more about XTuner on [GitHub](https://github.com/InternLM/xtuner).

## Key Features

*   **Dropless Training:**
    *   Train 200B+ parameter MoE models without expert parallelism.
    *   Optimized parallelism for efficient training.
*   **Long Sequence Support:**
    *   Train 200B MoE models with 64k sequence lengths.
    *   Full support for DeepSpeed Ulysses sequence parallelism.
    *   Maintains stability even with expert load imbalance.
*   **Superior Efficiency:**
    *   Supports MoE training up to 1T parameters.
    *   Achieves breakthrough FSDP training throughput.
    *   Hardware optimization on Ascend A3 Supernode.

## Speed Benchmark

<div align=center>
  <img src="https://github.com/user-attachments/assets/fa42d587-068d-427b-b88c-25a164b3511c" style="width:80%">
</div>

## What's New

*   **[2025/09]** XTuner V1 Released! A Next-Generation Training Engine.

## XTuner V1: Deep Dive

XTuner V1 is built for training ultra-large-scale MoE models, offering significant advantages over traditional architectures.  It is optimized for the current academic research landscape.

### Core Benefits

*   **Scalability:** Train massive models efficiently without the complexity of traditional approaches.
*   **Memory Efficiency:** Optimized for long sequence lengths.
*   **Performance:** Optimized to get the most out of your hardware.

<div align=center>
  <img src="https://github.com/user-attachments/assets/98519a93-1ce8-49f0-a7ab-d7968c9d67a6" style="width:90%">
</div>

## Roadmap

XTuner V1 is constantly evolving, focusing on improvements for pre-training, instruction fine-tuning, and reinforcement learning of ultra-large MoE models, with a focus on Ascend NPU optimization.

### Training Engine

Our vision is to make XTuner V1 a versatile training backend that seamlessly integrates with the broader open-source ecosystem.

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

We actively welcome community contributions to our algorithm component!

**Implemented**

*   ✅ **Multimodal Pre-training:** Supports vision-language model training.
*   ✅ **Multimodal Supervised Fine-tuning:** Optimized for instruction following.
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

*   Use [GraphGen](https://github.com/open-sciencelab/GraphGen) to create synthetic data.

## Contributing

We welcome contributions!  See [CONTRIBUTING.md](.github/CONTRIBUTING.md) for guidelines.

## Acknowledgements

XTuner V1 is built upon the work of these projects:

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

We appreciate the contributions of all the projects mentioned above.

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

Released under the [Apache License 2.0](LICENSE).  Adhere to the licenses of any models or datasets used.