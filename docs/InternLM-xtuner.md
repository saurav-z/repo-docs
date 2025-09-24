<div align="center">
  <img src="https://github.com/InternLM/lmdeploy/assets/36994684/0cf8d00f-e86b-40ba-9b54-dc8f1bc6c8d8" width="600"/>
  <br /><br />
  <a href="https://github.com/InternLM/xtuner">
    <img src="https://img.shields.io/github/stars/InternLM/xtuner?style=social" alt="GitHub stars">
  </a>
  <a href="https://github.com/InternLM/xtuner/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/InternLM/xtuner.svg" alt="License">
  </a>
  <a href="https://pypi.org/project/xtuner/">
    <img src="https://img.shields.io/pypi/v/xtuner" alt="PyPI version">
  </a>
  <a href="https://pypi.org/project/xtuner/">
    <img src="https://static.pepy.tech/badge/xtuner" alt="Downloads">
  </a>
  <a href="https://github.com/InternLM/xtuner/issues">
    <img src="https://img.shields.io/github/issues-closed-raw/InternLM/xtuner" alt="Closed Issues">
  </a>
  <a href="https://github.com/InternLM/xtuner/issues">
    <img src="https://img.shields.io/github/issues-raw/InternLM/xtuner" alt="Open Issues">
  </a>
  <br/>

  Join the community:
  <br/>
  <a href="https://cdn.vansin.top/internlm/xtuner.jpg">
    <img src="https://img.shields.io/badge/-grey?style=social&logo=wechat&label=WeChat" alt="WeChat">
  </a>
  <a href="https://twitter.com/intern_lm">
    <img src="https://img.shields.io/badge/-grey?style=social&logo=twitter&label=Twitter" alt="Twitter">
  </a>
  <a href="https://discord.gg/xa29JuW87d">
    <img src="https://img.shields.io/badge/-grey?style=social&logo=discord&label=Discord" alt="Discord">
  </a>
  <br/>
  Explore our models on:
  <br/>
  <a href="https://huggingface.co/xtuner">
    <img src="https://img.shields.io/badge/-gery?style=social&label=%F0%9F%A4%97%20Huggingface" alt="Hugging Face">
  </a>
  <a href="https://www.modelscope.cn/organization/xtuner">
    <img src="https://img.shields.io/badge/-gery?style=social&label=%F0%9F%A4%96%20ModelScope" alt="ModelScope">
  </a>
  <a href="https://openxlab.org.cn/usercenter/xtuner">
    <img src="https://img.shields.io/badge/-gery?style=social&label=%F0%9F%A7%AA%20OpenXLab" alt="OpenXLab">
  </a>
  <a href="https://www.wisemodel.cn/organization/xtuner">
    <img src="https://img.shields.io/badge/-gery?style=social&label=%F0%9F%A7%A0%20WiseModel" alt="WiseModel">
  </a>

  <br/>
  [English](README.md) | [简体中文](README_zh-CN.md)
</div>

## XTuner: Unleash the Power of Efficient LLM Training

XTuner is a cutting-edge, next-generation training engine designed for ultra-large-scale Mixture-of-Experts (MoE) models, enabling efficient and scalable training.  [Explore the original repository](https://github.com/InternLM/xtuner).

## Key Features

*   **Dropless Training for Scalability:**
    *   Train massive MoE models (up to 200B parameters) without requiring expert parallelism, simplifying training complexity.
    *   Efficient parallelism strategies reduce the need for extensive expert parallelism, optimizing resource utilization.
*   **Long Sequence Support:**
    *   Train 200B MoE models on sequence lengths up to 64k without sequence parallelism, thanks to advanced memory optimization.
    *   Full support for DeepSpeed Ulysses sequence parallelism enables linearly scalable maximum sequence length.
    *   Maintains stability even with expert load imbalances during long sequence training.
*   **Superior Efficiency:**
    *   Supports MoE model training up to 1 trillion parameters.
    *   Achieves groundbreaking FSDP training throughput, surpassing traditional 3D parallel schemes for MoE models exceeding 200B parameters.
    *   Optimized for Ascend A3 Supernode hardware, delivering exceptional training efficiency that exceeds NVIDIA H800.

## Speed Benchmark

<div align=center>
  <img src="https://github.com/user-attachments/assets/fa42d587-068d-427b-b88c-25a164b3511c" style="width:80%">
</div>

## News

*   **\[2025/09]** XTuner V1 Released! A Next-Generation Training Engine Built for Ultra-Large MoE Models

## XTuner V1 Deep Dive

XTuner V1 is revolutionizing LLM training, offering a streamlined approach specifically tailored for the demands of ultra-large-scale MoE models. Unlike conventional 3D parallel training architectures, XTuner V1 focuses on efficiency and performance.

### Key Advantages

*   **Simplified Training:** Train massive models with reduced complexity.
*   **Optimized Parallelism:** Enables efficient training across diverse hardware.
*   **Enhanced Memory Efficiency:** Train longer sequences, unlocking greater model potential.
*   **Industry-Leading Performance:** Experience unparalleled throughput and efficiency gains.

## Roadmap

XTuner V1 is dedicated to continuously improving the training efficiency of pre-training, instruction fine-tuning, and reinforcement learning for ultra-large MoE models, with a special focus on Ascend NPU optimization.

### Training Engine

XTuner V1 strives to become a versatile training backend that seamlessly integrates into the broader open-source ecosystem.

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

The algorithm component is actively evolving.  Community contributions are highly encouraged!

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

*   Use [GraphGen](https://github.com/open-sciencelab/GraphGen) to create synthetic data for fine-tuning.

## Contributing

We welcome all contributions to XTuner.  Please review the [CONTRIBUTING.md](.github/CONTRIBUTING.md) for our contributing guidelines.

## Acknowledgements

The development of XTuner V1's training engine has been greatly inspired by and built upon the excellent work of the open-source community. We extend our sincere gratitude to the following pioneering projects:

**Training Engine:**

*   [Torchtitan](https://github.com/pytorch/torchtitan) - A PyTorch native platform for training generative AI models
*   [Deepspeed](https://github.com/deepspeedai/DeepSpeed) - Microsoft's deep learning optimization library
*   [MindSpeed](https://gitee.com/ascend/MindSpeed) - Ascend's high-performance training acceleration library
*   [Megatron](https://github.com/NVIDIA/Megatron-LM) - NVIDIA's large-scale transformer training framework

**Reinforcement Learning:**

XTuner V1's reinforcement learning capabilities have been enhanced through insights and best practices from:

*   [veRL](https://github.com/volcengine/verl) - Volcano Engine Reinforcement Learning for LLMs
*   [SLIME](https://github.com/THUDM/slime) - THU's scalable RLHF implementation
*   [AReal](https://github.com/inclusionAI/AReaL) - Ant Reasoning Reinforcement Learning for LLMs
*   [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) - An Easy-to-use, Scalable and High-performance RLHF Framework based on Ray

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

This project is released under the [Apache License 2.0](LICENSE). Please also adhere to the Licenses of models and datasets being used.