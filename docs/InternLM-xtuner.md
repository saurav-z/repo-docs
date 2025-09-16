<div align="center">
  <img src="https://github.com/InternLM/lmdeploy/assets/36994684/0cf8d00f-e86b-40ba-9b54-dc8f1bc6c8d8" width="600" alt="XTuner Logo">
  <br /><br />

[![GitHub Repo stars](https://img.shields.io/github/stars/InternLM/xtuner?style=social)](https://github.com/InternLM/xtuner/stargazers)
[![license](https://img.shields.io/github/license/InternLM/xtuner.svg)](https://github.com/InternLM/xtuner/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/xtuner)](https://pypi.org/project/xtuner/)
[![Downloads](https://static.pepy.tech/badge/xtuner)](https://pypi.org/project/xtuner/)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/InternLM/xtuner)](https://github.com/InternLM/xtuner/issues)
[![open issues](https://img.shields.io/github/issues-raw/InternLM/xtuner)](https://github.com/InternLM/xtuner/issues)

👋 Join us on:
[![Static Badge](https://img.shields.io/badge/-grey?style=social&logo=wechat&label=WeChat)](https://cdn.vansin.top/internlm/xtuner.jpg)
[![Static Badge](https://img.shields.io/badge/-grey?style=social&logo=twitter&label=Twitter)](https://twitter.com/intern_lm)
[![Static Badge](https://img.shields.io/badge/-grey?style=social&logo=discord&label=Discord)](https://discord.gg/xa29JuW87d)

🔍 Explore our models on:
[![Static Badge](https://img.shields.io/badge/-gery?style=social&label=🤗%20Huggingface)](https://huggingface.co/xtuner)
[![Static Badge](https://img.shields.io/badge/-gery?style=social&label=🤖%20ModelScope)](https://www.modelscope.cn/organization/xtuner)
[![Static Badge](https://img.shields.io/badge/-gery?style=social&label=🧰%20OpenXLab)](https://openxlab.org.cn/usercenter/xtuner)
[![Static Badge](https://img.shields.io/badge/-gery?style=social&label=🧠%20WiseModel)](https://www.wisemodel.cn/organization/xtuner)

English | [简体中文](README_zh-CN.md)

</div>

## XTuner: The Next-Generation Training Engine for Ultra-Large Language Models

XTuner is a cutting-edge training engine designed to efficiently fine-tune and train ultra-large language models, offering unparalleled scalability and performance.  [Visit the original repository](https://github.com/InternLM/xtuner) for more information.

## Key Features

*   **Dropless Training:** Train massive MoE models with simplified parallelism and optimized efficiency.
    *   Scalable training without expert parallelism for models up to 200B parameters.
    *   Efficient expert parallelism for larger models (up to 600B).
    *   Optimized parallelism strategy for enhanced Dropless training.

*   **Long Sequence Support:** Train on extended sequence lengths with memory-efficient designs.
    *   Train 200B MoE models on 64k sequence lengths without sequence parallelism.
    *   Full support for DeepSpeed Ulysses sequence parallelism.
    *   Maintains stability during long sequence training with expert load imbalance.

*   **Superior Efficiency:** Achieve breakthrough performance and support for models up to 1T parameters.
    *   Supports MoE training up to 1T parameters.
    *   FSDP training throughput surpasses traditional 3D parallel schemes for MoE models above 200B scale.
    *   Hardware optimization on Ascend A3 Supernode exceeds NVIDIA H800 performance.

<div align=center>
  <img src="https://github.com/user-attachments/assets/98519a93-1ce8-49f0-a7ab-d7968c67a6" style="width:90%">
</div>

## Roadmap

XTuner V1 is continuously evolving to improve training efficiency and expand model support, with a focus on Ascend NPU optimization.

### Training Engine

| Model         | GPU(FP8) | GPU(BF16) | NPU(BF16) |
|---------------|----------|-----------|-----------|
| Intern S1     | ✅       | ✅        | ✅        |
| Intern VL     | ✅       | ✅        | ✅        |
| Qwen3 Dense   | ✅       | ✅        | ✅        |
| Qwen3 MoE     | ✅       | ✅        | ✅        |
| GPT OSS       | ✅       | ✅        | 🚧        |
| Deepseek V3   | ✅       | ✅        | 🚧        |
| KIMI K2       | ✅       | ✅        | 🚧        |

### Algorithm

*   **Implemented**
    *   ✅ Multimodal Pre-training - Full support for vision-language model training
    *   ✅ Multimodal Supervised Fine-tuning - Optimized for instruction following
    *   ✅ [GRPO](https://arxiv.org/pdf/2402.03300) - Group Relative Policy Optimization

*   **Coming Soon**
    *   🔄 [MPO](https://arxiv.org/pdf/2411.10442) - Mixed Preference Optimization
    *   🔄 [DAPO](https://arxiv.org/pdf/2503.14476) - Dynamic Sampling Policy Optimization
    *   🔄 Multi-turn Agentic RL - Advanced agent training capabilities

### Inference Engine Integration

*   [x] LMDeploy
*   [ ] vLLM
*   [ ] SGLang

### Data Preparation

*   Use [GraphGen](https://github.com/open-sciencelab/GraphGen) to create synthetic data for fine-tuning.

## Contributing

Contributions to XTuner are welcome!  See [CONTRIBUTING.md](.github/CONTRIBUTING.md) for guidelines.

## Acknowledgements

XTuner V1 is built upon the work of the open-source community. We thank these projects:

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

This project is licensed under the [Apache License 2.0](LICENSE).  Please also adhere to the licenses of the models and datasets used.