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

## XTuner: Fine-tune LLMs Efficiently with the Next-Generation Training Engine

XTuner is a powerful and versatile toolkit designed for efficient and scalable fine-tuning of Large Language Models (LLMs), enabling researchers and developers to push the boundaries of AI.  [Visit the original repository](https://github.com/InternLM/xtuner) for more details.

## Key Features

*   **Dropless Training:**  Train massive MoE models (up to 1T parameters) efficiently.
    *   Scalable without complex expert parallelism.
    *   Optimized parallelism for efficient training.
*   **Long Sequence Support:** Train on 64k sequence lengths for 200B MoE models.
    *   Memory-efficient design for extended sequence handling.
    *   Full DeepSpeed Ulysses support.
    *   Maintains stability during training.
*   **Superior Efficiency:**
    *   Supports MoE training up to 1T parameters.
    *   Breakthrough performance exceeding traditional 3D parallel schemes.
    *   Optimized for Ascend A3 Supernode hardware.
*   **Multimodal Support:** Pre-training and fine-tuning capabilities for vision-language models.
*   **Algorithm Support**:
    * GRPO
    * MPO (Coming Soon)
    * DAPO (Coming Soon)
    * Multi-turn Agentic RL (Coming Soon)

## Roadmap

XTuner V1 is constantly evolving, focusing on efficiency and optimization, particularly for Ascend NPU.

### Training Engine Support

| Model        | GPU(FP8) | GPU(BF16) | NPU(BF16) |
|--------------|----------|-----------|-----------|
| Intern S1    | ✅        | ✅         | ✅         |
| Intern VL    | ✅        | ✅         | ✅         |
| Qwen3 Dense  | ✅        | ✅         | ✅         |
| Qwen3 MoE    | ✅        | ✅         | ✅         |
| GPT OSS      | ✅        | ✅         | 🚧         |
| Deepseek V3  | ✅        | ✅         | 🚧         |
| KIMI K2      | ✅        | ✅         | 🚧         |

### Inference Engine Integration

*   ✅ LMDeploy
*   ☐ vLLM
*   ☐ SGLang

### Data Preparation

*   Use [GraphGen](https://github.com/open-sciencelab/GraphGen) for synthetic data generation.

## Contributing

We welcome contributions! See [CONTRIBUTING.md](.github/CONTRIBUTING.md) for guidelines.

## Acknowledgements

XTuner builds upon the work of several open-source projects. We are grateful to:

*   **Training Engine:** Torchtitan, Deepspeed, MindSpeed, Megatron.
*   **Reinforcement Learning:** veRL, SLIME, AReal, OpenRLHF.

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

This project is licensed under the [Apache License 2.0](LICENSE).