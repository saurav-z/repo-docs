# Slime: Supercharge LLM Post-Training with RL Scaling

**Slime is a cutting-edge, SGLang-native framework designed to revolutionize LLM post-training through Reinforcement Learning (RL) scaling.** Explore the original repository [here](https://github.com/THUDM/slime).

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://thudm.github.io/slime/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/THUDM/slime)

## Key Features

*   **High-Performance Training:** Experience efficient LLM training in various modes, optimized through seamless integration with Megatron and SGLang.
*   **Flexible Data Generation:** Design custom data generation workflows with ease using our flexible data generation interfaces and server-based engines.

## Why Choose Slime?

Slime provides a robust framework for RL-based LLM post-training, enabling you to enhance your language models with improved performance and capabilities. It's a powerful solution for researchers and developers looking to push the boundaries of LLM training.

## Core Components

*   **Training (Megatron):** Manages the primary training process, reading data and synchronizing parameters.
*   **Rollout (SGLang + Router):** Generates new training data, including rewards and verifier outputs, and stores it for training.
*   **Data Buffer:** Acts as a bridge between modules, managing prompt initialization, custom data, and rollout generation.

## Quick Start

Get started quickly with our comprehensive guide: [Quick Start Guide](./docs/en/get_started/quick_start.md).

## Further Resources

*   **Blogs:**
    *   Our vision: [slime: An SGLang-Native Post-Training Framework for RL Scaling](https://lmsys.org/blog/2025-07-09-slime/).
    *   Our ideas on agentic training: [Agent-Oriented Design: An Asynchronous and Decoupled Framework for Agentic RL](https://www.notion.so/Agent-Oriented-Design-An-Asynchronous-and-Decoupled-Framework-for-Agentic-RL-2278e692d081802cbdd5d37cef76a547).
    *   slime has served as the RL framework for GLM-4.5: [GLM-4.5: Reasoning, Coding, and Agentic Abilities](https://z.ai/blog/glm-4.5)
*   **Examples:** Explore additional use cases in the [examples](examples/) directory.
*   **Usage Documentation:** Detailed instructions in [Usage Documentation](docs/en/get_started/usage.md).
*   **Debugging Guide:** Helpful tips in [Debugging Guide](docs/en/developer_guide/debug.md).
*   **Q&A:** Find answers to frequently asked questions in the [Q&A](docs/en/get_started/qa.md).

## Arguments

Slime arguments are categorized into three types:

1.  **Megatron Arguments:** Inherited directly from Megatron.  Configure using standard Megatron arguments (e.g., `--tensor-model-parallel-size 2`).
2.  **SGLang Arguments:**  Use the `--sglang-` prefix for SGLang arguments (e.g., `--sglang-mem-fraction-static`).
3.  **Slime-Specific Arguments:** Consult [slime/utils/arguments.py](slime/utils/arguments.py) for details.

## Contributing

We welcome contributions!  Submit an Issue or PR to suggest new features, provide performance feedback, or share user experience feedback.

*   Use [pre-commit](https://pre-commit.com/) for code style consistency:

    ```bash
    apt install pre-commit -y
    pre-commit install
    ```

## Acknowledgements

Special thanks to the following projects and communities: SGLang, Megatron‑LM, mbridge, OpenRLHF, veRL, Pai-Megatron-Patch and others.