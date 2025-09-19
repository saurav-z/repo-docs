# Slime: LLM Post-Training Framework for RL Scaling

**Slime** is a cutting-edge framework designed to revolutionize Large Language Model (LLM) post-training through Reinforcement Learning (RL) scaling, enabling advanced capabilities and efficient training. **[Learn more on GitHub](https://github.com/THUDM/slime)**

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://thudm.github.io/slime/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/THUDM/slime)

## Key Features of Slime

*   **High-Performance Training:**  Leverages the power of Megatron and SGLang for efficient training across various modes, optimizing LLM performance.
*   **Flexible Data Generation:** Enables the creation of tailored training datasets through custom data generation interfaces and server-based engines, promoting adaptability.

## Core Capabilities

Slime offers two key benefits for LLM post-training:

*   **Efficient RL Scaling:** Facilitates the scaling of RL for LLMs, enabling complex training scenarios.
*   **Enhanced Training Efficiency:** Combines Megatron and SGLang for effective training across multiple modalities.

## Architecture Overview

![Architecture Overview](./imgs/arch.png)

*   **Training (Megatron):**  Manages the main training process, pulls data from the Data Buffer, and synchronizes model parameters with the rollout module after each training cycle.
*   **Rollout (SGLang + Router):** Generates new data (including rewards and verifier outputs) and stores it in the Data Buffer, enhancing data generation workflows.
*   **Data Buffer:** Acts as a bridge, handling prompt initialization, custom data, and rollout generation methods for a seamless data flow.

## Getting Started

*   **Quick Start Guide:**  Access a comprehensive guide to quickly set up your environment, prepare data, launch training, and explore key code snippets.  [Quick Start Guide](./docs/en/get_started/quick_start.md)
*   **Usage Documentation:** For complete instructions on usage and configurations, please refer to the detailed [Usage Documentation](docs/en/get_started/usage.md).

##  Arguments Explained

Arguments in Slime are categorized for clarity:

1.  **Megatron Arguments:**  Configure Megatron parameters using `PYTHONPATH`, for example: `--tensor-model-parallel-size 2`.
2.  **SGLang Arguments:** Use SGLang arguments by prefixing them with `--sglang-`, e.g., `--sglang-mem-fraction-static`.
3.  **Slime-Specific Arguments:** Explore Slime's unique arguments in [slime/utils/arguments.py](slime/utils/arguments.py).

## Contributing & Development

We welcome contributions from the community to improve Slime!

*   **Contribution Guidelines:** Submit issues and pull requests for new features, performance enhancements, and feedback.
*   **Code Style:** Use [pre-commit](https://pre-commit.com/) to ensure consistent code style for your contributions.

    ```bash
    apt install pre-commit -y
    pre-commit install
    ```

*   **Debugging:** Consult the [Debugging Guide](docs/en/developer_guide/debug.md) for assistance.

## Resources

*   **FAQ & Acknowledgements:** Find answers to frequently asked questions and acknowledgements in the [Q\&A](docs/en/get_started/qa.md)
*   **Blogs:**
    *   Our vision: [slime: An SGLang-Native Post-Training Framework for RL Scaling](https://lmsys.org/blog/2025-07-09-slime/).
    *   Our ideas on agentic training: [Agent-Oriented Design: An Asynchronous and Decoupled Framework for Agentic RL](https://www.notion.so/Agent-Oriented-Design-An-Asynchronous-and-Decoupled-Framework-for-Agentic-RL-2278e692d081802cbdd5d37cef76a547).
    *   slime has served as the RL framework for GLM-4.5: [GLM-4.5: Reasoning, Coding, and Agentic Abilities](https://z.ai/blog/glm-4.5)
*   **Special Thanks:**  SGLang, Megatron‑LM, mbridge, OpenRLHF, veRL, Pai-Megatron-Patch and others.