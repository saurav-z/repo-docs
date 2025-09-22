# Slime: Unleashing the Power of LLMs for Reinforcement Learning at Scale

**Slime is an innovative LLM post-training framework designed to supercharge Reinforcement Learning (RL) for large language models, enabling efficient training and flexible data generation.** ([View on GitHub](https://github.com/THUDM/slime))

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://thudm.github.io/slime/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/THUDM/slime)

**Key Features:**

*   **High-Performance Training:** Leverages Megatron and SGLang for efficient RL training across various modes.
*   **Flexible Data Generation:** Enables custom data generation workflows through user-defined interfaces and server-based engines.
*   **Modular Architecture:**  Features a clear separation of concerns with modules for training (Megatron), rollout (SGLang + router), and data buffering.
*   **Integration with SGLang:** Seamlessly integrates with SGLang for advanced language model functionalities.
*   **Open Source & Community Driven:** Contributions are welcome!

## Why Choose Slime?

Slime provides a robust and efficient framework for post-training LLMs using RL. It offers the flexibility to customize data generation and supports high-performance training, making it ideal for researchers and developers working with large language models.  Slime has been used as the RL framework for GLM-4.5: [GLM-4.5: Reasoning, Coding, and Agentic Abilities](https://z.ai/blog/glm-4.5).

## Core Components:

*   **Training (Megatron):** Executes the main training process, reading data from the Data Buffer and synchronizing parameters.
*   **Rollout (SGLang + router):** Generates new data, including rewards and verifier outputs, and stores it in the Data Buffer.
*   **Data Buffer:** Manages prompt initialization, custom data, and rollout generation methods, acting as a bridge between the training and rollout modules.

## Getting Started

*   **Quick Start:**  Explore the [Quick Start Guide](./docs/en/get_started/quick_start.md) for environment setup, data preparation, and training initiation.
*   **Examples:** Discover usage examples in the [examples](examples/) directory.
*   **Usage Documentation:**  Find complete usage instructions in the [Usage Documentation](docs/en/get_started/usage.md).

## Arguments & Configuration

Slime accepts arguments in three main categories:

1.  **Megatron arguments:** Pass arguments directly, such as `--tensor-model-parallel-size 2`.
2.  **SGLang arguments:**  Prefix SGLang arguments with `--sglang-`, e.g., `--sglang-mem-fraction-static`.
3.  **Slime-specific arguments:** Consult [slime/utils/arguments.py](slime/utils/arguments.py) for available options.

## Development & Contribution

*   **Contributions are welcome!** Submit issues or pull requests with new features, performance tuning suggestions, or user experience feedback.
*   **Code Style:** Use [pre-commit](https://pre-commit.com/) for code style consistency:

    ```bash
    apt install pre-commit -y
    pre-commit install
    ```
*   **Debugging:** Refer to the [Debugging Guide](docs/en/developer_guide/debug.md) for helpful tips.

## FAQ & Acknowledgements

*   **Q\&A:** Access frequently asked questions in the [Q\&A](docs/en/get_started/qa.md).
*   **Special Thanks:** SGLang, Megatron‑LM, mbridge, OpenRLHF, veRL, Pai-Megatron-Patch and others.
*   **Citation:**

    ```bibtext
    @misc{slime_github,
      author       = {Zilin Zhu and Chengxing Xie and Xin Lv and slime Contributors},
      title        = {slime: An LLM post-training framework for RL Scaling.},
      year         = {2025},
      howpublished = {\url{https://github.com/THUDM/slime}},
      note         = {GitHub repository. Corresponding author: Xin Lv},
      urldate      = {2025-06-19}
    }