# Slime: Supercharge Your LLM Post-Training with RL Scaling

**Slime** is a cutting-edge LLM post-training framework designed to optimize and scale Reinforcement Learning (RL) for large language models, enabling efficient training and flexible data generation. Explore the original repository on [GitHub](https://github.com/THUDM/slime).

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://thudm.github.io/slime/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/THUDM/slime)

## Key Features of Slime:

*   **High-Performance Training:** Leverages Megatron and SGLang for efficient RL training across various modes.
*   **Flexible Data Generation:** Enables custom data generation workflows using configurable interfaces and server-based engines.
*   **Efficient Architecture:**  Integrates a streamlined architecture with optimized modules.
*   **Comprehensive Documentation:** Includes quick start guides, usage instructions, and developer resources.
*   **Open Source & Community Driven:** Welcomes contributions and provides debugging tips.

## Core Capabilities

Slime provides two main functionalities:

1.  **Efficient Training**: Supports efficient training in various modes by connecting Megatron with SGLang.
2.  **Flexible Data Generation**: Enables arbitrary training data generation workflows through custom data generation interfaces and server-based engines.

## Architecture Overview

![arch](./imgs/arch.png)

**Key Modules:**

*   **training (Megatron):**  Handles the primary training process, data input, and parameter synchronization.
*   **rollout (SGLang + router):**  Generates new data, including rewards and verifier outputs, and stores it in the Data Buffer.
*   **data buffer:**  Manages data flow, prompt initialization, and rollout generation methods.

## Getting Started

*   **Quick Start:** Get up and running quickly with the [Quick Start Guide](./docs/en/get_started/quick_start.md) which covers environment setup, data preparation, training startup, and key code analysis.
*   **Examples:** Explore various use cases in the [examples](examples/) directory.
*   **Usage Documentation:** For complete usage instructions, please refer to the [Usage Documentation](docs/en/get_started/usage.md).

## Advanced Topics

*   **Arguments:** Understand the argument structure including Megatron, SGLang and slime-specific arguments.
*   **Checkpoint Format Conversion:** Explore how to handle checkpoint format conversions.
*   **Starting the Training Process:** Detailed instructions on how to start the training process.
*   **Developer Guide:** Learn how to contribute, use pre-commit for code style, and find debugging tips.
*   **FAQ & Acknowledgements:** Find frequently asked questions and see the project's acknowledgements.

## Contributing

We welcome contributions!  If you have suggestions for new features, performance tuning, or feedback on user experience, feel free to submit an Issue or PR 😊.

*   Use [pre-commit](https://pre-commit.com/) to ensure code style consistency for your commits:

    ```bash
    apt install pre-commit -y
    pre-commit install
    ```

*   For debugging tips, please refer to the [Debugging Guide](docs/en/developer_guide/debug.md)

## Frequently Asked Questions and Acknowledgements

For frequently asked questions, please see the [Q\&A](docs/en/get_started/qa.md).

Special thanks to the following projects & communities: SGLang, Megatron‑LM, mbridge, OpenRLHF, veRL, Pai-Megatron-Patch and others.

To cite Slime, please use the following BibTeX entry:

```bibtext
@misc{slime_github,
  author       = {Zilin Zhu and Chengxing Xie and Xin Lv and slime Contributors},
  title        = {slime: An LLM post-training framework for RL Scaling},
  year         = {2025},
  howpublished = {\url{https://github.com/THUDM/slime}},
  note         = {GitHub repository. Corresponding author: Xin Lv},
  urldate      = {2025-06-19}
}