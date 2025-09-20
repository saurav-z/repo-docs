# SLIME: An RL Scaling Framework for LLMs

**SLIME empowers efficient LLM post-training through advanced RL techniques, enabling high-performance training and flexible data generation.**  Explore the power of SLIME, a cutting-edge framework for enhancing large language models with Reinforcement Learning (RL).

[View the original repository on GitHub](https://github.com/THUDM/slime)

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://thudm.github.io/slime/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/THUDM/slime)

## Key Features

*   **High-Performance Training:**  Leverages Megatron and SGLang for efficient RL training across various configurations.
*   **Flexible Data Generation:** Provides customizable data generation workflows with custom interfaces and server-based engines.
*   **Seamless Integration:** Compatible with SGLang and Megatron-LM for simplified workflows.
*   **Agentic RL Support:** Supporting the development of agentic capabilities, facilitating advanced RL models.

## Architecture Overview

[Image of Architecture Diagram (arch.png from original repo)]

**Core Modules:**

*   **training (Megatron):** Handles the primary training process, ingesting data from the Data Buffer and synchronizing parameters to the rollout module after each training step.
*   **rollout (SGLang + router):** Generates new training data, including rewards and verifier outputs, and stores it in the Data Buffer.
*   **data buffer:** Acts as a bridge, managing prompt initialization, custom data, and rollout generation methods.

## Quick Start

Get up and running quickly with SLIME!

*   Comprehensive Quick Start Guide:  [Quick Start Guide](./docs/en/get_started/quick_start.md) covering environment setup, data preparation, training startup, and key code analysis.
*   Explore practical examples: [examples](examples/) showcasing various use cases.

## Arguments Walk Through

SLIME utilizes three categories of arguments for configuration:

1.  **Megatron Arguments:** Configure Megatron parameters via `PYTHONPATH` (e.g., `--tensor-model-parallel-size 2`).
2.  **SGLang Arguments:** Utilize arguments for SGLang, prefixed with `--sglang-` (e.g., `--sglang-mem-fraction-static`).
3.  **SLIME-Specific Arguments:** Refer to [slime/utils/arguments.py](slime/utils/arguments.py) for a complete list.

For complete usage instructions, please refer to the [Usage Documentation](docs/en/get_started/usage.md).

## Developer Guide

*   **Contributions Welcome:**  We encourage suggestions, feature requests, and feedback through Issues or Pull Requests!
*   **Code Style Consistency:**  Use [pre-commit](https://pre-commit.com/) to ensure code style consistency for your commits:

    ```bash
    apt install pre-commit -y
    pre-commit install
    ```
*   **Debugging:** Refer to the [Debugging Guide](docs/en/developer_guide/debug.md) for troubleshooting tips.

## FAQ & Acknowledgements

*   **FAQ:** Find answers to frequently asked questions in the [Q\&A](docs/en/get_started/qa.md) section.
*   **Acknowledgements:** Special thanks to the following projects & communities: SGLang, Megatron‑LM, mbridge, OpenRLHF, veRL, Pai-Megatron-Patch and others.