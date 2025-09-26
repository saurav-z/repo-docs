# Slime: The LLM Post-Training Framework for RL Scaling

**Slime** is a powerful framework designed to scale Reinforcement Learning (RL) for Large Language Models (LLMs), enabling efficient training and flexible data generation. [Learn more on GitHub](https://github.com/THUDM/slime).

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://thudm.github.io/slime/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/THUDM/slime)

## Key Features

*   **High-Performance Training:** Train LLMs efficiently using Megatron and SGLang integration, supporting various training modes.
*   **Flexible Data Generation:** Create custom training data workflows with ease through a flexible data generation interface and server-based engines.

## Core Capabilities

Slime provides two main capabilities:

1.  **High-Performance Training**: It supports efficient training in various modes by connecting Megatron with SGLang.
2.  **Flexible Data Generation**: It enables arbitrary training data generation workflows through custom data generation interfaces and server-based engines.

## Architecture Overview

![arch](./imgs/arch.png)

Slime's architecture is designed for efficient RL scaling:

*   **training (Megatron):** Handles the main training process, reading data from the Data Buffer, and synchronizing parameters to the rollout module after training.
*   **rollout (SGLang + router):** Generates new data, including rewards and verifier outputs, and stores it in the Data Buffer.
*   **data buffer:** Serves as a bridge module managing prompt initialization, custom data, and rollout generation methods.

## Quick Start

Get up and running quickly with our comprehensive Quick Start guide, covering environment setup, data preparation, and training:

*   [Quick Start Guide](./docs/en/get_started/quick_start.md)
*   [Examples](examples/)

## Arguments Walk Through

Slime arguments are categorized into three types:

1.  **Megatron arguments:** Utilize standard Megatron arguments via `PYTHONPATH`. Example: `--tensor-model-parallel-size 2`.
2.  **SGLang arguments:** Configure SGLang settings by prefixing arguments with `--sglang-`. Example: `--sglang-mem-fraction-static`.
3.  **Slime-specific arguments:** See [slime/utils/arguments.py](slime/utils/arguments.py) for details.

For full usage instructions, see the [Usage Documentation](docs/en/get_started/usage.md).

## Developer Guide

We welcome contributions!

*   **Contributions are welcome!** Submit Issues or PRs for new features, performance improvements, and feedback.
*   **Code Style:** Use [pre-commit](https://pre-commit.com/) for consistent code style:
    ```bash
    apt install pre-commit -y
    pre-commit install
    ```
*   **Debugging:** See the [Debugging Guide](docs/en/developer_guide/debug.md) for assistance.

## FAQ & Acknowledgements

*   Find answers to frequently asked questions in the [Q\&A](docs/en/get_started/qa.md).
*   **Special Thanks:** SGLang, Megatron‑LM, mbridge, OpenRLHF, veRL, Pai-Megatron-Patch, and other projects and communities.
*   **Citation:** Please use the following BibTeX entry when citing Slime:

    ```bibtext
    @misc{slime_github,
      author       = {Zilin Zhu and Chengxing Xie and Xin Lv and slime Contributors},
      title        = {slime: An LLM post-training framework for RL Scaling},
      year         = {2025},
      howpublished = {\url{https://github.com/THUDM/slime}},
      note         = {GitHub repository. Corresponding author: Xin Lv},
      urldate      = {2025-06-19}
    }