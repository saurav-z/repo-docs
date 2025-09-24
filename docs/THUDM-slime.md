# Slime: An LLM Post-Training Framework for RL Scaling

**Slime empowers efficient LLM post-training and RL scaling, enabling you to optimize your models for real-world applications.** ([View the Original Repo](https://github.com/THUDM/slime))

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://thudm.github.io/slime/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/THUDM/slime)

Slime is a powerful framework designed for post-training Large Language Models (LLMs) and scaling Reinforcement Learning (RL) applications. It provides a streamlined approach for enhancing LLM performance through efficient training and flexible data generation.

## Key Features

*   **High-Performance Training:** Train your LLMs efficiently using a combination of Megatron and SGLang, supporting various training modes.
*   **Flexible Data Generation:** Create custom training datasets with ease using specialized interfaces and server-based engines.
*   **Modular Architecture:** The system is built with three core modules: training (Megatron), rollout (SGLang + router), and a data buffer for smooth data flow.

## Architecture Overview

![arch](./imgs/arch.png)

**Module Descriptions:**

*   **Training (Megatron):** Manages the primary training process, reading data from the Data Buffer and synchronizing parameters with the rollout module after training.
*   **Rollout (SGLang + Router):** Generates new data (including rewards and verifier outputs) and stores it in the Data Buffer.
*   **Data Buffer:** A crucial module that manages prompt initialization, custom data, and rollout generation methods, acting as a bridge between training and rollout.

## Quick Start

Get up and running quickly with our comprehensive quick start guide. This includes environment setup, data preparation, training initiation, and in-depth code analysis.

*   [Quick Start Guide](./docs/en/get_started/quick_start.md)
*   Explore various use cases in our [examples](examples/).

## Argument Descriptions

Slime supports arguments in three categories:

1.  **Megatron arguments:** Inherited from Megatron via `PYTHONPATH`. Configure these arguments such as `--tensor-model-parallel-size 2`.
2.  **SGLang arguments:** Supported arguments for the installed SGLang, prefixed with `--sglang-`.  Example: `--sglang-mem-fraction-static`.
3.  **Slime-specific arguments:**  Defined in [slime/utils/arguments.py](slime/utils/arguments.py).

For complete usage instructions, please consult our [Usage Documentation](docs/en/get_started/usage.md).

## Developer Guide

We welcome contributions from the community!

*   **Contribute:** Submit Issues or Pull Requests for new features, performance improvements, or user experience feedback.
*   **Code Style:**  Ensure code style consistency by using [pre-commit](https://pre-commit.com/):

    ```bash
    apt install pre-commit -y
    pre-commit install
    ```
*   **Debugging:** Use the [Debugging Guide](docs/en/developer_guide/debug.md) for assistance.

## FAQ & Acknowledgements

*   Find answers to frequently asked questions in our [Q\&A](docs/en/get_started/qa.md).
*   **Special Thanks:**  We extend our gratitude to the following projects and communities: SGLang, Megatron‑LM, mbridge, OpenRLHF, veRL, Pai-Megatron-Patch and others.
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
    ```