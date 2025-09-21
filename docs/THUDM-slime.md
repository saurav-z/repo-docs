# Slime: Revolutionizing LLM Post-Training with RL Scaling

**Slime** is a powerful LLM post-training framework designed to supercharge reinforcement learning (RL) scaling, offering high-performance training and flexible data generation capabilities. [Explore the original repository](https://github.com/THUDM/slime) for deeper insights.

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://thudm.github.io/slime/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/THUDM/slime)

## Key Features:

*   **High-Performance Training:** Efficiently trains LLMs using RL, integrating Megatron with SGLang for optimized performance.
*   **Flexible Data Generation:** Enables custom data generation workflows through adaptable interfaces and server-based engines.
*   **Agentic RL Support:** Designed with agentic training in mind, supporting asynchronous and decoupled RL frameworks.

## Core Capabilities:

*   **Efficient Training:** Streamlines the RL training process, optimizing the utilization of computational resources.
*   **Custom Data Workflows:** Facilitates the creation of tailor-made datasets to address specific training requirements.

## Key Components:

*   **Training (Megatron):** Manages the central training operations, interacting with data sources and parameter synchronization.
*   **Rollout (SGLang + Router):** Generates data and feedback, feeding new information into the training loop.
*   **Data Buffer:** Acts as a critical intermediary, handling prompt initialization, custom data integration, and rollout control.

## Quick Start:

Get up and running quickly with our detailed Quick Start Guide: [Quick Start Guide](./docs/en/get_started/quick_start.md)

Also, find examples for specific use cases: [examples](examples/)

## Arguments Walk Through:

*   **Megatron Arguments:** Utilize all Megatron arguments via `PYTHONPATH`. Example: `--tensor-model-parallel-size 2`.
*   **SGLang Arguments:** Use all arguments for the installed SGLang, prefixed with `--sglang-`. Example: `--sglang-mem-fraction-static`.
*   **Slime-Specific Arguments:** Explore slime-specific arguments in [slime/utils/arguments.py](slime/utils/arguments.py)

For detailed information, refer to the [Usage Documentation](docs/en/get_started/usage.md).

## Developer Guide:

*   **Contribution Guidelines:** Welcome contributions! Submit Issues or PRs for new features, performance improvements, or user experience feedback.
*   **Code Style:** Employ [pre-commit](https://pre-commit.com/) to maintain code style consistency:

    ```bash
    apt install pre-commit -y
    pre-commit install
    ```

*   **Debugging:** For troubleshooting, check the [Debugging Guide](docs/en/developer_guide/debug.md).

## FAQ & Acknowledgements:

*   **Frequently Asked Questions:** Consult the [Q\&A](docs/en/get_started/qa.md).
*   **Special Thanks:** Appreciations to SGLang, Megatron‑LM, mbridge, OpenRLHF, veRL, Pai-Megatron-Patch, and other projects and communities.