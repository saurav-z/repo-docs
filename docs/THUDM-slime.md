# Slime: Unleashing the Power of LLMs with RL Scaling

**Slime** is a cutting-edge LLM post-training framework designed for Reinforcement Learning (RL) scaling, enabling efficient model optimization and flexible data generation. [Explore the Slime Repository](https://github.com/THUDM/slime) for more details.

## Key Features:

*   **High-Performance Training:** Slime leverages Megatron and SGLang to deliver efficient training across various configurations.
*   **Flexible Data Generation:**  Create custom training datasets with ease using our versatile data generation interfaces and server-based engines.
*   **Modular Architecture:**  A clear and concise architecture, including training (Megatron), rollout (SGLang + router), and a data buffer for streamlined data management.
*   **Comprehensive Documentation:** Access detailed guides on Quick Start, Usage, Developer Guide, and FAQs to streamline your Slime experience.
*   **Integration with Leading Technologies:** Designed to work seamlessly with Megatron-LM, SGLang, and other leading projects.

## Table of Contents

-   [Architecture Overview](#architecture-overview)
-   [Quick Start](#quick-start)
-   [Checkpoint Format Conversion](#checkpoint-format-conversion)
-   [Starting the Training Process](#starting-the-training-process)
-   [Argument Descriptions](#argument-descriptions)
-   [Developer Guide](#developer-guide)
-   [FAQ & Acknowledgements](#faq--acknowledgements)

## Architecture Overview

![arch](./imgs/arch.png)

**Module Descriptions:**

*   **Training (Megatron):** Manages the main training process, pulling data from the Data Buffer and syncing parameters with the rollout module.
*   **Rollout (SGLang + Router):** Generates new data (including rewards and verifier outputs) and stores it in the Data Buffer.
*   **Data Buffer:** Serves as a crucial bridge, handling prompt initialization, custom data, and rollout generation methods.

## Quick Start

Get up and running quickly! The [Quick Start Guide](./docs/en/get_started/quick_start.md) covers environment setup, data preparation, training initiation, and key code analysis. Explore the [examples](examples/) directory for additional use cases.

## Arguments Walk Through

Slime's arguments are categorized for easy configuration:

1.  **Megatron Arguments:** Utilize existing Megatron arguments via `PYTHONPATH`, such as `--tensor-model-parallel-size 2`.
2.  **SGLang Arguments:** Use all SGLang arguments with the `--sglang-` prefix (e.g., `--sglang-mem-fraction-static`).
3.  **Slime-Specific Arguments:** Review [slime/utils/arguments.py](slime/utils/arguments.py) for Slime-specific arguments.

For comprehensive usage instructions, refer to the [Usage Documentation](docs/en/get_started/usage.md).

## Developer Guide

*   **Contribute:** We welcome contributions! Submit Issues or PRs with new features, performance enhancements, or feedback.
*   **Code Style:** Ensure code consistency using [pre-commit](https://pre-commit.com/):

    ```bash
    apt install pre-commit -y
    pre-commit install
    ```

*   **Debugging:** Consult the [Debugging Guide](docs/en/developer_guide/debug.md) for debugging tips.

## FAQ & Acknowledgements

*   **Q&A:** Find answers to frequently asked questions in the [Q&A](docs/en/get_started/qa.md).
*   **Acknowledgements:** Special thanks to SGLang, Megatron‑LM, mbridge, OpenRLHF, veRL, Pai-Megatron-Patch, and other valuable projects and communities.
*   **Citation:**

    ```bibtex
    @misc{slime_github,
      author       = {Zilin Zhu and Chengxing Xie and Xin Lv and slime Contributors},
      title        = {slime: An LLM post-training framework for RL Scaling},
      year         = {2025},
      howpublished = {\url{https://github.com/THUDM/slime}},
      note         = {GitHub repository. Corresponding author: Xin Lv},
      urldate      = {2025-06-19}
    }
    ```