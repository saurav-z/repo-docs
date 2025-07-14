<div align="center">

<a href="https://art.openpipe.ai"><picture>
<img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
</picture></a>

<p align="center">
  <h1>Agent Reinforcement Trainer (ART)</h1>
</p>

<p>
  Train advanced LLM agents efficiently with ART, leveraging GRPO for real-world tasks.
</p>

[![PRs-Welcome][contribute-image]][contribute-url]
[![Downloads][downloads-image]][pypi-url]
[![Train Agent](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)

[![Join Discord](https://img.shields.io/badge/Join%20Discord-5865F2?style=plastic&logo=discord&logoColor=white)](https://discord.gg/zbBHRUpwf4)
[![Documentation](https://img.shields.io/badge/Documentation-orange?style=plastic&logo=gitbook&logoColor=white)](https://art.openpipe.ai)

</div>

## About ART: Revolutionize LLM Agent Training

**Agent Reinforcement Trainer (ART)** is an open-source framework designed to simplify and accelerate the training of multi-step LLM agents for complex real-world tasks using GRPO (Generalized Policy Gradient Optimization).  ART provides an ergonomic harness for integrating GRPO into any python application, allowing you to improve agent reliability through experience. Check out the original repo [here](https://github.com/OpenPipe/ART).

## Key Features

*   **Effortless Reward Engineering:**  ART leverages the RULER system to eliminate the need for complex, hand-crafted reward functions.
*   **Accelerated Development:**  Reduce development time by 2-3x by bypassing manual reward function creation.
*   **Versatile & Adaptable:** ART is designed to work across any task without modification.
*   **Proven Performance:** ART consistently matches or surpasses the performance of agents trained with traditional, hand-crafted rewards.
*   **Seamless Integration:**  Easily integrate ART as a drop-in replacement for manual reward functions.
*   **Modular Architecture:** Includes a client-server setup to simplify training and deployment.
*   **Flexible Training:** Supports training on local GPUs or remote environments.
*   **Observability:** Integrates with platforms like W&B, Langfuse, and OpenPipe for simplified debugging and monitoring.
*   **Optimized Defaults:** Pre-configured with intelligent defaults, ensuring efficient and stable training.

## RULER: Zero-Shot Agent Rewards

**RULER** (Relative Universal LLM-Elicited Rewards) automatically scores agent trajectories using an LLM-as-judge, removing the need for hand-crafted reward functions. Simply define your task in the system prompt, and RULER handles the rest—no labeled data, expert feedback, or reward engineering required.

```python
# Before: Hours of reward engineering
def complex_reward_function(trajectory):
    # 50+ lines of careful scoring logic...
    pass

# After: One line with RULER
judged_group = await ruler_score_group(group, "openai/o3")
```

[📖 Learn more about RULER →](https://art.openpipe.ai/fundamentals/ruler)

## Quick Start: Example Notebooks

Jumpstart your agent training with these example notebooks:

| Agent Task        | Example Notebook                                                                                                             | Description                               | Comparative Performance                                                                                                                                     |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **2048**          | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)                   | Qwen 2.5 3B learns to play 2048           | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/2048/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/2048/benchmark_2048.ipynb)                            |
| **Temporal Clue** | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/temporal_clue/temporal-clue.ipynb) | Qwen 2.5 7B learns to solve Temporal Clue | [Link coming soon]                                                                                                                                          |
| **Tic Tac Toe**   | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb)     | Qwen 2.5 3B learns to play Tic Tac Toe    | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/tic-tac-toe-local/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/tic_tac_toe/benchmark_tic_tac_toe.ipynb) |
| **Codenames**     | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/codenames/Codenames_RL.ipynb)      | Qwen 2.5 3B learns to play Codenames      | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/codenames/win_rate_over_time.png" height="72"> [benchmarks](/examples/codenames/Codenames_RL.ipynb)                            |

## Why Choose ART?

*   **Simplified Integration:** Easily incorporate RL training into existing applications.
*   **Flexible Deployment:** Run the ART client locally and leverage a remote GPU-enabled server for training.
*   **Enhanced Observability:** Integrate with platforms like Weights & Biases, Langfuse, and OpenPipe for improved debugging.
*   **Optimized for Efficiency:** Benefit from intelligent defaults, optimized for training speed and reliability.

## Installation

Install the ART client with a single command:

```bash
pip install openpipe-art
```

## Example: ART•E Agent

Discover how ART was used to train the ART•E Agent, a Qwen 2.5 14B model, to beat o3 in email retrieval. Read more in the [ART•E Agent blog post](https://openpipe.ai/blog/art-e-mail-agent)!

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## Training Loop Overview

ART's architecture consists of a **client** and a **server** to manage the training workflow:

1.  **Inference:**
    1.  Your code uses the ART client to perform agentic workflows.
    2.  Completion requests are routed to the ART server.
    3.  Each message is stored in a Trajectory.
    4.  A reward is assigned to each Trajectory upon completion.

2.  **Training:**
    1.  Trajectories are grouped and sent to the server.
    2.  The server trains your model using GRPO, initializing from the latest checkpoint.
    3.  The trained model is saved and loaded into vLLM.
    4.  Inference resumes.

This loop continues for a specified number of iterations.

## Supported Models

ART supports most vLLM/HuggingFace-transformers compatible causal language models, particularly those supported by [Unsloth](https://docs.unsloth.ai/get-started/all-our-models).  Note that Gemma 3 is currently unsupported.  Report any issues on [Discord](https://discord.gg/zbBHRUpwf4) or [GitHub](https://github.com/openpipe/art/issues).

## Contribute

ART is actively developed, and contributions are welcome! See the [CONTRIBUTING.md](CONTRIBUTING.md) file for more details.

## Citation

```bibtex
@misc{hilton2025art,
  author = {Brad Hilton and Kyle Corbitt and David Corbitt and Saumya Gandhi and Angky William and Bohdan Kovalenskyi and Andie Jones},
  title = {ART: Agent Reinforcement Trainer},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/openpipe/art}}
}
```

## License

This repository is licensed under the [Apache-2.0 License](LICENSE).

## Credits

ART is built upon the work of many in the open-source RL community, and we're particularly grateful to the authors of:

*   [Unsloth](https://github.com/unslothai/unsloth)
*   [vLLM](https://github.com/vllm-project/vllm)
*   [trl](https://github.com/huggingface/trl)
*   [torchtune](https://github.com/pytorch/torchtune)
*   [SkyPilot](https://github.com/skypilot-org/skypilot)

Thanks to our partners for helping us test ART!

[pypi-url]: https://pypi.org/project/openpipe-art/
[contribute-url]: https://github.com/openpipe/art/blob/main/CONTRIBUTING.md
[contribute-image]: https://img.shields.io/badge/PRs-welcome-blue.svg
[downloads-image]: https://img.shields.io/pypi/dm/openpipe-art?color=364fc7&logoColor=364fc7