<div align="center">

<a href="https://art.openpipe.ai"><picture>
<img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
</picture></a>

<p align="center">
  <h1>Agent Reinforcement Trainer (ART)</h1>
</p>

<p>
  Train multi-step agents for real-world tasks using GRPO.
</p>

[![PRs-Welcome][contribute-image]][contribute-url]
[![Downloads][downloads-image]][pypi-url]
[![Train Agent](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)

[![Join Discord](https://img.shields.io/badge/Join%20Discord-5865F2?style=plastic&logo=discord&logoColor=white)](https://discord.gg/zbBHRUpwf4)
[![Documentation](https://img.shields.io/badge/Documentation-orange?style=plastic&logo=gitbook&logoColor=white)](https://art.openpipe.ai)

</div>

## Agent Reinforcement Trainer (ART): Supercharge Your LLM Agents with Experience

**ART is an open-source reinforcement learning (RL) framework designed to enhance the performance and reliability of your LLM agents by enabling them to learn from experience.** This enables you to train LLM agents to solve complex, multi-step tasks with improved accuracy and efficiency.  [Explore the ART repository](https://github.com/OpenPipe/ART) to get started.

**Key Features:**

*   **Accelerated Development:** ART simplifies reward function design, often reducing development time by 2-3x.
*   **Universal Applicability:** ART works across various tasks without requiring modification.
*   **Superior Performance:** ART can match or outperform hand-crafted rewards in many benchmarks.
*   **Simplified Integration:** ART seamlessly integrates into your existing projects, making it a drop-in replacement for manual reward functions.
*   **Flexible Training:** Train agents locally or leverage the cloud for GPU-enabled environments.
*   **Observability & Debugging:** Integrations with platforms like W&B, Langfuse, and OpenPipe enhance visibility and simplify debugging.
*   **Customizable with Intelligent Defaults:** ART offers optimized default settings while enabling users to adjust training parameters and inference engine configurations.

## 📏 RULER: Zero-Shot Agent Rewards

**RULER (Relative Universal LLM-Elicited Rewards)** automatically scores agent trajectories using an LLM-as-judge, eliminating the need for hand-crafted reward functions.

[📖 Learn more about RULER →](https://art.openpipe.ai/fundamentals/ruler)

## 📒 Example Notebooks: Train LLM Agents on Diverse Tasks

Get hands-on experience with ART by exploring these example notebooks:

| Agent Task        | Example Notebook                                                                                                             | Description                               | Comparative Performance                                                                                                                                     |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ART•E [RULER]**         | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/art-e/art-e.ipynb)               | Qwen 2.5 7B learns to search emails using RULER     | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/email_agent/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/art-e/art_e/evaluate/display_benchmarks.ipynb)                                                                                                                                          |
| **2048**          | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)                   | Qwen 2.5 3B learns to play 2048           | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/2048/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/2048/benchmark_2048.ipynb)                            |
| **Temporal Clue** | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/temporal_clue/temporal-clue.ipynb) | Qwen 2.5 7B learns to solve Temporal Clue | [Link coming soon]                                                                                                                                          |
| **Tic Tac Toe**   | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb)     | Qwen 2.5 3B learns to play Tic Tac Toe    | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/tic-tac-toe-local/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/tic_tac_toe/benchmark_tic_tac_toe.ipynb) |
| **Codenames**     | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/codenames/Codenames_RL.ipynb)      | Qwen 2.5 3B learns to play Codenames      | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/codenames/win_rate_over_time.png" height="72"> [benchmarks](/examples/codenames/Codenames_RL.ipynb)                            |

## Why Use ART?

ART empowers you to:

*   **Integrate RL into Existing Applications:** ART provides convenient wrappers and a modular service to streamline the integration of RL training.
*   **Train from Anywhere:** Run ART clients locally or utilize the ART server to kick off GPU-enabled environments.
*   **Improve Observability and Debugging:** Leverage integrations with platforms like W&B, Langfuse, and OpenPipe.
*   **Customize Your Training:** ART is customizable with intelligent defaults to meet specific needs.

## Installation

Easily install ART to begin training LLM agents:

```bash
pip install openpipe-art
```

## 🤖 ART•E Agent

Learn how to train Qwen 2.5 14B to outperform o3 at email retrieval with the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) blog post!

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## 🔁 Training Loop Overview

ART's training loop is divided into client and server components, designed for efficient and iterative agent improvement:

1.  **Inference:**
    1.  Your code interacts with the ART client to perform agentic workflows, often running multiple rollouts in parallel to gather data.
    2.  Completion requests are routed to the ART server, which executes the model's latest LoRA in vLLM.
    3.  Each `system`, `user`, and `assistant` message is stored in a Trajectory.
    4.  When a rollout finishes, your code assigns a `reward` to its Trajectory.
2.  **Training:**
    1.  Once rollouts are complete, Trajectories are grouped and sent to the server. Inference is blocked during training.
    2.  The server trains your model using GRPO, starting from the latest checkpoint (or an empty LoRA initially).
    3.  The server saves the updated LoRA and loads it into vLLM.
    4.  Inference resumes at step 1.

The loop continues until a defined number of iterations are complete.

## 🧩 Supported Models

ART supports a wide range of vLLM/HuggingFace-transformers compatible causal language models, especially those compatible with [Unsloth](https://docs.unsloth.ai/get-started/all-our-models), with the current exception of Gemma 3.

## 🤝 Contributing

ART is actively being developed, and your contributions are welcome!  See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 📖 Citation

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

## ⚖️ License

This project is licensed under the [Apache-2.0 License](LICENSE).

## 🙏 Credits

ART is built upon the work of many contributors in the open-source RL community. We are especially grateful to the authors of:

*   [Unsloth](https://github.com/unslothai/unsloth)
*   [vLLM](https://github.com/vllm-project/vllm)
*   [trl](https://github.com/huggingface/trl)
*   [torchtune](https://github.com/pytorch/torchtune)
*   [SkyPilot](https://github.com/skypilot-org/skypilot)

Thank you to our partners who've helped us test ART!

[pypi-url]: https://pypi.org/project/openpipe-art/
[contribute-url]: https://github.com/openpipe/art/blob/main/CONTRIBUTING.md
[contribute-image]: https://img.shields.io/badge/PRs-welcome-blue.svg
[downloads-image]: https://img.shields.io/pypi/dm/openpipe-art?color=364fc7&logoColor=364fc7