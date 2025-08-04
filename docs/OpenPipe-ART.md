<div align="center">

<a href="https://art.openpipe.ai"><picture>
<img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
</picture></a>

<p align="center">
  <h1>Agent Reinforcement Trainer (ART)</h1>
</p>

<p>
Train advanced agents with large language models using GRPO for real-world tasks.
</p>

[![PRs-Welcome][contribute-image]][contribute-url]
[![Downloads][downloads-image]][pypi-url]
[![Train Agent](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)

[![Join Discord](https://img.shields.io/badge/Join%20Discord-5865F2?style=plastic&logo=discord&logoColor=white)](https://discord.gg/zbBHRUpwf4)
[![Documentation](https://img.shields.io/badge/Documentation-orange?style=plastic&logo=gitbook&logoColor=white)](https://art.openpipe.ai)

</div>

## Unlock the Power of AI Agents with ART

**Agent Reinforcement Trainer (ART)** is an open-source framework designed to empower you to train multi-step agents using the GRPO reinforcement learning method, enabling LLMs to learn from experience and excel at real-world tasks. Visit the [original repository](https://github.com/OpenPipe/ART) for the source code.

**Key Features of ART:**

*   **Zero-Shot Rewards with RULER:** Leverage Relative Universal LLM-Elicited Rewards (RULER) to automatically score agent trajectories, eliminating the need for manual reward functions.
*   **Faster Development:** Significantly accelerate your development process by skipping the time-consuming task of reward function engineering.
*   **Versatile Application:** Adaptable to any task without modification, providing a flexible solution for various agent-based projects.
*   **Superior Performance:** Achieve performance that matches or exceeds hand-crafted rewards in many benchmarks.
*   **Simplified Integration:** Easily integrate ART into your existing projects, acting as a drop-in replacement for manual reward functions.
*   **Modular Architecture:**  ART is divided into a client and server, allowing for flexible deployment and use.
*   **Training Flexibility:**  Train from anywhere, using either your local GPU or the ART server to kick off a GPU-enabled environment.

```python
# Before: Hours of reward engineering
def complex_reward_function(trajectory):
    # 50+ lines of careful scoring logic...
    pass

# After: One line with RULER
judged_group = await ruler_score_group(group, "openai/o3")
```

[📖 Learn more about RULER →](https://art.openpipe.ai/fundamentals/ruler)

## ART Overview

ART streamlines the process of integrating GRPO into your Python applications, improving agent reliability by allowing LLMs to learn from experience.

## 📒 Notebook Examples

Quickly get started with these hands-on examples:

| Agent Task         | Example Notebook                                                                                                             | Description                                     | Comparative Performance                                                                                                                                                                             |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ART•E [RULER]**  | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/art-e/art-e.ipynb)                 | Qwen 2.5 7B learns to search emails using RULER | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/email_agent/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/art-e/art_e/evaluate/display_benchmarks.ipynb) |
| **2048**           | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)                   | Qwen 2.5 3B learns to play 2048                 | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/2048/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/2048/benchmark_2048.ipynb)                            |
| **Temporal Clue**  | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/temporal_clue/temporal-clue.ipynb) | Qwen 2.5 7B learns to solve Temporal Clue       | [Link coming soon]                                                                                                                                                                                  |
| **Tic Tac Toe**    | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb)     | Qwen 2.5 3B learns to play Tic Tac Toe          | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/tic-tac-toe-local/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/tic_tac_toe/benchmark_tic_tac_toe.ipynb) |
| **Codenames**      | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/codenames/Codenames_RL.ipynb)      | Qwen 2.5 3B learns to play Codenames            | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/codenames/win_rate_over_time.png" height="72"> [benchmarks](/examples/codenames/Codenames_RL.ipynb)                            |
| **AutoRL [RULER]** | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/auto_rl.ipynb)                     | Train Qwen 2.5 7B to master any task            | [Link coming soon]                                                                                                                                                                                  |

## 📰 ART News & Updates

Stay up-to-date with the latest research and advancements in agent training:

*   🗞️ **[AutoRL: Zero-Data Training for Any Task](https://x.com/mattshumer_/status/1950572449025650733)** - Train custom AI models without labeled data using automatic input generation and RULER evaluation.
*   🗞️ **[RULER: Easy Mode for RL Rewards](https://openpipe.ai/blog/ruler-easy-mode-for-rl-rewards)** is now available for automatic reward generation in reinforcement learning.
*   🗞️ **[ART·E: How We Built an Email Research Agent That Beats o3](https://openpipe.ai/blog/art-e-mail-agent)** demonstrates a Qwen 2.5 14B email agent outperforming OpenAI's o3.
*   🗞️ **[ART Trainer: A New RL Trainer for Agents](https://openpipe.ai/blog/art-trainer)** enables easy training of LLM-based agents using GRPO.

[📖 See all blog posts →](https://openpipe.ai/blog)

## Why Choose ART?

*   **Seamless Integration:** Easily incorporate RL training into your existing applications with convenient wrappers.
*   **Flexible Deployment:** Train agents locally or remotely, leveraging cloud resources for optimal performance.
*   **Enhanced Observability:** Integrate with popular platforms like W&B, Langfuse, and OpenPipe for simplified debugging and monitoring.
*   **Intelligent Defaults:** Benefit from optimized default settings for efficient and stable training.

## Installation

Get started with ART by running the following command:

```
pip install openpipe-art
```

## 🤖 ART•E Agent

Explore a real-world application of ART by checking out the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) blog post. Learn how Qwen 2.5 14B was trained to outperform o3 in email retrieval!

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## 🔁 Training Loop Explained

ART functions with a clear client-server architecture:

1.  **Inference Phase:**

    *   Your code uses the ART client for agentic workflows.
    *   Completions are handled by the ART server, using the latest model weights in vLLM.
    *   Agent interactions are recorded as Trajectories.
    *   Upon completion, a reward is assigned to each Trajectory.
2.  **Training Phase:**

    *   Trajectories are grouped and sent to the server for training.
    *   The server trains your model with GRPO, starting from the latest checkpoint.
    *   Trained weights are saved and loaded into vLLM.
    *   Inference resumes, and the cycle repeats.

This loop runs until a predefined number of iterations are completed.

## 🧩 Supported Models

ART supports most vLLM/HuggingFace-transformers compatible causal language models, including those supported by [Unsloth](https://docs.unsloth.ai/get-started/all-our-models), with the exception of Gemma 3.

## 🤝 Contributing

We welcome contributions! Please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidance.

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

## 🙏 Acknowledgements

ART is built upon the contributions of the open-source RL community, especially the creators of:

*   [Unsloth](https://github.com/unslothai/unsloth)
*   [vLLM](https://github.com/vllm-project/vllm)
*   [trl](https://github.com/huggingface/trl)
*   [torchtune](https://github.com/pytorch/torchtune)
*   [SkyPilot](https://github.com/skypilot-org/skypilot)

Thank you to our partners for their invaluable testing and support.

[pypi-url]: https://pypi.org/project/openpipe-art/
[contribute-url]: https://github.com/openpipe/art/blob/main/CONTRIBUTING.md
[contribute-image]: https://img.shields.io/badge/PRs-welcome-blue.svg
[downloads-image]: https://img.shields.io/pypi/dm/openpipe-art?color=364fc7&logoColor=364fc7