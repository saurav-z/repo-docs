<div align="center">

<a href="https://art.openpipe.ai"><picture>
<img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
</picture></a>

<p align="center">
  <h1>Agent Reinforcement Trainer (ART)</h1>
</p>
</div>

**Train powerful multi-step agents for real-world tasks with Agent Reinforcement Trainer (ART), the open-source RL framework that makes training LLM-based agents easy and efficient.**

[View the original repository on GitHub](https://github.com/OpenPipe/ART)

[![PRs-Welcome](https://img.shields.io/badge/PRs-welcome-blue.svg)](https://github.com/openpipe/art/blob/main/CONTRIBUTING.md)
[![Downloads](https://img.shields.io/pypi/dm/openpipe-art?color=364fc7&logoColor=364fc7)](https://pypi.org/project/openpipe-art/)
[![Train Agent](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)
[![Join Discord](https://img.shields.io/badge/Join%20Discord-5865F2?style=plastic&logo=discord&logoColor=white)](https://discord.gg/zbBHRUpwf4)
[![Documentation](https://img.shields.io/badge/Documentation-orange?style=plastic&logo=gitbook&logoColor=white)](https://art.openpipe.ai)

## Key Features

*   **Effortless Reward Engineering:** Leverage **RULER**, a zero-shot reward system that uses LLMs as judges to automatically score agent trajectories, eliminating the need for hand-crafted reward functions.
*   **Accelerated Development:** Experience 2-3x faster development by skipping the time-consuming process of reward function engineering.
*   **General-Purpose Applicability:** ART's architecture allows it to work across any task without requiring modifications.
*   **High Performance:** Achieve results that match or exceed hand-crafted rewards in many benchmarks.
*   **Seamless Integration:** Easily integrate ART into your existing applications with a drop-in replacement for manual reward functions.
*   **Modular Architecture**: ART divides functionality into a client and server for flexible training from anywhere.
*   **Extensible Observability**: Easily integrate platforms like W&B, Langfuse, and OpenPipe for debugging.
*   **Customizable Defaults**: Fine-tune training parameters and inference engine configurations, or utilize pre-optimized defaults.

## RULER: Zero-Shot Agent Rewards

RULER (Relative Universal LLM-Elicited Rewards) revolutionizes agent training by eliminating the need for hand-crafted reward functions. Simply define your task in the system prompt, and RULER handles the rest—**no labeled data, expert feedback, or reward engineering required**.

[📖 Learn more about RULER →](https://art.openpipe.ai/fundamentals/ruler)

```python
# Before: Hours of reward engineering
def complex_reward_function(trajectory):
    # 50+ lines of careful scoring logic...
    pass

# After: One line with RULER
judged_group = await ruler_score_group(group, "openai/o3")
```

## ART Overview

ART is an open-source RL framework that enhances agent reliability by allowing LLMs to learn from experience. ART provides an ergonomic harness for integrating GRPO into any Python application. Check out the [docs](https://art.openpipe.ai) for a comprehensive guide.

## Getting Started with Example Notebooks

Explore the capabilities of ART with these example notebooks, showcasing agent training on various tasks.

| Agent Task         | Example Notebook                                                                                                             | Description                                     | Comparative Performance                                                                                                                                                                             |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ART•E [RULER]**  | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/art-e.ipynb)                       | Qwen 2.5 7B learns to search emails using RULER | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/email_agent/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/art-e/art_e/evaluate/display_benchmarks.ipynb) |
| **2048**           | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)                   | Qwen 2.5 3B learns to play 2048                 | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/2048/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/2048/benchmark_2048.ipynb)                            |
| **Temporal Clue**  | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/temporal_clue/temporal-clue.ipynb) | Qwen 2.5 7B learns to solve Temporal Clue       | [Link coming soon]                                                                                                                                                                                  |
| **Tic Tac Toe**    | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb)     | Qwen 2.5 3B learns to play Tic Tac Toe          | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/tic-tac-toe-local/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/tic_tac_toe/benchmark_tic_tac_toe.ipynb) |
| **Codenames**      | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/codenames/Codenames_RL.ipynb)      | Qwen 2.5 3B learns to play Codenames            | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/codenames/win_rate_over_time.png" height="72"> [benchmarks](/examples/codenames/Codenames_RL.ipynb)                            |
| **AutoRL [RULER]** | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/auto_rl.ipynb)                     | Train Qwen 2.5 7B to master any task            | [Link coming soon]                                                                                                                                                                                  |

## Latest News and Updates

Stay informed with the latest developments in ART and AI agent research.

-   🗞️ **[AutoRL: Zero-Data Training for Any Task](https://x.com/mattshumer_/status/1950572449025650733)** - Train custom AI models without labeled data using automatic input generation and RULER evaluation.
-   🗞️ **[RULER: Easy Mode for RL Rewards](https://openpipe.ai/blog/ruler-easy-mode-for-rl-rewards)** is now available for automatic reward generation in reinforcement learning.
-   🗞️ **[ART·E: How We Built an Email Research Agent That Beats o3](https://openpipe.ai/blog/art-e-mail-agent)** demonstrates a Qwen 2.5 14B email agent outperforming OpenAI's o3.
-   🗞️ **[ART Trainer: A New RL Trainer for Agents](https://openpipe.ai/blog/art-trainer)** enables easy training of LLM-based agents using GRPO.

[📖 See all blog posts →](https://openpipe.ai/blog)

## Why Choose ART?

*   **Easy Integration:** ART provides convenient wrappers for integrating RL training into existing applications.
*   **Flexible Training:** Train from any location; run the ART client on your laptop, on a local GPU, or let the ART server kick off an ephemeral GPU-enabled environment.
*   **Observability & Debugging**: Integrations with hosted platforms like W&B, Langfuse, and OpenPipe.
*   **Customizable & Efficient**: Optimized defaults, and configurable training parameters and inference engine configurations.

## Installation

Get started quickly by installing the ART library:

```bash
pip install openpipe-art
```

## 🤖 ART•E Agent Example

Explore the ART•E Agent use case, showing how to train Qwen 2.5 14B to excel in email retrieval.

[Read the ART•E Agent blog post](https://openpipe.ai/blog/art-e-mail-agent)

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## 🔁 Training Loop Explained

ART employs a client-server architecture for efficient agent training:

1.  **Inference:** Your code uses the ART client to perform an agentic workflow. Completion requests are routed to the ART server. Each `system`, `user`, and `assistant` message is stored in a Trajectory.
2.  **Training:** Trajectories are grouped and sent to the server. The server trains your model using GRPO, saves the LoRA, and loads it into vLLM.

## 🧩 Supported Models

ART supports most vLLM/HuggingFace-transformers compatible causal language models. If you experience any issues, please contact us on [Discord](https://discord.gg/zbBHRUpwf4) or open an issue on [GitHub](https://github.com/openpipe/art/issues)!

## 🤝 Contributing

ART is an open-source project, and contributions are highly encouraged! See the [CONTRIBUTING.md](CONTRIBUTING.md) file for details.

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

ART leverages the contributions of several open-source projects. We are especially grateful to the authors of:

*   [Unsloth](https://github.com/unslothai/unsloth)
*   [vLLM](https://github.com/vllm-project/vllm)
*   [trl](https://github.com/huggingface/trl)
*   [torchtune](https://github.com/pytorch/torchtune)
*   [SkyPilot](https://github.com/skypilot-org/skypilot)

We thank our partners for their invaluable support in testing ART.