<div align="center">

<a href="https://art.openpipe.ai"><picture>
<img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
</picture></a>

<p align="center">
  <h1>Agent Reinforcement Trainer (ART): Train LLM Agents with Ease</h1>
</p>

<p>
  ART empowers you to train intelligent agents for complex tasks using Reinforcement Learning from AI feedback (RLAIF) and GRPO.
</p>

[![PRs-Welcome][contribute-image]][contribute-url]
[![PyPI version](https://img.shields.io/pypi/v/openpipe-art?color=364fc7)][pypi-url]
[![Train Agent](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/2048/2048.ipynb)

[![Join Discord](https://img.shields.io/badge/Join%20Discord-5865F2?style=plastic&logo=discord&logoColor=white)](https://discord.gg/zbBHRUpwf4)
[![Documentation](https://img.shields.io/badge/Documentation-orange?style=plastic&logo=gitbook&logoColor=white)](https://art.openpipe.ai)

</div>

## Introduction

**Agent Reinforcement Trainer (ART)** is an open-source framework designed to streamline the training of multi-step agents for real-world tasks by leveraging the power of Large Language Models (LLMs) and GRPO (Gradient-Based Policy Optimization) for efficient learning. Eliminate the need for complex reward engineering and accelerate your agent development.

[View the original repository on GitHub](https://github.com/OpenPipe/ART)

## Key Features

*   **Zero-Shot Reward Engineering with RULER:**
    *   **RULER** (Relative Universal LLM-Elicited Rewards) automatically scores agent trajectories using an LLM-as-judge, eliminating the need for hand-crafted reward functions.
    *   Define tasks in the system prompt, and RULER handles the rest.
    *   No labeled data, expert feedback, or manual reward engineering are required.
*   **Faster Development:** 2-3x faster development compared to traditional methods.
*   **General Purpose:** ART can be applied across any task without modifications.
*   **Strong Performance:** Achieves or surpasses the performance of hand-crafted rewards in multiple benchmarks.
*   **Easy Integration:** ART is a drop-in replacement for manual reward functions.
*   **Modular Architecture:** Separates the client and server components for flexible training and deployment.
*   **Built-in Integrations:** Seamlessly integrate with platforms like Weights & Biases (W&B), Langfuse, and OpenPipe for enhanced observability and debugging.
*   **Customizable:** Offers customizable training parameters and inference engine configurations.
*   **Supports a Wide Range of Models:** Compatible with most vLLM/HuggingFace-transformers compatible causal language models.

```python
# Before: Complex Reward Engineering
def complex_reward_function(trajectory):
    # 50+ lines of careful scoring logic...
    pass

# After: Using RULER for Simplified Rewards
judged_group = await ruler_score_group(group, "openai/o3")
```

[📖 Learn more about RULER →](https://art.openpipe.ai/fundamentals/ruler)

## Quickstart: Train an Agent

ART provides an ergonomic harness for integrating GRPO into any python application. Get started quickly by running a notebook example. When you're ready to learn more, check out the [docs](https://art.openpipe.ai).

## Example Notebooks

| Agent Task          | Example Notebook                                                                                                                       | Description                                         | Comparative Performance                                                                                                                                                                                                     |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ART•E LangGraph** | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/langgraph/art-e-langgraph.ipynb)   | Qwen 2.5 7B learns to search emails using LangGraph | [Link coming soon]                                                                                                                                                                                                          |
| **MCP•RL**          | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/mcp-rl/mcp-rl.ipynb)               | Qwen 2.5 3B masters the NWS MCP server              | [Link coming soon]                                                                                                                                                                                                          |
| **ART•E [RULER]**   | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/art-e.ipynb)                       | Qwen 2.5 7B learns to search emails using RULER     | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/email_agent/accuracy-training-progress.svg" height="72"> [benchmarks](/dev/art-e/art_e/evaluate/display_benchmarks.ipynb)                              |
| **2048**            | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/2048/2048.ipynb)                   | Qwen 2.5 3B learns to play 2048                     | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/2048/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/2048/display_benchmarks.ipynb)                                                |
| **Temporal Clue**   | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/temporal_clue/temporal-clue.ipynb) | Qwen 2.5 7B learns to solve Temporal Clue           | [Link coming soon]                                                                                                                                                                                                          |
| **Tic Tac Toe**     | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb)     | Qwen 2.5 3B learns to play Tic Tac Toe              | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/tic-tac-toe-local/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/tic_tac_toe/display-benchmarks.ipynb)                            |
| **Codenames**       | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/codenames/Codenames_RL.ipynb)      | Qwen 2.5 3B learns to play Codenames                | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/codenames/win_rate_over_time.png" height="72"> [benchmarks](https://github.com/OpenPipe/art-notebooks/blob/main/examples/codenames/Codenames_RL.ipynb) |
| **AutoRL [RULER]**  | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/auto_rl.ipynb)                     | Train Qwen 2.5 7B to master any task                | [Link coming soon]                                                                                                                                                                                                          |

## ART News & Updates

Stay informed about the latest advancements in agent training with ART:

*   🗞️ **[ART now integrates seamlessly with LangGraph](https://art.openpipe.ai/integrations/langgraph-integration)**
*   🗞️ **[MCP•RL: Teach Your Model to Master Any MCP Server](https://x.com/corbtt/status/1953171838382817625)**
*   🗞️ **[AutoRL: Zero-Data Training for Any Task](https://x.com/mattshumer_/status/1950572449025650733)**
*   🗞️ **[RULER: Easy Mode for RL Rewards](https://openpipe.ai/blog/ruler-easy-mode-for-rl-rewards)**
*   🗞️ **[ART·E: How We Built an Email Research Agent That Beats o3](https://openpipe.ai/blog/art-e-mail-agent)**
*   🗞️ **[ART Trainer: A New RL Trainer for Agents](https://openpipe.ai/blog/art-trainer)**

[📖 See all blog posts →](https://openpipe.ai/blog)

## Why Choose ART?

*   **Simplified RL Integration:** Easily incorporate RL training into your existing applications.
*   **Flexible Training:** Train agents from your local machine and leverage ephemeral, GPU-enabled environments or local GPUs.
*   **Enhanced Debugging:** Integrate with platforms like W&B, Langfuse, and OpenPipe for better observability and easier debugging.
*   **Intelligent Defaults:** Utilize optimized default configurations for efficient and stable training, or customize to fit your specific needs.

## Installation

Install ART and start training your agents:

```bash
pip install openpipe-art
```

## 🤖 ART•E Agent

Explore the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) blog post to see how we trained Qwen 2.5 14B to outperform o3 at email retrieval.

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## 🔁 Training Loop

ART's architecture consists of a **client** and a **server**:

1.  **Inference:**
    *   Your code uses the ART client to perform an agentic workflow.
    *   Completion requests are routed to the ART server, which runs the model's latest LoRA in vLLM.
    *   Each `system`, `user`, and `assistant` message is stored in a Trajectory.
    *   When a rollout finishes, your code assigns a `reward` to its Trajectory.
2.  **Training:**
    *   Trajectories are grouped and sent to the server.
    *   The server trains your model using GRPO, initializing from the latest checkpoint.
    *   The server saves the newly trained LoRA and loads it into vLLM.
    *   Inference is unblocked, and the loop resumes.

This loop runs until a specified number of iterations are complete.

## 🧩 Supported Models

ART is compatible with most vLLM/HuggingFace-transformers compatible causal language models, including those supported by [Unsloth](https://docs.unsloth.ai/get-started/all-our-models).

If you encounter issues with a specific model, please report them on [Discord](https://discord.gg/zbBHRUpwf4) or [GitHub](https://github.com/openpipe/art/issues).

## 🤝 Contributing

ART welcomes contributions! Refer to [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

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

ART is built upon the work of many contributors. Special thanks to the authors of:

*   [Unsloth](https://github.com/unslothai/unsloth)
*   [vLLM](https://github.com/vllm-project/vllm)
*   [trl](https://github.com/huggingface/trl)
*   [torchtune](https://github.com/pytorch/torchtune)
*   [SkyPilot](https://github.com/skypilot-org/skypilot)

Thank you to our partners for their support and testing!

[pypi-url]: https://pypi.org/project/openpipe-art/
[contribute-url]: https://github.com/openpipe/art/blob/main/CONTRIBUTING.md
[contribute-image]: https://img.shields.io/badge/PRs-welcome-blue.svg