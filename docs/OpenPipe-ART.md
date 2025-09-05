<div align="center">

<a href="https://art.openpipe.ai"><picture>
<img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
</picture></a>

<p align="center">
  <h1>Agent Reinforcement Trainer (ART)</h1>
</p>

<p>
Train multi-step agents for real-world tasks with GRPO, and revolutionize how you build AI agents.
</p>

[![PRs-Welcome][contribute-image]][contribute-url]
[![PyPI version](https://img.shields.io/pypi/v/openpipe-art?color=364fc7)][pypi-url]
[![Train Agent](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/2048/2048.ipynb)

[![Join Discord](https://img.shields.io/badge/Join%20Discord-5865F2?style=plastic&logo=discord&logoColor=white)](https://discord.gg/zbBHRUpwf4)
[![Documentation](https://img.shields.io/badge/Documentation-orange?style=plastic&logo=gitbook&logoColor=white)](https://art.openpipe.ai)

</div>

## 🚀 Agent Reinforcement Trainer (ART): Train Smarter Agents Faster

ART is an open-source Reinforcement Learning (RL) framework designed to empower you to train intelligent, multi-step agents capable of handling complex real-world tasks.  Leveraging GRPO, ART offers an ergonomic harness for integrating RL into your Python applications, enabling LLMs to learn from experience and improve agent reliability.  Learn how to build and train LLM agents with [the ART project on GitHub](https://github.com/OpenPipe/ART).

**Key Features:**

*   **Zero-Shot Reward Engineering with RULER:** Utilize LLM-as-judge to automatically score agent trajectories, eliminating the need for manual reward functions.  Reduce development time by 2-3x.
*   **Versatile & Adaptable:** ART can be used across any task without modification.
*   **High Performance:** Achieve performance that matches or even surpasses hand-crafted reward functions.
*   **Seamless Integration:** Drop-in replacement for manual reward functions, simplifying your workflow.
*   **Open Source & Community-Driven:** Benefit from a collaborative ecosystem and actively contribute to the development of advanced agent training.
*   **Modular & Customizable:**  Easily customize training parameters and inference engine configurations to meet your specific needs.

## 🌟 RULER: Zero-Shot Agent Rewards

**RULER (Relative Universal LLM-Elicited Rewards)** revolutionizes RL by using an LLM to automatically evaluate agent behavior, removing the need for hand-crafted reward functions. Define your task, and RULER handles the scoring, saving you time and effort.

**Key Benefits of RULER:**

*   **Faster Development:** Drastically reduce development time by skipping manual reward function engineering.
*   **General Purpose:** Works across a wide variety of tasks without modification.
*   **Superior Performance:** Achieve or exceed the performance of hand-crafted rewards in many benchmarks.
*   **Easy to Integrate:** Simple to integrate into your existing projects, offering a streamlined solution.

```python
# Before: Reward engineering
def complex_reward_function(trajectory):
    # 50+ lines of careful scoring logic...
    pass

# After: One line with RULER
judged_group = await ruler_score_group(group, "openai/o3")
```

[📖 Learn more about RULER →](https://art.openpipe.ai/fundamentals/ruler)

## 💻 Quickstart & Examples

Get hands-on quickly by exploring our interactive notebooks.  Train agents on diverse tasks and see ART in action.

**Example Notebooks:**

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

## 📰 Stay Updated: ART News

Keep up-to-date with the latest developments in ART and explore our recent research.

*   **ART Now Integrates with LangGraph:** Train your LangGraph agents using reinforcement learning for smarter multi-step reasoning and improved tool usage.
*   **MCP•RL: Teaching Models to Master MCP Servers:** Learn how to automatically train models to use MCP server tools with reinforcement learning.
*   **AutoRL: Zero-Data Training for Any Task:**  Train custom AI models without labeled data using automatic input generation and RULER evaluation.
*   **RULER: Easy Mode for RL Rewards** is now available for automatic reward generation in reinforcement learning.
*   **ART·E: How We Built an Email Research Agent That Beats o3:** Learn how our Qwen 2.5 14B email agent outperformed OpenAI's o3.
*   **ART Trainer: A New RL Trainer for Agents** enables easy training of LLM-based agents using GRPO.

[📖 See all blog posts →](https://openpipe.ai/blog)

## ⚙️ How ART Works: Training Loop Overview

ART employs a client-server architecture to streamline the RL training process.

**1. Inference**

*   Your code uses the ART client to perform agentic workflows.
*   Completion requests are routed to the ART server, which runs the latest LoRA in vLLM.
*   Each `system`, `user`, and `assistant` message is stored in a Trajectory.
*   When a rollout finishes, your code assigns a `reward` to its Trajectory.

**2. Training**

*   Trajectories are grouped and sent to the server.
*   The server trains your model using GRPO.
*   The server saves the newly trained LoRA and loads it into vLLM.
*   The loop resumes with inference.

This loop continues until the training is complete.

## 💻 Installation

Easily integrate ART into your project with a simple pip install:

```bash
pip install openpipe-art
```

## 🤖 ART•E Agent

Explore how ART can be applied to real-world tasks by examining the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) blog post, where we detail the training of Qwen 2.5 14B to beat o3 at email retrieval.

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## 🧩 Supported Models

ART is designed to be compatible with most vLLM/HuggingFace-transformers compatible causal language models.  If you encounter any issues with model compatibility, please let us know via [Discord](https://discord.gg/zbBHRUpwf4) or by opening an issue on [GitHub](https://github.com/openpipe/art/issues)!

## 🤝 Contributing

We welcome contributions!  Check out our [CONTRIBUTING.md](CONTRIBUTING.md) file for details on how to get involved.

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

ART is built upon the work of many contributors.  We are particularly grateful to the authors of the following projects:

*   [Unsloth](https://github.com/unslothai/unsloth)
*   [vLLM](https://github.com/vllm-project/vllm)
*   [trl](https://github.com/huggingface/trl)
*   [torchtune](https://github.com/pytorch/torchtune)
*   [SkyPilot](https://github.com/skypilot-org/skypilot)

Thank you to our partners for helping us test ART!
[pypi-url]: https://pypi.org/project/openpipe-art/
[contribute-url]: https://github.com/openpipe/art/blob/main/CONTRIBUTING.md
[contribute-image]: https://img.shields.io/badge/PRs-welcome-blue.svg