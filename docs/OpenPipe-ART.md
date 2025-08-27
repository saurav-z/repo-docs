<div align="center">

<a href="https://art.openpipe.ai">
  <img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
</a>

<p align="center">
  <h1>Agent Reinforcement Trainer (ART)</h1>
</p>

</div>

**Supercharge your LLM agents with ART, an open-source framework for Reinforcement Learning (RL) that allows LLMs to learn from experience and become more reliable and efficient.** 

[![PRs-Welcome][contribute-image]][contribute-url]
[![PyPI version](https://img.shields.io/pypi/v/openpipe-art?color=364fc7)][pypi-url]
[![Train Agent](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/2048/2048.ipynb)
[![Join Discord](https://img.shields.io/badge/Join%20Discord-5865F2?style=plastic&logo=discord&logoColor=white)](https://discord.gg/zbBHRUpwf4)
[![Documentation](https://img.shields.io/badge/Documentation-orange?style=plastic&logo=gitbook&logoColor=white)](https://art.openpipe.ai)

## Key Features of ART

*   **Effortless Integration:** Seamlessly integrate RL into your existing Python applications.
*   **Automatic Behavior Improvement:** Train agents to improve multi-step reasoning and decision-making.
*   **Optimized Tool Usage:** Enhance agent efficiency by teaching them when and how to use tools effectively.
*   **RULER Compatibility:** Train agents without the need for hand-crafted reward functions, simplifying the RL process.
*   **Flexible Training:** Run ART on your local machine or leverage GPU-enabled environments.
*   **Observability & Debugging:** Integrate with platforms like W&B, Langfuse, and OpenPipe for easy monitoring.
*   **Intelligent Defaults:** Take advantage of optimized defaults for efficient and stable training.

## 🦜🔗 LangGraph Integration

ART integrates with LangGraph, enabling you to train sophisticated ReAct-style agents that improve through reinforcement learning, and build agents that can reason, use tools, and adapt their behavior over time without manual prompt engineering.

```python
import art
from art.langgraph import init_chat_model, wrap_rollout
from langgraph.prebuilt import create_react_agent

async def email_rollout(model: art.Model, scenario: str) -> art.Trajectory:
    # Create LangGraph agent with ART's chat model
    chat_model = init_chat_model(model.name)
    agent = create_react_agent(chat_model, tools)

    await agent.ainvoke({"messages": [("user", scenario)]})
    return art.Trajectory(reward=1.0, messages_and_choices=[])

# Train your agent
scenarios = ["Find urgent emails", "Search Q4 budget"]

# Using wrap_rollout (captures interactions automatically)
groups = await art.gather_trajectory_groups([
    art.TrajectoryGroup(wrap_rollout(model, email_rollout)(model, s) for _ in range(4))
    for s in scenarios
])

await model.train(groups)
```

[📖 Learn more about LangGraph integration →](https://art.openpipe.ai/integrations/langgraph-integration) | [🏋️ Try the notebook →](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/langgraph/art-e-langgraph.ipynb)

## ART Overview

ART is an open-source RL framework that improves agent reliability by allowing LLMs to **learn from experience**. ART provides an ergonomic harness for integrating GRPO into any python application. For a quick hands-on introduction, run one of the notebooks below. When you're ready to learn more, check out the [docs](https://art.openpipe.ai).

## 📒 Example Notebooks: Train Your Agents

Explore a variety of tasks with pre-built notebooks demonstrating ART's capabilities:

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

## 📰 ART News and Updates

Stay up-to-date on the latest advancements in ART:

*   🗞️ **[MCP•RL: Teach Your Model to Master Any MCP Server](https://x.com/corbtt/status/1953171838382817625)** - Automatically train models to effectively use MCP server tools through reinforcement learning.
*   🗞️ **[AutoRL: Zero-Data Training for Any Task](https://x.com/mattshumer_/status/1950572449025650733)** - Train custom AI models without labeled data using automatic input generation and RULER evaluation.
*   🗞️ **[RULER: Easy Mode for RL Rewards](https://openpipe.ai/blog/ruler-easy-mode-for-rl-rewards)** is now available for automatic reward generation in reinforcement learning.
*   🗞️ **[ART·E: How We Built an Email Research Agent That Beats o3](https://openpipe.ai/blog/art-e-mail-agent)** demonstrates a Qwen 2.5 14B email agent outperforming OpenAI's o3.
*   🗞️ **[ART Trainer: A New RL Trainer for Agents](https://openpipe.ai/blog/art-trainer)** enables easy training of LLM-based agents using GRPO.

[📖 See all blog posts →](https://openpipe.ai/blog)

## Why Choose ART?

*   **Modular Design:** Easily integrate RL training into your existing applications.
*   **Flexible Deployment:** Train agents on your local machine or in the cloud.
*   **Simplified Debugging:** Benefit from integrations with leading observability platforms.
*   **Optimized for Efficiency:** Benefit from training parameters and inference engine configurations that are optimized for both training efficiency and stability.

## Installation

Get started with ART in your project:

```bash
pip install openpipe-art
```

## 🤖 ART•E Agent: Example Use Case

Learn how ART can be used for real-world tasks, such as training an email retrieval agent in this [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) blog post.

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## 🔁 Training Loop Explained

ART's architecture is divided into a **client** and a **server**. The client interacts with your codebase, while the server handles the training process.

1.  **Inference:**

    *   Your code uses the ART client to perform an agentic workflow.
    *   Completion requests are routed to the ART server, which runs the model's latest LoRA in vLLM.
    *   Each `system`, `user`, and `assistant` message is stored in a Trajectory.
    *   When a rollout finishes, your code assigns a `reward` to its Trajectory.

2.  **Training:**

    *   Trajectories are grouped and sent to the server. Inference is blocked while training executes.
    *   The server trains your model using GRPO, initializing from the latest checkpoint.
    *   The server saves the trained LoRA and loads it into vLLM.
    *   Inference resumes.

This loop repeats until the training is complete.

## 🧩 Supported Models

ART is designed to work with most vLLM/HuggingFace-transformers compatible causal language models, including those supported by [Unsloth](https://docs.unsloth.ai/get-started/all-our-models). Gemma 3 is currently not supported. Please report any model compatibility issues on [Discord](https://discord.gg/zbBHRUpwf4) or [GitHub](https://github.com/openpipe/art/issues).

## 🤝 Contributing

ART is an open-source project, and contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

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

ART is built on the contributions of many. We are especially grateful to the authors of:

*   [Unsloth](https://github.com/unslothai/unsloth)
*   [vLLM](https://github.com/vllm-project/vllm)
*   [trl](https://github.com/huggingface/trl)
*   [torchtune](https://github.com/pytorch/torchtune)
*   [SkyPilot](https://github.com/skypilot-org/skypilot)

And a special thank you to our partners.

[pypi-url]: https://pypi.org/project/openpipe-art/
[contribute-url]: https://github.com/openpipe/art/blob/main/CONTRIBUTING.md
[contribute-image]: https://img.shields.io/badge/PRs-welcome-blue.svg