<div align="center">
    <a href="https://art.openpipe.ai"><picture>
        <img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
    </picture></a>
    <p align="center">
        <h1>Agent Reinforcement Trainer (ART)</h1>
    </p>
</div>

## Train LLM Agents with Reinforcement Learning (RL)

**ART (Agent Reinforcement Trainer)** is an open-source framework designed to empower you to train multi-step agents, enabling LLMs to learn from experience and excel at real-world tasks.

*   **[Explore the ART Repository](https://github.com/OpenPipe/ART)**
*   **[View Documentation](https://art.openpipe.ai)**

### Key Features

*   **Zero-Shot Reward Engineering:** Leverage RULER (Relative Universal LLM-Elicited Rewards) to eliminate the need for manual reward functions, accelerating development by 2-3x.
*   **General-Purpose:** ART works across various tasks without modification, making it versatile for diverse applications.
*   **High Performance:** Achieve results that match or exceed those of hand-crafted reward systems in many benchmarks.
*   **Easy Integration:** Seamlessly integrate ART into your existing projects as a drop-in replacement for manual reward functions.
*   **Open-Source and Customizable:** Benefit from an open-source framework with intelligent defaults and the flexibility to configure training parameters.
*   **Train from Anywhere:** Run the ART client on your laptop and let the ART server kick off an ephemeral GPU-enabled environment, or run on a local GPU.
*   **Integrations:** W&B, Langfuse, and OpenPipe provide flexible observability and **simplify debugging**.

## Zero-Shot Reward with RULER

**RULER** (Relative Universal LLM-Elicited Rewards) revolutionizes agent training by using an LLM as a judge to automatically score agent trajectories. This eliminates the need for hand-crafted reward functions, saving time and effort.

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

ART is an open-source RL framework that improves agent reliability by allowing LLMs to **learn from experience**. ART provides an ergonomic harness for integrating GRPO into any python application.

## Quickstart with Notebooks

Get started with ART using these example notebooks, demonstrating agent training across various tasks:

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

## 📰 ART News & Updates

Stay updated with the latest developments in ART and agent training:

*   🗞️ **[ART now integrates seamlessly with LangGraph](https://art.openpipe.ai/integrations/langgraph-integration)** - Train your LangGraph agents with reinforcement learning for smarter multi-step reasoning and improved tool usage.
*   🗞️ **[MCP•RL: Teach Your Model to Master Any MCP Server](https://x.com/corbtt/status/1953171838382817625)** - Automatically train models to effectively use MCP server tools through reinforcement learning.
*   🗞️ **[AutoRL: Zero-Data Training for Any Task](https://x.com/mattshumer_/status/1950572449025650733)** - Train custom AI models without labeled data using automatic input generation and RULER evaluation.
*   🗞️ **[RULER: Easy Mode for RL Rewards](https://openpipe.ai/blog/ruler-easy-mode-for-rl-rewards)** is now available for automatic reward generation in reinforcement learning.
*   🗞️ **[ART·E: How We Built an Email Research Agent That Beats o3](https://openpipe.ai/blog/art-e-mail-agent)** demonstrates a Qwen 2.5 14B email agent outperforming OpenAI's o3.
*   🗞️ **[ART Trainer: A New RL Trainer for Agents](https://openpipe.ai/blog/art-trainer)** enables easy training of LLM-based agents using GRPO.

[📖 See all blog posts →](https://openpipe.ai/blog)

## Why Choose ART?

*   **Simplified RL Integration:** ART provides convenient wrappers for introducing RL training into existing applications, abstracting the training server into a modular service.
*   **Flexible Training:** Train from anywhere, whether on your local machine or using GPU-enabled environments.
*   **Customization and Observability:** Benefit from integrations with platforms like W&B, Langfuse, and OpenPipe, offering flexible observability and simplified debugging.

## Installation

Install ART with a simple pip command:

```bash
pip install openpipe-art
```

## 🤖 ART•E Agent: Real-World Example

Explore the practical application of ART by examining the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent), which demonstrates how to train Qwen 2.5 14B to outperform o3 in email retrieval.

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## 🔁 Training Loop Explained

ART's architecture divides functionality between a **client** and a **server**.

**1. Inference**
    1.  Your code uses the ART client to perform an agentic workflow.
    2.  Completion requests are routed to the ART server.
    3.  Each system, user, and assistant message is stored in a Trajectory.
    4.  When a rollout finishes, your code assigns a reward to its Trajectory.

**2. Training**
    1.  Trajectories are grouped and sent to the server.
    2.  The server trains your model using GRPO, saving the LoRA.
    3.  Inference is unblocked and the loop resumes at step 1.

## 🧩 Supported Models

ART supports most vLLM/HuggingFace-transformers compatible causal language models. Please let us know if you encounter any issues on [Discord](https://discord.gg/zbBHRUpwf4) or [GitHub](https://github.com/openpipe/art/issues).

## 🤝 Contributing

We welcome contributions! See our [CONTRIBUTING.md](https://github.com/openpipe/art/blob/main/CONTRIBUTING.md) file for details.

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

This project is available under the [Apache-2.0 License](LICENSE).

## 🙏 Credits

ART is built upon the work of several open-source projects, including:

*   [Unsloth](https://github.com/unslothai/unsloth)
*   [vLLM](https://github.com/vllm-project/vllm)
*   [trl](https://github.com/huggingface/trl)
*   [torchtune](https://github.com/pytorch/torchtune)
*   [SkyPilot](https://github.com/skypilot-org/skypilot)

Thank you to our partners for their valuable contributions.