<div align="center">
  <a href="https://art.openpipe.ai">
    <picture>
      <img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
    </picture>
  </a>

  <h1>Agent Reinforcement Trainer (ART): Unleash AI Agent Potential</h1>
  <p>
    <b>Train powerful, multi-step AI agents for real-world tasks using GRPO and open-source tools.</b>
  </p>

  [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-blue.svg)](https://github.com/openpipe/art/blob/main/CONTRIBUTING.md)
  [![Downloads](https://img.shields.io/pypi/dm/openpipe-art?color=364fc7&logoColor=364fc7)](https://pypi.org/project/openpipe-art/)
  [![Train Agent](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)
  [![Join Discord](https://img.shields.io/badge/Join%20Discord-5865F2?style=plastic&logo=discord&logoColor=white)](https://discord.gg/zbBHRUpwf4)
  [![Documentation](https://img.shields.io/badge/Documentation-orange?style=plastic&logo=gitbook&logoColor=white)](https://art.openpipe.ai)

</div>

## Key Features

*   **No Labeled Data Required:** ART leverages GRPO and server tool analysis, enabling agents to learn from experience without the need for pre-labeled datasets.
*   **General-Purpose Agent Training:** ART is designed to optimize models for any MCP (Model Context Protocol) server and various tasks.
*   **Performance-Driven:** Achieve or exceed state-of-the-art (SOTA) performance in multiple benchmarks.
*   **Easy Integration:** Seamlessly integrate ART without any server-side modifications.
*   **Open-Source:** ART is an open-source RL framework.

## Train Agents for Any Task

ART is an open-source reinforcement learning (RL) framework designed to improve agent reliability. It allows you to train agents to master complex tasks, even with zero or few-shot examples.

**Example Training Code Snippet:**

```python
from art.rewards import ruler_score_group

# Specialize a model for NWS MCP server
MCP_SERVER_URL = "https://server.smithery.ai/@smithery-ai/national-weather-service/mcp"

# Generate training scenarios based on MCP tools
scenarios = await generate_scenarios(
    num_scenarios=24,
    server_url=MCP_SERVER_URL,
)

# ...run the agent...

# Use RULER to assign relative scores to each trajectory
scored_groups = []
for group in groups:
    judged_group = await ruler_score_group(group)
    scored_groups.append(judged_group)

# Train the model to improve performance on the MCP server
await model.train(scored_groups)
```

## MCP•RL: Master MCP Servers

With **MCP•RL**, you can train agents to effectively use any Model Context Protocol (MCP) server with minimal setup. Simply provide a server URL, and MCP•RL will:

1.  **Automatically discover server tools.**
2.  **Design input tasks** that utilize those tools.
3.  **Train the model to improve performance** on the MCP server using RULER.
4.  **Test on new tasks** to validate the trained model.

<img src="assets/MCP_RL_diagram.svg" width="7000" alt="MCP RL Diagram">

## ART Notebooks: Train and Evaluate

Explore how ART is used in different scenarios.

| Agent Task         | Example Notebook                                                                                                             | Description                                     | Comparative Performance                                                                                                                                                                             |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **MCP•RL**         | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/mcp-rl/mcp-rl.ipynb)            | Qwen 2.5 3B masters the NWS MCP server          | [Link coming soon]                                                                                                                                                                                  |
| **ART•E [RULER]**  | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/art-e.ipynb)                 | Qwen 2.5 7B learns to search emails using RULER | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/email_agent/accuracy-training-progress.svg" height="72" alt="ART E email performance"> [benchmarks](/examples/art-e/art_e/evaluate/display_benchmarks.ipynb) |
| **2048**           | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)                   | Qwen 2.5 3B learns to play 2048                 | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/2048/accuracy-training-progress.svg" height="72" alt="2048 performance"> [benchmarks](/examples/2048/benchmark_2048.ipynb)                            |
| **Temporal Clue**  | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/temporal_clue/temporal-clue.ipynb) | Qwen 2.5 7B learns to solve Temporal Clue       | [Link coming soon]                                                                                                                                                                                  |
| **Tic Tac Toe**    | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb)     | Qwen 2.5 3B learns to play Tic Tac Toe          | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/tic-tac-toe-local/accuracy-training-progress.svg" height="72" alt="tic tac toe performance"> [benchmarks](/examples/tic_tac_toe/benchmark_tic_tac_toe.ipynb) |
| **Codenames**      | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/codenames/Codenames_RL.ipynb)      | Qwen 2.5 3B learns to play Codenames            | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/codenames/win_rate_over_time.png" height="72" alt="Codenames performance"> [benchmarks](/examples/codenames/Codenames_RL.ipynb)                            |
| **AutoRL [RULER]** | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/auto_rl.ipynb)                     | Train Qwen 2.5 7B to master any task            | [Link coming soon]                                                                                                                                                                                  |

## ART News and Updates

Stay up-to-date on the latest developments in ART:

*   🗞️ **[AutoRL: Zero-Data Training for Any Task](https://x.com/mattshumer_/status/1950572449025650733)** - Train custom AI models without labeled data using automatic input generation and RULER evaluation.
*   🗞️ **[RULER: Easy Mode for RL Rewards](https://openpipe.ai/blog/ruler-easy-mode-for-rl-rewards)** is now available for automatic reward generation in reinforcement learning.
*   🗞️ **[ART·E: How We Built an Email Research Agent That Beats o3](https://openpipe.ai/blog/art-e-mail-agent)** demonstrates a Qwen 2.5 14B email agent outperforming OpenAI's o3.
*   🗞️ **[ART Trainer: A New RL Trainer for Agents](https://openpipe.ai/blog/art-trainer)** enables easy training of LLM-based agents using GRPO.

[📖 See all blog posts →](https://openpipe.ai/blog)

## Why Choose ART?

*   **Simplified Integration:** ART provides convenient wrappers for introducing RL training into existing applications. It abstracts the training server into a modular service.
*   **Flexible Training:** Train agents locally, or leverage remote GPU environments for faster results.
*   **Observability:** Integrations with platforms like W&B, Langfuse, and OpenPipe simplify debugging and improve visibility into the training process.
*   **Customizable and Efficient:**  ART offers intelligent defaults and customizable training parameters to optimize for efficiency and stability.

## Installation

Get started with ART by installing the Python package:

```bash
pip install openpipe-art
```

## 🤖 ART•E Agent

Learn from the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) blog post how we trained Qwen 2.5 14B to outperform o3 at email retrieval!

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700" alt="ART E Performance Graph">

## Training Loop Overview

ART operates with a **client** and **server** architecture for flexibility. The client (OpenAI-compatible) interfaces with your codebase. The server manages model training and inference, abstracting the complexity of the RL loop.

**Here’s how it works:**

1.  **Inference:**
    *   Your code uses the ART client to perform agent actions.
    *   Completion requests are routed to the ART server (vLLM)
    *   All system, user, and assistant messages are saved in a Trajectory.
    *   When a rollout finishes, your code assigns a reward.

2.  **Training:**
    *   Trajectories are grouped and sent to the server.
    *   The server trains your model using GRPO.
    *   The server saves the trained LoRA, loads into vLLM
    *   Inference resumes.

## 🧩 Supported Models

ART supports most vLLM/HuggingFace-transformers compatible causal language models, including those supported by [Unsloth](https://docs.unsloth.ai/get-started/all-our-models). If you encounter issues with a specific model, please reach out on [Discord](https://discord.gg/zbBHRUpwf4) or create an issue on [GitHub](https://github.com/openpipe/art/issues).

## 🤝 Contribute to ART

ART is an evolving project, and contributions are welcome! See [CONTRIBUTING.md](https://github.com/openpipe/art/blob/main/CONTRIBUTING.md) for guidelines.

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

ART is available under the [Apache-2.0 License](LICENSE).

## 🙏 Credits

ART builds on the work of many in the open-source RL community. Special thanks to:

*   [Unsloth](https://github.com/unslothai/unsloth)
*   [vLLM](https://github.com/vllm-project/vllm)
*   [trl](https://github.com/huggingface/trl)
*   [torchtune](https://github.com/pytorch/torchtune)
*   [SkyPilot](https://github.com/skypilot-org/skypilot)

Thank you to our partners!

**[Visit the ART GitHub Repository](https://github.com/OpenPipe/ART) to get started.**