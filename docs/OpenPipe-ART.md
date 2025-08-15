<div align="center">
  <a href="https://art.openpipe.ai">
    <picture>
      <img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
    </picture>
  </a>
  <h1>Agent Reinforcement Trainer (ART): Train LLM Agents for Real-World Tasks</h1>
</div>

**ART empowers you to train powerful multi-step agents using Reinforcement Learning from Experience (RLE).** Learn how to fine-tune and build AI agents that excel at complex tasks. [Explore the ART Repository on GitHub](https://github.com/OpenPipe/ART).

## Key Features

*   🎯 **No Labeled Data Required:** ART leverages your model's environment and tools to generate scenarios and learn what tasks a server will be used for.
*   ⚙️ **General-Purpose & Customizable:** Optimized for any Model Context Protocol (MCP) server and offers intelligent defaults for easy customization.
*   🚀 **SOTA Performance:** Achieve leading-edge performance, matching or exceeding state-of-the-art results in various benchmarks.
*   🔌 **Effortless Integration:** Seamlessly integrates with your existing MCP server without modifications.

## Quick Start: Train Agents with MCP•RL

ART's MCP•RL lets you train agents to use any MCP server:

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

## Core Concepts

ART is an open-source Reinforcement Learning framework for building reliable agents. It helps LLMs learn from experience by allowing you to easily integrate GRPO into your applications. Dive into practical examples with our notebooks!

## 📒 Example Notebooks

Train agents on various tasks, including:

| Agent Task         | Example Notebook                                                                                                             | Description                                     | Comparative Performance                                                                                                                                                                             |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **MCP•RL**         | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/mcp-rl/mcp-rl.ipynb)               | Qwen 2.5 3B masters the NWS MCP server          | [Link coming soon]                                                                                                                                                                                  |
| **ART•E [RULER]**  | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/art-e.ipynb)                       | Qwen 2.5 7B learns to search emails using RULER | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/email_agent/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/art-e/art_e/evaluate/display_benchmarks.ipynb) |
| **2048**           | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)                   | Qwen 2.5 3B learns to play 2048                 | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/2048/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/2048/benchmark_2048.ipynb)                            |
| **Temporal Clue**  | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/temporal_clue/temporal-clue.ipynb) | Qwen 2.5 7B learns to solve Temporal Clue       | [Link coming soon]                                                                                                                                                                                  |
| **Tic Tac Toe**    | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb)     | Qwen 2.5 3B learns to play Tic Tac Toe          | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/tic-tac-toe-local/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/tic_tac_toe/benchmark_tic_tac_toe.ipynb) |
| **Codenames**      | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/codenames/Codenames_RL.ipynb)      | Qwen 2.5 3B learns to play Codenames            | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/codenames/win_rate_over_time.png" height="72"> [benchmarks](/examples/codenames/Codenames_RL.ipynb)                            |
| **AutoRL [RULER]** | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/auto_rl.ipynb)                     | Train Qwen 2.5 7B to master any task            | [Link coming soon]                                                                                                                                                                                  |

## 📰 ART News & Updates

Stay up-to-date with the latest advancements and research from ART:

*   🗞️ **[MCP•RL: Teach Your Model to Master Any MCP Server](https://x.com/corbtt/status/1953171838382817625)** - Automatically train models to effectively use MCP server tools through reinforcement learning.
*   🗞️ **[AutoRL: Zero-Data Training for Any Task](https://x.com/mattshumer_/status/1950572449025650733)** - Train custom AI models without labeled data using automatic input generation and RULER evaluation.
*   🗞️ **[RULER: Easy Mode for RL Rewards](https://openpipe.ai/blog/ruler-easy-mode-for-rl-rewards)** is now available for automatic reward generation in reinforcement learning.
*   🗞️ **[ART·E: How We Built an Email Research Agent That Beats o3](https://openpipe.ai/blog/art-e-mail-agent)** demonstrates a Qwen 2.5 14B email agent outperforming OpenAI's o3.
*   🗞️ **[ART Trainer: A New RL Trainer for Agents](https://openpipe.ai/blog/art-trainer)** enables easy training of LLM-based agents using GRPO.

[📖 See all blog posts →](https://openpipe.ai/blog)

## Why Choose ART?

*   **Simplified RL Integration:** ART provides user-friendly wrappers to integrate RL training into your projects.
*   **Flexible Training:** Run the ART client on your local machine and offload the training to a GPU-enabled server.
*   **Enhanced Observability:** Integrations with platforms like W&B, Langfuse, and OpenPipe simplify debugging and improve model performance.
*   **Optimized Defaults:** Benefit from pre-configured training parameters and inference settings, while also retaining the flexibility to customize them.

## Installation

Easily integrate ART into your Python projects using pip:

```bash
pip install openpipe-art
```

## 🤖 ART•E Agent

Learn about a real-world application of ART by exploring the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) blog post. See how we trained Qwen 2.5 14B to outperform o3 in email retrieval!

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## 🔁 Training Loop Explained

ART employs a client-server architecture, consisting of an OpenAI-compatible client and a dedicated server. The training process involves the following steps:

1.  **Inference:**
    *   Your code uses the ART client to execute an agentic workflow, including parallel rollouts.
    *   Completion requests are routed to the ART server, running the model's latest LoRA with vLLM.
    *   Each message (`system`, `user`, `assistant`) is stored in a Trajectory.
    *   Rollout completion triggers a reward assignment.

2.  **Training:**
    *   Once the rollouts are complete, Trajectories are grouped and sent to the server. Inference is paused during training.
    *   The server trains your model using GRPO, potentially starting from a checkpoint.
    *   The trained LoRA is saved and loaded into vLLM.
    *   Inference is resumed, and the loop repeats.

This loop continues until a predefined number of iterations is reached.

## 🧩 Supported Models

ART is compatible with most causal language models that are supported by vLLM/HuggingFace-transformers, particularly those that are supported by [Unsloth](https://docs.unsloth.ai/get-started/all-our-models). We do not support Gemma 3 at the moment. Let us know if you have any other model compatibility issues on [Discord](https://discord.gg/zbBHRUpwf4) or report an issue on [GitHub](https://github.com/openpipe/art/issues)!

## 🤝 Contributing

ART is under active development. We warmly welcome contributions from the community! For guidelines, please see the [CONTRIBUTING.md](CONTRIBUTING.md) file.

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

The source code for this repository is released under the [Apache-2.0 License](LICENSE).

## 🙏 Credits

ART's development draws on the collective knowledge of the open-source RL community. We're especially grateful to the projects below:

*   [Unsloth](https://github.com/unslothai/unsloth)
*   [vLLM](https://github.com/vllm-project/vllm)
*   [trl](https://github.com/huggingface/trl)
*   [torchtune](https://github.com/pytorch/torchtune)
*   [SkyPilot](https://github.com/skypilot-org/skypilot)

Thank you to our partners for helping us test ART in the wild!