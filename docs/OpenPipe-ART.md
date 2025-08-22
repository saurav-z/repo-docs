<div align="center">

<a href="https://art.openpipe.ai"><picture>
<img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
</picture></a>

<p align="center">
  <h1>Agent Reinforcement Trainer (ART)</h1>
</p>

<p>
  Supercharge your LLMs: Train multi-step agents for real-world tasks with ART using GRPO.
</p>

[![PRs-Welcome][contribute-image]][contribute-url]
[![PyPI version](https://img.shields.io/pypi/v/openpipe-art?color=364fc7)][pypi-url]
[![Train Agent](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)

[![Join Discord](https://img.shields.io/badge/Join%20Discord-5865F2?style=plastic&logo=discord&logoColor=white)](https://discord.gg/zbBHRUpwf4)
[![Documentation](https://img.shields.io/badge/Documentation-orange?style=plastic&logo=gitbook&logoColor=white)](https://art.openpipe.ai)

</div>

## Introduction to Agent Reinforcement Trainer (ART)

**ART is an open-source framework designed to enhance the reliability of your AI agents by enabling them to learn and improve through experience using GRPO.** Built for Python applications, ART provides an easy-to-use harness for integrating reinforcement learning, enabling you to train LLM-based agents for complex, real-world tasks.  Visit the [official ART repository on GitHub](https://github.com/OpenPipe/ART) to get started.

## Key Features of ART

*   **No Labeled Data Required:** ART learns by analyzing the tools and environment of a model, eliminating the need for pre-labeled training data.
*   **General-Purpose Applicability:** ART can optimize models for use with any Model Context Protocol (MCP) server, enhancing flexibility.
*   **Superior Performance:** Achieves or exceeds state-of-the-art (SOTA) results in two out of three benchmarks.
*   **Simplified Integration:** Requires no custom modifications to your existing MCP server setup.
*   **Modular Design:** Separates client and server functionalities for flexible deployment.
*   **Flexible Training:** Supports training from any client machine or on a local GPU.
*   **Easy Debugging:** Integrations with platforms such as W&B, Langfuse, and OpenPipe.
*   **Customization with Defaults:** Configure training parameters and inference engine configurations to meet your needs or use our optimized defaults.

## How to Get Started: MCP•RL Example

Train your agents using the cutting-edge **MCP•RL** method.  Here's a quick example to get you started:

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

## ART Notebook Examples: Hands-on Learning

Explore various ART applications through our interactive notebooks. Each notebook provides a hands-on approach to mastering different agent tasks.

| Agent Task         | Example Notebook                                                                                                             | Description                                     | Comparative Performance                                                                                                                                                                             |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **MCP•RL**         | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/mcp-rl/mcp-rl.ipynb)               | Qwen 2.5 3B masters the NWS MCP server          | [Link coming soon]                                                                                                                                                                                  |
| **ART•E [RULER]**  | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/art-e.ipynb)                       | Qwen 2.5 7B learns to search emails using RULER | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/email_agent/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/art-e/art_e/evaluate/display_benchmarks.ipynb) |
| **2048**           | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)                   | Qwen 2.5 3B learns to play 2048                 | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/2048/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/2048/benchmark_2048.ipynb)                            |
| **Temporal Clue**  | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/temporal_clue/temporal-clue.ipynb) | Qwen 2.5 7B learns to solve Temporal Clue       | [Link coming soon]                                                                                                                                                                                  |
| **Tic Tac Toe**    | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb)     | Qwen 2.5 3B learns to play Tic Tac Toe          | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/tic-tac-toe-local/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/tic_tac_toe/benchmark_tic_tac_toe.ipynb) |
| **Codenames**      | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/codenames/Codenames_RL.ipynb)      | Qwen 2.5 3B learns to play Codenames            | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/codenames/win_rate_over_time.png" height="72"> [benchmarks](/examples/codenames/Codenames_RL.ipynb)                            |
| **AutoRL [RULER]** | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/auto_rl.ipynb)                     | Train Qwen 2.5 7B to master any task            | [Link coming soon]                                                                                                                                                                                  |

## Stay Updated with ART News

Stay informed about the latest developments in ART and our research:

*   🗞️ **[MCP•RL: Teach Your Model to Master Any MCP Server](https://x.com/corbtt/status/1953171838382817625)** - Automatically train models to effectively use MCP server tools through reinforcement learning.
*   🗞️ **[AutoRL: Zero-Data Training for Any Task](https://x.com/mattshumer_/status/1950572449025650733)** - Train custom AI models without labeled data using automatic input generation and RULER evaluation.
*   🗞️ **[RULER: Easy Mode for RL Rewards](https://openpipe.ai/blog/ruler-easy-mode-for-rl-rewards)** is now available for automatic reward generation in reinforcement learning.
*   🗞️ **[ART·E: How We Built an Email Research Agent That Beats o3](https://openpipe.ai/blog/art-e-mail-agent)** demonstrates a Qwen 2.5 14B email agent outperforming OpenAI's o3.
*   🗞️ **[ART Trainer: A New RL Trainer for Agents](https://openpipe.ai/blog/art-trainer)** enables easy training of LLM-based agents using GRPO.

[📖 See all blog posts →](https://openpipe.ai/blog)

## Why Choose ART?

ART offers a streamlined solution for integrating RL into your existing applications. It abstracts the training server into a modular service, removing the need for direct code interaction.

## Installation

Easily integrate ART into your Python projects with a simple `pip` command:

```bash
pip install openpipe-art
```

## 🤖 ART•E Agent: A Real-World Example

Discover how ART can be applied to practical tasks, like the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent), which demonstrates the training of a Qwen 2.5 14B model to outperform o3 in email retrieval.

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## ART Training Loop Explained

ART's functionality is divided into a **client** and a **server**, which works as follows:

1.  **Inference:**
    *   Your code uses the ART client to perform an agentic workflow.
    *   Completion requests route to the ART server.
    *   Each `system`, `user`, and `assistant` message is stored in a Trajectory.
    *   Your code assigns a reward to its Trajectory when a rollout finishes.
2.  **Training:**
    *   Trajectories are grouped and sent to the server.
    *   The server trains your model using GRPO, initializing from the latest checkpoint.
    *   The server saves the new LoRA and loads it into vLLM.
    *   Inference is unblocked and the loop resumes.

This training loop runs until the set iterations are complete.

## 🧩 Supported Models

ART is designed to work with most vLLM/HuggingFace-transformers compatible causal language models, including those supported by [Unsloth](https://docs.unsloth.ai/get-started/all-our-models). If you experience any model compatibility issues, please reach out on [Discord](https://discord.gg/zbBHRUpwf4) or open an issue on [GitHub](https://github.com/openpipe/art/issues)!

## 🤝 Contributing

We welcome contributions to ART!  Please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file for details.

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

ART is built on the foundation of numerous open-source projects. Special thanks to the authors of:

*   [Unsloth](https://github.com/unslothai/unsloth)
*   [vLLM](https://github.com/vllm-project/vllm)
*   [trl](https://github.com/huggingface/trl)
*   [torchtune](https://github.com/pytorch/torchtune)
*   [SkyPilot](https://github.com/skypilot-org/skypilot)

We are also grateful to our partners for their support in testing ART.
```

[pypi-url]: https://pypi.org/project/openpipe-art/
[contribute-url]: https://github.com/openpipe/art/blob/main/CONTRIBUTING.md
[contribute-image]: https://img.shields.io/badge/PRs-welcome-blue.svg