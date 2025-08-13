<div align="center">

<a href="https://art.openpipe.ai"><picture>
<img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
</picture></a>

<p align="center">
  <h1>Agent Reinforcement Trainer (ART)</h1>
</p>

<p>
  Train powerful LLM agents for real-world tasks with ease using Agent Reinforcement Trainer (ART).
</p>

[![PRs-Welcome][contribute-image]][contribute-url]
[![PyPI version](https://img.shields.io/pypi/v/openpipe-art?color=364fc7)][pypi-url]
[![Train Agent](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)

[![Join Discord](https://img.shields.io/badge/Join%20Discord-5865F2?style=plastic&logo=discord&logoColor=white)](https://discord.gg/zbBHRUpwf4)
[![Documentation](https://img.shields.io/badge/Documentation-orange?style=plastic&logo=gitbook&logoColor=white)](https://art.openpipe.ai)

</div>

## Agent Reinforcement Trainer (ART): The Revolutionary Framework for LLM Agent Training

ART empowers you to train sophisticated, multi-step agents for complex real-world tasks using Gradient-based Reinforcement Policy Optimization (GRPO). This open-source framework simplifies the integration of Reinforcement Learning (RL) into your existing applications, enabling you to build more reliable and effective LLM-powered agents. **Get started today and unlock the power of self-improving agents!**  [Explore the ART repository](https://github.com/OpenPipe/ART).

**Key Features:**

*   **Easy Integration:** Seamlessly integrate RL training into your existing applications.
*   **No Labeled Data Required (MCP•RL):**  ART's MCP•RL feature learns tasks by analyzing tools on your server.
*   **General-Purpose & Flexible:** Optimizes models for any Model Context Protocol (MCP) server.
*   **State-of-the-Art Performance:** Achieves competitive or superior results in various benchmarks.
*   **Modular Architecture:**  ART divides functionality into a client and server, offering flexibility and control.
*   **Train from Anywhere:** Run ART on your local machine or leverage GPU-enabled environments.
*   **Customizable & Optimized Defaults:**  Configure training parameters or use optimized defaults for efficiency and stability.
*   **Observability and Debugging:** Integrates with platforms like W&B, Langfuse, and OpenPipe.

##  Key Use Cases & Examples

###  MCP•RL: Mastering Model Context Protocol Servers

Train agents to effectively utilize any MCP server with minimal setup.

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

### ART•E: Building a Smarter Email Agent

See how ART can be used to train an email agent that beats existing solutions.

## ART Overview

ART is an open-source RL framework designed to improve agent reliability by allowing LLMs to **learn from experience**. It provides a user-friendly interface for integrating GRPO into any Python application. Explore the documentation and start your journey with ART today!

## 📒 Example Notebooks: Hands-on Training

Get hands-on experience with ART through our comprehensive example notebooks:

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

Stay up-to-date with the latest developments in ART:

*   🗞️ **[MCP•RL: Teach Your Model to Master Any MCP Server](https://x.com/corbtt/status/1953171838382817625)** - Automatically train models to effectively use MCP server tools through reinforcement learning.
*   🗞️ **[AutoRL: Zero-Data Training for Any Task](https://x.com/mattshumer_/status/1950572449025650733)** - Train custom AI models without labeled data using automatic input generation and RULER evaluation.
*   🗞️ **[RULER: Easy Mode for RL Rewards](https://openpipe.ai/blog/ruler-easy-mode-for-rl-rewards)** is now available for automatic reward generation in reinforcement learning.
*   🗞️ **[ART·E: How We Built an Email Research Agent That Beats o3](https://openpipe.ai/blog/art-e-mail-agent)** demonstrates a Qwen 2.5 14B email agent outperforming OpenAI's o3.
*   🗞️ **[ART Trainer: A New RL Trainer for Agents](https://openpipe.ai/blog/art-trainer)** enables easy training of LLM-based agents using GRPO.

[📖 See all blog posts →](https://openpipe.ai/blog)

## Why Use ART?

*   **Simplified RL Integration:** ART provides wrappers to introduce RL into your existing applications.  It abstracts the complexities of the training server.
*   **Flexible Training Environments:**  Run the ART client locally and leverage the ART server to launch ephemeral GPU-enabled environments, or use your local GPU.
*   **Enhanced Observability:** Integrate with platforms like W&B, Langfuse, and OpenPipe for simplified debugging.
*   **Intelligent Defaults:** Benefit from optimized training parameters, or customize for your specific needs.

## Installation

Install ART and integrate it into your existing project.

```bash
pip install openpipe-art
```

## 🤖 ART•E Agent Example

Explore the real-world application of ART with the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) blog post, which details the training of a Qwen 2.5 14B model to outperform o3 at email retrieval.

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## 🔁 Training Loop Explained

ART's architecture is divided into a client and server for maximum flexibility and ease of use.

1.  **Inference:**
    *   Your code utilizes the ART client to perform an agentic workflow, often executing several rollouts in parallel.
    *   Completion requests are routed to the ART server, which runs the model's latest LoRA in vLLM.
    *   Each message is stored in a Trajectory.
    *   A `reward` is assigned to each Trajectory.
2.  **Training:**
    *   Trajectories are grouped and sent to the server.
    *   The server trains your model using GRPO, initializing from the latest checkpoint.
    *   The newly trained LoRA is saved and loaded into vLLM.
    *   Inference is unblocked and the loop resumes at step 1.

## 🧩 Supported Models

ART is designed to work with most vLLM/HuggingFace-transformers compatible causal language models supported by [Unsloth](https://docs.unsloth.ai/get-started/all-our-models).  If you encounter issues with a specific model, please reach out on [Discord](https://discord.gg/zbBHRUpwf4) or open an issue on [GitHub](https://github.com/openpipe/art/issues)!

## 🤝 Contributing

ART is an open-source project, and your contributions are greatly appreciated!  See the [CONTRIBUTING.md](CONTRIBUTING.md) file for more details.

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

This repository's source code is available under the [Apache-2.0 License](LICENSE).

## 🙏 Credits

ART is built upon the contributions of many. We are deeply grateful to the open source RL community, and especially to the following projects:

-   [Unsloth](https://github.com/unslothai/unsloth)
-   [vLLM](https://github.com/vllm-project/vllm)
-   [trl](https://github.com/huggingface/trl)
-   [torchtune](https://github.com/pytorch/torchtune)
-   [SkyPilot](https://github.com/skypilot-org/skypilot)

We extend our gratitude to our partners who have helped us test ART!

[pypi-url]: https://pypi.org/project/openpipe-art/
[contribute-url]: https://github.com/openpipe/art/blob/main/CONTRIBUTING.md
[contribute-image]: https://img.shields.io/badge/PRs-welcome-blue.svg