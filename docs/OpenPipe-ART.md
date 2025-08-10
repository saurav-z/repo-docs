<div align="center">
  <a href="https://art.openpipe.ai">
    <picture>
      <img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
    </picture>
  </a>
  <h1>Agent Reinforcement Trainer (ART): Train LLM Agents with Ease</h1>
  <p>Supercharge your language models with ART, an open-source framework for training powerful and reliable agents using Reinforcement Learning (RL).</p>

  [![PRs-Welcome][contribute-image]][contribute-url]
  [![PyPI version](https://img.shields.io/pypi/v/openpipe-art?color=364fc7)][pypi-url]
  [![Train Agent](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)
  [![Join Discord](https://img.shields.io/badge/Join%20Discord-5865F2?style=plastic&logo=discord&logoColor=white)](https://discord.gg/zbBHRUpwf4)
  [![Documentation](https://img.shields.io/badge/Documentation-orange?style=plastic&logo=gitbook&logoColor=white)](https://art.openpipe.ai)
</div>

## Key Features of ART:

*   **Train Agents for Complex Tasks:** Easily train multi-step agents for real-world tasks using GRPO (Gradient-based Reinforcement Policy Optimization).
*   **No Labeled Data Required (for MCP•RL):** Leverage ART's MCP•RL feature to train agents without the need for extensive labeled datasets. ART analyzes tool functionality to design effective training scenarios.
*   **General-Purpose & Adaptable:** ART is designed to optimize models for a wide range of applications and is compatible with any Model Context Protocol (MCP) server.
*   **High Performance:** Achieve state-of-the-art (SOTA) results in various benchmarks.
*   **Simplified Integration:** Integrate seamlessly with existing MCP servers without requiring server-side modifications.
*   **Modular RL Training Loop:** ART provides a clear separation of client and server for easy integration and scaling.
*   **Flexible Deployment:** Run the ART client locally and leverage cloud resources for training, offering flexibility in resource allocation.
*   **Built-in Observability:** Integrate with platforms like Weights & Biases (W&B), Langfuse, and OpenPipe for robust monitoring and debugging capabilities.
*   **Customizable & Optimized Defaults:** Configure training parameters or take advantage of the intelligent defaults to ensure efficiency and stability.

## Getting Started: MCP•RL - Mastering Model Context Protocol Servers

ART empowers you to train agents to effectively interact with any Model Context Protocol (MCP) server, all with minimal setup.

**Here's how MCP•RL works:**

1.  **Automatic Tool Discovery:** ART identifies and understands the tools available on your MCP server.
2.  **Dynamic Task Generation:** Based on the discovered tools, ART designs relevant input tasks.
3.  **RL-Driven Training:** The model undergoes iterative training on the MCP server using RULER (Robust, Unified, Lightweight Evaluation of Rewards).
4.  **Performance Validation:** The trained model's performance is validated on novel tasks to ensure effectiveness.

**Example Implementation:**

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

## ART Overview

ART simplifies agent development by providing an ergonomic harness for integrating GRPO into any Python application, allowing LLMs to **learn from experience** and improve agent reliability. Explore the provided notebooks for hands-on examples. Dive deeper into the documentation for advanced usage: [Documentation](https://art.openpipe.ai).

## 📒 Example Notebooks: Train Agents on Various Tasks

| Agent Task          | Example Notebook                                                                                                             | Description                                     | Comparative Performance                                                                                                                                                                             |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **MCP•RL**          | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/mcp-rl/mcp-rl.ipynb)               | Qwen 2.5 3B masters the NWS MCP server          | [Link coming soon]                                                                                                                                                                                  |
| **ART•E [RULER]**   | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/art-e.ipynb)                       | Qwen 2.5 7B learns to search emails using RULER | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/email_agent/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/art-e/art_e/evaluate/display_benchmarks.ipynb) |
| **2048**            | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)                   | Qwen 2.5 3B learns to play 2048                 | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/2048/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/2048/benchmark_2048.ipynb)                            |
| **Temporal Clue**   | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/temporal_clue/temporal-clue.ipynb) | Qwen 2.5 7B learns to solve Temporal Clue       | [Link coming soon]                                                                                                                                                                                  |
| **Tic Tac Toe**     | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb)     | Qwen 2.5 3B learns to play Tic Tac Toe          | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/tic-tac-toe-local/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/tic_tac_toe/benchmark_tic_tac_toe.ipynb) |
| **Codenames**       | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/codenames/Codenames_RL.ipynb)      | Qwen 2.5 3B learns to play Codenames            | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/codenames/win_rate_over_time.png" height="72"> [benchmarks](/examples/codenames/Codenames_RL.ipynb)                            |
| **AutoRL [RULER]**  | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/auto_rl.ipynb)                     | Train Qwen 2.5 7B to master any task            | [Link coming soon]                                                                                                                                                                                  |

## 📰 ART News and Research

Stay up-to-date with the latest advancements and explore our research on building cutting-edge AI agents.

*   🗞️ **[MCP•RL: Teach Your Model to Master Any MCP Server](https://x.com/corbtt/status/1953171838382817625)** - Automatically train models to effectively use MCP server tools through reinforcement learning.
*   🗞️ **[AutoRL: Zero-Data Training for Any Task](https://x.com/mattshumer_/status/1950572449025650733)** - Train custom AI models without labeled data using automatic input generation and RULER evaluation.
*   🗞️ **[RULER: Easy Mode for RL Rewards](https://openpipe.ai/blog/ruler-easy-mode-for-rl-rewards)** is now available for automatic reward generation in reinforcement learning.
*   🗞️ **[ART·E: How We Built an Email Research Agent That Beats o3](https://openpipe.ai/blog/art-e-mail-agent)** demonstrates a Qwen 2.5 14B email agent outperforming OpenAI's o3.
*   🗞️ **[ART Trainer: A New RL Trainer for Agents](https://openpipe.ai/blog/art-trainer)** enables easy training of LLM-based agents using GRPO.

[📖 See all blog posts →](https://openpipe.ai/blog)

## Why Choose ART?

*   **Simplified RL Integration:** ART provides wrappers to easily integrate RL training into your existing applications.
*   **Flexible Training Options:** Train agents from any environment. Run the ART client locally and utilize cloud resources for training.
*   **Enhanced Observability:** Integrations with W&B, Langfuse, and OpenPipe streamline debugging and monitoring.
*   **Intelligent Defaults & Customization:** Take advantage of optimized default settings or configure training parameters to meet your specific needs.

## Installation

Install ART using pip:

```bash
pip install openpipe-art
```

## 🤖 ART•E Agent: Real-World Application

Discover how ART can be applied to real-world problems, such as email retrieval. Check out the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) blog post to learn how we trained Qwen 2.5 14B to outperform o3 in email retrieval.

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## 🔁 Training Loop Explained

ART utilizes a client-server architecture to streamline the training process:

1.  **Inference:**
    *   Your code uses the ART client for agent workflows (often running multiple rollouts in parallel).
    *   Completion requests are routed to the ART server, which runs the model's latest LoRA in vLLM.
    *   Each message (system, user, assistant) is stored in a Trajectory.
    *   Rewards are assigned to each Trajectory upon rollout completion.

2.  **Training:**
    *   Trajectories are grouped and sent to the server.
    *   The server trains the model using GRPO.
    *   The server saves the trained LoRA and loads it into vLLM.
    *   The loop repeats until the specified number of iterations.

## 🧩 Supported Models

ART supports most vLLM/HuggingFace-transformers compatible causal language models, particularly those supported by [Unsloth](https://docs.unsloth.ai/get-started/all-our-models). Gemma 3 is currently unsupported. Report any issues on [Discord](https://discord.gg/zbBHRUpwf4) or [GitHub](https://github.com/openpipe/art/issues).

## 🤝 Contributing

We welcome contributions! Please review [CONTRIBUTING.md](CONTRIBUTING.md) for details.

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

ART builds upon the work of many in the open-source RL community. We are particularly grateful to:

*   [Unsloth](https://github.com/unslothai/unsloth)
*   [vLLM](https://github.com/vllm-project/vllm)
*   [trl](https://github.com/huggingface/trl)
*   [torchtune](https://github.com/pytorch/torchtune)
*   [SkyPilot](https://github.com/skypilot-org/skypilot)

Thank you to our partners for helping us test ART!

[pypi-url]: https://pypi.org/project/openpipe-art/
[contribute-url]: https://github.com/openpipe/art/blob/main/CONTRIBUTING.md
[contribute-image]: https://img.shields.io/badge/PRs-welcome-blue.svg