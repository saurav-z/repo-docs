<div align="center">
  <a href="https://art.openpipe.ai">
    <picture>
      <img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
    </picture>
  </a>

  <h1>Agent Reinforcement Trainer (ART)</h1>

  <p><b>Supercharge your LLMs: Train multi-step agents for real-world tasks with ART, leveraging GRPO for optimal performance.</b></p>

  [![PRs-Welcome][contribute-image]][contribute-url]
  [![PyPI version](https://img.shields.io/pypi/v/openpipe-art?color=364fc7)][pypi-url]
  [![Train Agent](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)
  [![Join Discord](https://img.shields.io/badge/Join%20Discord-5865F2?style=plastic&logo=discord&logoColor=white)](https://discord.gg/zbBHRUpwf4)
  [![Documentation](https://img.shields.io/badge/Documentation-orange?style=plastic&logo=gitbook&logoColor=white)](https://art.openpipe.ai)
</div>

## Overview

ART is an open-source reinforcement learning (RL) framework designed to enhance the reliability of your AI agents. It enables LLMs to learn from experience using the Generalized Reward Policy Optimization (GRPO) technique. This framework provides a user-friendly interface for integrating GRPO into any Python application.

## Key Features

*   **No Labeled Data Required:** Train agents without the need for pre-labeled datasets. ART learns by analyzing the tools available on your server.
*   **General-Purpose:** Optimize your models for any Model Context Protocol (MCP) server.
*   **Superior Performance:** Achieve state-of-the-art (SOTA) performance in 2 out of 3 benchmarks.
*   **Effortless Integration:** Requires no modifications to your MCP server.
*   **Flexible Training:** Run the ART client locally or leverage GPU-enabled environments for training.
*   **Observability & Debugging:** Integrations with platforms like W&B, Langfuse, and OpenPipe.
*   **Customization with Defaults:** Configure training parameters or use the default settings optimized for efficiency and stability.

## MCP•RL: Master Your MCP Server

ART makes it easy to train agents to interact with any MCP (Model Context Protocol) server.  Simply provide the server URL, and ART does the rest:

1.  Discovers the server's tools automatically.
2.  Creates input tasks to utilize those tools.
3.  Trains the model to enhance its performance on the MCP server via RULER.
4.  Validates the trained model on new tasks.

### Code Example

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

## Notebook Examples

Jumpstart your ART journey with these hands-on notebooks:

| Agent Task          | Example Notebook                                                                                                             | Description                                     | Comparative Performance                                                                                                                                                                             |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **MCP•RL**          | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/mcp-rl/mcp-rl.ipynb)               | Qwen 2.5 3B masters the NWS MCP server          | [Link coming soon]                                                                                                                                                                                  |
| **ART•E [RULER]**   | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/art-e.ipynb)                       | Qwen 2.5 7B learns to search emails using RULER | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/email_agent/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/art-e/art_e/evaluate/display_benchmarks.ipynb) |
| **2048**            | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)                   | Qwen 2.5 3B learns to play 2048                 | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/2048/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/2048/benchmark_2048.ipynb)                            |
| **Temporal Clue**   | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/temporal_clue/temporal-clue.ipynb) | Qwen 2.5 7B learns to solve Temporal Clue       | [Link coming soon]                                                                                                                                                                                  |
| **Tic Tac Toe**     | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb)     | Qwen 2.5 3B learns to play Tic Tac Toe          | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/tic-tac-toe-local/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/tic_tac_toe/benchmark_tic_tac_toe.ipynb) |
| **Codenames**       | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/codenames/Codenames_RL.ipynb)      | Qwen 2.5 3B learns to play Codenames            | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/codenames/win_rate_over_time.png" height="72"> [benchmarks](/examples/codenames/Codenames_RL.ipynb)                            |
| **AutoRL [RULER]**  | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/auto_rl.ipynb)                     | Train Qwen 2.5 7B to master any task            | [Link coming soon]                                                                                                                                                                                  |

## ART News

Stay informed about the latest advancements and research in agent training:

*   🗞️ **[MCP•RL: Teach Your Model to Master Any MCP Server](https://x.com/corbtt/status/1953171838382817625)** - Automatically train models to effectively use MCP server tools through reinforcement learning.
*   🗞️ **[AutoRL: Zero-Data Training for Any Task](https://x.com/mattshumer_/status/1950572449025650733)** - Train custom AI models without labeled data using automatic input generation and RULER evaluation.
*   🗞️ **[RULER: Easy Mode for RL Rewards](https://openpipe.ai/blog/ruler-easy-mode-for-rl-rewards)** is now available for automatic reward generation in reinforcement learning.
*   🗞️ **[ART·E: How We Built an Email Research Agent That Beats o3](https://openpipe.ai/blog/art-e-mail-agent)** demonstrates a Qwen 2.5 14B email agent outperforming OpenAI's o3.
*   🗞️ **[ART Trainer: A New RL Trainer for Agents](https://openpipe.ai/blog/art-trainer)** enables easy training of LLM-based agents using GRPO.

[📖 See all blog posts →](https://openpipe.ai/blog)

## Installation

Install ART agents from any machine running Python.

```bash
pip install openpipe-art
```

## 🤖 ART•E Agent

Explore a real-world application of ART by checking out the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) blog post.  Learn how we trained Qwen 2.5 14B to outperform o3 in email retrieval!

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## 🔁 Training Loop

ART's architecture is divided into a client and a server. The client (compatible with OpenAI) acts as the interface between ART and your codebase, while the server handles the complexity of inference and training.

1.  **Inference:**
    *   Your code uses the ART client to perform an agentic workflow.
    *   Completion requests are directed to the ART server, which runs the model's latest LoRA in vLLM.
    *   Each `system`, `user`, and `assistant` message is stored in a Trajectory.
    *   When a rollout concludes, your code assigns a `reward` to the Trajectory.
2.  **Training:**
    *   Once rollouts are complete, Trajectories are grouped and sent to the server. Inference is blocked during training.
    *   The server trains your model using GRPO, starting from the latest checkpoint.
    *   The server saves the trained LoRA and loads it into vLLM.
    *   Inference resumes, and the loop restarts.

This process continues until the defined number of iterations is finished.

## 🧩 Supported Models

ART supports most vLLM/HuggingFace-transformers compatible causal language models, including those supported by [Unsloth](https://docs.unsloth.ai/get-started/all-our-models).  If you encounter any issues with specific models, please reach out on [Discord](https://discord.gg/zbBHRUpwf4) or open an issue on [GitHub](https://github.com/openpipe/art/issues).

## 🤝 Contributing

Contributions to ART are highly encouraged! For detailed information on contributing, please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file.

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

ART is built upon the foundation laid by the open-source RL community. We are especially grateful to the creators of:

*   [Unsloth](https://github.com/unslothai/unsloth)
*   [vLLM](https://github.com/vllm-project/vllm)
*   [trl](https://github.com/huggingface/trl)
*   [torchtune](https://github.com/pytorch/torchtune)
*   [SkyPilot](https://github.com/skypilot-org/skypilot)

We appreciate our partners who have helped us test ART.

[pypi-url]: https://pypi.org/project/openpipe-art/
[contribute-url]: https://github.com/openpipe/art/blob/main/CONTRIBUTING.md
[contribute-image]: https://img.shields.io/badge/PRs-welcome-blue.svg