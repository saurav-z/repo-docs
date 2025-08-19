<div align="center">
  <a href="https://art.openpipe.ai">
    <picture>
      <img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
    </picture>
  </a>
  <h1>Agent Reinforcement Trainer (ART): Supercharge Your AI Agents with GRPO</h1>
</div>

<!-- Introduction -->
ART empowers you to train and refine AI agents to excel at complex, real-world tasks using GRPO (Gradient Reinforcement Policy Optimization), a powerful and efficient training method.  [Explore the ART repository](https://github.com/OpenPipe/ART) and start building smarter agents today!

<!-- Badges -->
[![PRs-Welcome](https://img.shields.io/badge/PRs-welcome-blue.svg)](https://github.com/openpipe/art/blob/main/CONTRIBUTING.md)
[![PyPI version](https://img.shields.io/pypi/v/openpipe-art?color=364fc7)](https://pypi.org/project/openpipe-art/)
[![Train Agent](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)
[![Join Discord](https://img.shields.io/badge/Join%20Discord-5865F2?style=plastic&logo=discord&logoColor=white)](https://discord.gg/zbBHRUpwf4)
[![Documentation](https://img.shields.io/badge/Documentation-orange?style=plastic&logo=gitbook&logoColor=white)](https://art.openpipe.ai)

<!-- Key Features Section -->
## Key Features of ART

*   **Train Agents Without Labeled Data:** ART leverages GRPO to learn from experience, reducing the need for expensive labeled datasets.
*   **General-Purpose:**  Designed to optimize models for various tasks and environments, including MCP servers.
*   **High Performance:** Achieve or surpass state-of-the-art (SOTA) results in key benchmarks.
*   **Easy Integration:** Seamlessly integrate with your existing projects; no complex modifications to your systems are needed.

## MCP•RL: Train Agents to Master Model Context Protocol (MCP) Servers

<img src="assets/MCP_RL_diagram.svg" width="7000">

**MCP•RL** is a core component of ART that enables you to train agents to effectively use any MCP (Model Context Protocol) server. Just provide the server URL, and MCP•RL will:

1.  **Discover Server Tools:** Automatically identifies available tools.
2.  **Generate Tasks:** Creates input scenarios to utilize those tools.
3.  **Optimize with RULER:** Trains the model to improve performance on the MCP server.
4.  **Validate & Test:** Evaluates performance on new tasks.

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

<!-- ART Overview Section -->
## ART Overview: The Open-Source RL Framework for Smarter Agents

ART is an open-source RL framework built to enhance agent reliability by enabling LLMs to **learn from experience**.  It simplifies the integration of GRPO into any Python application, providing a streamlined and effective way to train intelligent agents.  For a quick hands-on start, check out the example notebooks below. To delve deeper, explore the [ART documentation](https://art.openpipe.ai).

<!-- Notebooks Section -->
## Example Notebooks: Get Started with ART

Explore these notebooks to see ART in action across different tasks:

| Agent Task         | Example Notebook                                                                                                             | Description                                     | Comparative Performance                                                                                                                                                                             |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **MCP•RL**         | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/mcp-rl/mcp-rl.ipynb)               | Qwen 2.5 3B masters the NWS MCP server          | [Link coming soon]                                                                                                                                                                                  |
| **ART•E [RULER]**  | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/art-e.ipynb)                       | Qwen 2.5 7B learns to search emails using RULER | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/email_agent/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/art-e/art_e/evaluate/display_benchmarks.ipynb) |
| **2048**           | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)                   | Qwen 2.5 3B learns to play 2048                 | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/2048/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/2048/benchmark_2048.ipynb)                            |
| **Temporal Clue**  | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/temporal_clue/temporal-clue.ipynb) | Qwen 2.5 7B learns to solve Temporal Clue       | [Link coming soon]                                                                                                                                                                                  |
| **Tic Tac Toe**    | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb)     | Qwen 2.5 3B learns to play Tic Tac Toe          | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/tic-tac-toe-local/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/tic_tac_toe/benchmark_tic_tac_toe.ipynb) |
| **Codenames**      | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/codenames/Codenames_RL.ipynb)      | Qwen 2.5 3B learns to play Codenames            | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/codenames/win_rate_over_time.png" height="72"> [benchmarks](/examples/codenames/Codenames_RL.ipynb)                            |
| **AutoRL [RULER]** | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/auto_rl.ipynb)                     | Train Qwen 2.5 7B to master any task            | [Link coming soon]                                                                                                                                                                                  |

<!-- ART News Section -->
## Stay Updated: ART News

Discover our latest research, advancements, and updates on building state-of-the-art agents.

*   🗞️ **[MCP•RL: Teach Your Model to Master Any MCP Server](https://x.com/corbtt/status/1953171838382817625)** - Train models to effectively use MCP server tools using reinforcement learning.
*   🗞️ **[AutoRL: Zero-Data Training for Any Task](https://x.com/mattshumer_/status/1950572449025650733)** - Train custom AI models without labeled data using automatic input generation and RULER evaluation.
*   🗞️ **[RULER: Easy Mode for RL Rewards](https://openpipe.ai/blog/ruler-easy-mode-for-rl-rewards)** is now available for automatic reward generation in reinforcement learning.
*   🗞️ **[ART·E: How We Built an Email Research Agent That Beats o3](https://openpipe.ai/blog/art-e-mail-agent)** showcases a Qwen 2.5 14B email agent outperforming OpenAI's o3.
*   🗞️ **[ART Trainer: A New RL Trainer for Agents](https://openpipe.ai/blog/art-trainer)** allows for simple training of LLM-based agents using GRPO.

[📖 See all blog posts →](https://openpipe.ai/blog)

<!-- Why ART Section -->
## Why Choose ART?

*   **Seamless Integration:** Easily integrate RL training into your existing applications. ART provides modular components, simplifying interactions with the training server.
*   **Flexible Training:** Run the ART client locally while leveraging an ephemeral, GPU-enabled ART server, or utilize a local GPU.
*   **Observability and Debugging:** ART integrates with platforms like W&B, Langfuse, and OpenPipe for enhanced observability and simplified debugging.
*   **Intelligent Defaults:** Enjoy customizable training parameters and inference engine configurations with intelligent defaults optimized for efficiency and stability.

<!-- Installation Section -->
## Installation

To install ART and integrate it into your project, simply run:

```bash
pip install openpipe-art
```

<!-- ART E Agent Section -->
## 🤖 ART•E Agent: A Real-World Application

Learn how ART can be used to build practical applications. Check out the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) blog post to see how we trained a Qwen 2.5 14B model to outperform o3 in email retrieval.

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

<!-- Training Loop Section -->
## 🔁 Training Loop Explained

ART employs a client-server architecture for its training loop. Here's how it works:

1.  **Inference:**
    1.  Your code uses the ART client to perform agentic workflows.
    2.  Completion requests are sent to the ART server, which runs the latest LoRA in vLLM.
    3.  Each `system`, `user`, and `assistant` message is stored as a Trajectory.
    4.  When a rollout finishes, your code assigns a `reward`.

2.  **Training:**
    1.  Trajectories are grouped and sent to the server. Inference is blocked.
    2.  The server trains your model using GRPO, initializing from the latest checkpoint.
    3.  The server saves the LoRA and loads it into vLLM.
    4.  Inference resumes.

This loop repeats until a set number of iterations are complete.

<!-- Supported Models Section -->
## 🧩 Supported Models

ART is designed to work with most vLLM/HuggingFace-transformers compatible causal language models, particularly those supported by [Unsloth](https://docs.unsloth.ai/get-started/all-our-models). We do not currently support Gemma 3. If you encounter issues with other models, please share your feedback on [Discord](https://discord.gg/zbBHRUpwf4) or open an issue on [GitHub](https://github.com/openpipe/art/issues)!

<!-- Contributing Section -->
## 🤝 Contributing

ART is under active development, and contributions are welcome! Please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file for more details.

<!-- Citation Section -->
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

<!-- License Section -->
## ⚖️ License

This repository's source code is available under the [Apache-2.0 License](LICENSE).

<!-- Credits Section -->
## 🙏 Credits

ART is built on the work of many, and we are grateful to the open-source RL community, especially:

*   [Unsloth](https://github.com/unslothai/unsloth)
*   [vLLM](https://github.com/vllm-project/vllm)
*   [trl](https://github.com/huggingface/trl)
*   [torchtune](https://github.com/pytorch/torchtune)
*   [SkyPilot](https://github.com/skypilot-org/skypilot)

Thank you to our partners for helping us test ART and we look forward to seeing what you build!