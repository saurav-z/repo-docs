<div align="center">

<a href="https://art.openpipe.ai"><picture>
<img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
</picture></a>

<p align="center">
  <h1>Agent Reinforcement Trainer (ART)</h1>
</p>

<p>
  Train powerful, multi-step agents for real-world tasks with Agent Reinforcement Trainer (ART).
</p>

[![PRs-Welcome][contribute-image]][contribute-url]
[![PyPI version](https://img.shields.io/pypi/v/openpipe-art?color=364fc7)][pypi-url]
[![Train Agent](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)

[![Join Discord](https://img.shields.io/badge/Join%20Discord-5865F2?style=plastic&logo=discord&logoColor=white)](https://discord.gg/zbBHRUpwf4)
[![Documentation](https://img.shields.io/badge/Documentation-orange?style=plastic&logo=gitbook&logoColor=white)](https://art.openpipe.ai)

</div>

## Unleash the Power of AI Agents with Agent Reinforcement Trainer (ART)

ART is an open-source framework for training powerful AI agents that learn from experience, enabling you to build and deploy AI agents capable of complex, multi-step tasks.  **Learn how to build and deploy your own AI agents to solve complex tasks.**  See the [original repo](https://github.com/OpenPipe/ART).

### Key Features:

*   **Effortless Agent Training:** Quickly integrate ART into your Python applications.
*   **No Labeled Data Required:** Train agents using reinforcement learning without the need for pre-labeled datasets, using RULER for reward generation.
*   **General-Purpose:**  ART can optimize models for any Model Context Protocol (MCP) server or custom task.
*   **High Performance:** Achieve state-of-the-art (SOTA) results on various benchmarks.
*   **Easy Integration:**  Minimal setup required to use your existing MCP server.

### Get Started:

Install the ART package:

```bash
pip install openpipe-art
```

Learn more about ART with these notebooks:

| Agent Task         | Example Notebook                                                                                                             | Description                                     | Comparative Performance                                                                                                                                                                             |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **MCP•RL**         | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/mcp-rl/mcp-rl.ipynb)               | Qwen 2.5 3B masters the NWS MCP server          | [Link coming soon]                                                                                                                                                                                  |
| **ART•E [RULER]**  | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/art-e.ipynb)                       | Qwen 2.5 7B learns to search emails using RULER | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/email_agent/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/art-e/art_e/evaluate/display_benchmarks.ipynb) |
| **2048**           | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)                   | Qwen 2.5 3B learns to play 2048                 | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/2048/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/2048/benchmark_2048.ipynb)                            |
| **Temporal Clue**  | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/temporal_clue/temporal-clue.ipynb) | Qwen 2.5 7B learns to solve Temporal Clue       | [Link coming soon]                                                                                                                                                                                  |
| **Tic Tac Toe**    | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb)     | Qwen 2.5 3B learns to play Tic Tac Toe          | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/tic-tac-toe-local/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/tic_tac_toe/benchmark_tic_tac_toe.ipynb) |
| **Codenames**      | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/codenames/Codenames_RL.ipynb)      | Qwen 2.5 3B learns to play Codenames            | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/codenames/win_rate_over_time.png" height="72"> [benchmarks](/examples/codenames/Codenames_RL.ipynb)                            |
| **AutoRL [RULER]** | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/auto_rl.ipynb)                     | Train Qwen 2.5 7B to master any task            | [Link coming soon]                                                                                                                                                                                  |

## How ART Works: MCP•RL

With ART, it's easy to train an agent to master Model Context Protocol (MCP) servers:

1.  **Automatic Tool Discovery:** ART automatically identifies and utilizes the available tools on the server.
2.  **Dynamic Input Task Generation:** ART designs input tasks that effectively leverage those tools.
3.  **Reinforcement Learning with RULER:**  The model is trained using reinforcement learning, improving its performance on the MCP server.
4.  **Validation Testing:** Performance is verified on new tasks.

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

## Core Components:

ART employs a client-server architecture to handle the training process efficiently:

1.  **Client:** Your code interacts with the ART client, which handles agent workflows and communicates with the server.
2.  **Server:** Runs independently on a GPU-enabled machine, managing the inference and training processes.

**Training Loop Overview:**

1.  **Inference:** The ART client performs agent actions and sends requests to the server, which runs the latest model in vLLM.
2.  **Reward Assignment:** Trajectories are generated and given a reward based on agent performance.
3.  **Training:**  The server groups trajectories and trains the model with GRPO, initialized with the latest checkpoint, then saves the model and loads it into vLLM.

## 📰 ART News and Resources

Stay up-to-date with the latest developments in ART:

*   🗞️ **[MCP•RL: Teach Your Model to Master Any MCP Server](https://x.com/corbtt/status/1953171838382817625)** - Automatically train models to effectively use MCP server tools through reinforcement learning.
*   🗞️ **[AutoRL: Zero-Data Training for Any Task](https://x.com/mattshumer_/status/1950572449025650733)** - Train custom AI models without labeled data using automatic input generation and RULER evaluation.
*   🗞️ **[RULER: Easy Mode for RL Rewards](https://openpipe.ai/blog/ruler-easy-mode-for-rl-rewards)** is now available for automatic reward generation in reinforcement learning.
*   🗞️ **[ART·E: How We Built an Email Research Agent That Beats o3](https://openpipe.ai/blog/art-e-mail-agent)** demonstrates a Qwen 2.5 14B email agent outperforming OpenAI's o3.
*   🗞️ **[ART Trainer: A New RL Trainer for Agents](https://openpipe.ai/blog/art-trainer)** enables easy training of LLM-based agents using GRPO.

[📖 See all blog posts →](https://openpipe.ai/blog)

## Why Choose ART?

*   **Simplified Integration:** Easily introduce RL training into your existing applications with convenient wrappers.
*   **Flexible Training:**  Train agents locally or leverage GPU-enabled environments via the ART server.
*   **Enhanced Debugging:** Take advantage of integrations with W&B, Langfuse, and OpenPipe for easy observability.
*   **Intelligent Defaults:** ART offers optimized default settings that can be customized for specific needs.

## 🤖 ART•E Agent Example

Explore a real-world application of ART with the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) blog post, which details how Qwen 2.5 14B was trained to outperform o3 at email retrieval.

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## Supported Models

ART supports a wide range of causal language models compatible with vLLM/HuggingFace-transformers, including those listed at [Unsloth](https://docs.unsloth.ai/get-started/all-our-models). Please report issues or ask for assistance on [Discord](https://discord.gg/zbBHRUpwf4) or [GitHub](https://github.com/openpipe/art/issues) if you encounter model compatibility problems.

## 🤝 Contributing

We welcome contributions!  Check out the [CONTRIBUTING.md](CONTRIBUTING.md) file for more information.

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

This repository is available under the [Apache-2.0 License](LICENSE).

## 🙏 Credits

ART is built upon the work of many in the open source RL community, especially:

*   [Unsloth](https://github.com/unslothai/unsloth)
*   [vLLM](https://github.com/vllm-project/vllm)
*   [trl](https://github.com/huggingface/trl)
*   [torchtune](https://github.com/pytorch/torchtune)
*   [SkyPilot](https://github.com/skypilot-org/skypilot)

We appreciate the contributions of our partners!