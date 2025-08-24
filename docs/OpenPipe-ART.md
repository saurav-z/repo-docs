<div align="center">
    <a href="https://art.openpipe.ai">
        <picture>
            <img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
        </picture>
    </a>
    <h1>Agent Reinforcement Trainer (ART): Train LLM Agents for Real-World Tasks</h1>
</div>

**ART empowers you to train powerful LLM agents that excel at real-world tasks by using Reinforcement Learning (RL).**

[![PRs-Welcome](https://img.shields.io/badge/PRs-welcome-blue.svg)](https://github.com/openpipe/art/blob/main/CONTRIBUTING.md)
[![PyPI version](https://img.shields.io/pypi/v/openpipe-art?color=364fc7)](https://pypi.org/project/openpipe-art/)
[![Train Agent](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)
[![Join Discord](https://img.shields.io/badge/Join%20Discord-5865F2?style=plastic&logo=discord&logoColor=white)](https://discord.gg/zbBHRUpwf4)
[![Documentation](https://img.shields.io/badge/Documentation-orange?style=plastic&logo=gitbook&logoColor=white)](https://art.openpipe.ai)

## Key Features of ART

*   **Effortless Agent Training:** Easily train LLM agents for diverse tasks using GRPO, accelerating the development process.
*   **No Labeled Data Needed:** ART leverages GRPO to learn from experience, eliminating the need for extensive labeled datasets.
*   **General Purpose:**  ART is designed to optimize models for any Model Context Protocol (MCP) server, offering broad applicability.
*   **Superior Performance:** ART achieves competitive performance in various benchmarks, demonstrating its effectiveness.
*   **Simple Integration:** Integrate ART with your existing MCP server with minimal customization required.

## MCP•RL: Mastering MCP Servers

With MCP•RL, train your agents to excel at using any Model Context Protocol (MCP) server by providing the server URL.  MCP•RL does the rest:

1.  **Automated Discovery:**  Automatically identifies the tools available on your MCP server.
2.  **Intelligent Task Design:**  Generates input tasks that effectively utilize the identified tools.
3.  **GRPO Optimization:**  Trains the model to enhance performance on the MCP server by using GRPO.
4.  **Performance Validation:**  Tests the trained model on new tasks to confirm its effectiveness.

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

## ART Overview: The Open-Source RL Framework

ART is an open-source RL framework that dramatically improves agent reliability and performance by enabling LLMs to learn through experience.  This allows you to integrate GRPO into any Python application with an ergonomic harness. Explore more at the [docs](https://art.openpipe.ai).

## Example Notebooks: Get Started Quickly

Explore these example notebooks to learn how to train agents with ART:

| Agent Task         | Example Notebook                                                                                                            | Description                                     | Comparative Performance                                                                                                                                                                             |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **MCP•RL**         | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/mcp-rl/mcp-rl.ipynb)               | Qwen 2.5 3B masters the NWS MCP server          | [Link coming soon]                                                                                                                                                                                  |
| **ART•E [RULER]**  | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/art-e.ipynb)                       | Qwen 2.5 7B learns to search emails using RULER | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/email_agent/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/art-e/art_e/evaluate/display_benchmarks.ipynb) |
| **2048**           | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)                   | Qwen 2.5 3B learns to play 2048                 | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/2048/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/2048/benchmark_2048.ipynb)                            |
| **Temporal Clue**  | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/temporal_clue/temporal-clue.ipynb) | Qwen 2.5 7B learns to solve Temporal Clue       | [Link coming soon]                                                                                                                                                                                  |
| **Tic Tac Toe**    | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb)     | Qwen 2.5 3B learns to play Tic Tac Toe          | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/tic-tac-toe-local/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/tic_tac_toe/benchmark_tic_tac_toe.ipynb) |
| **Codenames**      | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/codenames/Codenames_RL.ipynb)      | Qwen 2.5 3B learns to play Codenames            | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/codenames/win_rate_over_time.png" height="72"> [benchmarks](/examples/codenames/Codenames_RL.ipynb)                            |
| **AutoRL [RULER]** | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/auto_rl.ipynb)                     | Train Qwen 2.5 7B to master any task            | [Link coming soon]                                                                                                                                                                                  |

## ART News and Updates

Stay up-to-date with the latest advancements in ART and the broader RL landscape:

*   🗞️ **[MCP•RL: Teach Your Model to Master Any MCP Server](https://x.com/corbtt/status/1953171838382817625)** -  Automatically train models to effectively use MCP server tools through reinforcement learning.
*   🗞️ **[AutoRL: Zero-Data Training for Any Task](https://x.com/mattshumer_/status/1950572449025650733)** - Train custom AI models without labeled data using automatic input generation and RULER evaluation.
*   🗞️ **[RULER: Easy Mode for RL Rewards](https://openpipe.ai/blog/ruler-easy-mode-for-rl-rewards)** is now available for automatic reward generation in reinforcement learning.
*   🗞️ **[ART·E: How We Built an Email Research Agent That Beats o3](https://openpipe.ai/blog/art-e-mail-agent)** demonstrates a Qwen 2.5 14B email agent outperforming OpenAI's o3.
*   🗞️ **[ART Trainer: A New RL Trainer for Agents](https://openpipe.ai/blog/art-trainer)** enables easy training of LLM-based agents using GRPO.

[📖 See all blog posts →](https://openpipe.ai/blog)

## Why Choose ART?

*   **Simplified Integration:** ART simplifies the process of incorporating RL training into your existing applications.
*   **Flexible Training:**  ART allows you to train agents from your laptop and utilize ART server to kick off GPU-enabled environments.
*   **Enhanced Observability:** Integrations with platforms like W&B, Langfuse, and OpenPipe provide comprehensive observability, streamlining debugging.
*   **Intelligent Defaults and Customization:** ART offers optimized default training parameters and is highly customizable to meet specific needs.

## Installation

Easily install ART using pip:

```bash
pip install openpipe-art
```

## 🤖 ART•E Agent: Real-World Application

Learn how ART can be used in a real-world scenario by checking out the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) blog post. Discover how we trained Qwen 2.5 14B to surpass o3 in email retrieval.

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## 🔁 Training Loop in Detail

ART's functionality is split between a **client** and a **server**.

1.  **Inference:**
    1.  Use the ART client to run an agentic workflow, executing several rollouts in parallel.
    2.  Completion requests are sent to the ART server, which runs the model's latest LoRA using vLLM.
    3.  Each `system`, `user`, and `assistant` message is stored in a Trajectory.
    4.  When a rollout finishes, your code assigns a `reward` to its Trajectory.
2.  **Training:**
    1.  After each rollout finishes, Trajectories are grouped and sent to the server.
    2.  The server trains your model using GRPO.
    3.  The server saves the newly trained LoRA and loads it into vLLM.
    4.  Inference is unblocked.

This loop continues until the specified iterations are complete.

## 🧩 Supported Models

ART is compatible with most vLLM/HuggingFace-transformers compatible causal language models, including those supported by [Unsloth](https://docs.unsloth.ai/get-started/all-our-models). Currently, Gemma 3 is not supported. Please reach out on [Discord](https://discord.gg/zbBHRUpwf4) or open a [GitHub issue](https://github.com/openpipe/art/issues) if you have any issues.

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

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

Licensed under the [Apache-2.0 License](LICENSE).

## 🙏 Credits

ART builds upon the work of many.  Special thanks to:

*   [Unsloth](https://github.com/unslothai/unsloth)
*   [vLLM](https://github.com/vllm-project/vllm)
*   [trl](https://github.com/huggingface/trl)
*   [torchtune](https://github.com/pytorch/torchtune)
*   [SkyPilot](https://github.com/skypilot-org/skypilot)

We're grateful for the support of our partners.

[pypi-url]: https://pypi.org/project/openpipe-art/
[contribute-url]: https://github.com/openpipe/art/blob/main/CONTRIBUTING.md
[contribute-image]: https://img.shields.io/badge/PRs-welcome-blue.svg