<div align="center">

<a href="https://art.openpipe.ai"><picture>
<img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
</picture></a>

<p align="center">
  <h1>Agent Reinforcement Trainer (ART)</h1>
</p>

</div>

## Effortlessly Train LLM Agents with ART: The Open-Source Reinforcement Learning Framework

ART empowers you to train powerful, reliable AI agents by leveraging Reinforcement Learning (RL) techniques. Build agents that can perform multi-step tasks and excel in real-world scenarios.

**[Explore the ART Repository](https://github.com/OpenPipe/ART)**

[![PRs-Welcome][contribute-image]][contribute-url]
[![PyPI version](https://img.shields.io/pypi/v/openpipe-art?color=364fc7)][pypi-url]
[![Train Agent](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)
[![Join Discord](https://img.shields.io/badge/Join%20Discord-5865F2?style=plastic&logo=discord&logoColor=white)](https://discord.gg/zbBHRUpwf4)
[![Documentation](https://img.shields.io/badge/Documentation-orange?style=plastic&logo=gitbook&logoColor=white)](https://art.openpipe.ai)

## Key Features of ART

*   **Zero-Shot Learning:** Train agents without the need for labeled data using innovative techniques.
*   **Versatile Applications:** Optimize models for any Model Context Protocol (MCP) server.
*   **Superior Performance:** Achieve state-of-the-art (SOTA) results in various benchmarks.
*   **Easy Integration:** Integrate seamlessly with your existing projects without requiring significant changes to your MCP server.
*   **Flexible Training:** Train locally or leverage cloud resources for efficient experimentation.

## MCP•RL: Reinforcement Learning for Model Context Protocols

Train agents to effectively utilize any Model Context Protocol (MCP) server.

**How it Works:**

1.  **Discover Tools:** Automatically identify and analyze the tools offered by the server.
2.  **Generate Tasks:** Design input tasks that leverage the server's tools.
3.  **Train & Improve:** Train the model to enhance performance on the MCP server using RULER (Relative Utility Learning with Efficient Rewards).
4.  **Validate Performance:** Test the trained model's effectiveness with new tasks.

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

## ART Overview: RL Framework for Agent Improvement

ART is an open-source Reinforcement Learning (RL) framework designed to enhance agent reliability by enabling Language Learning Models (LLMs) to learn from experience. ART provides a streamlined interface for integrating GRPO into any Python application.

## 🚀 Quick Start: Example Notebooks

Jumpstart your ART journey with these interactive notebooks:

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

Stay up-to-date with the latest research and advancements in ART:

*   🗞️ **[MCP•RL: Teach Your Model to Master Any MCP Server](https://x.com/corbtt/status/1953171838382817625)** - Automatically train models to effectively use MCP server tools through reinforcement learning.
*   🗞️ **[AutoRL: Zero-Data Training for Any Task](https://x.com/mattshumer_/status/1950572449025650733)** - Train custom AI models without labeled data using automatic input generation and RULER evaluation.
*   🗞️ **[RULER: Easy Mode for RL Rewards](https://openpipe.ai/blog/ruler-easy-mode-for-rl-rewards)** is now available for automatic reward generation in reinforcement learning.
*   🗞️ **[ART·E: How We Built an Email Research Agent That Beats o3](https://openpipe.ai/blog/art-e-mail-agent)** demonstrates a Qwen 2.5 14B email agent outperforming OpenAI's o3.
*   🗞️ **[ART Trainer: A New RL Trainer for Agents](https://openpipe.ai/blog/art-trainer)** enables easy training of LLM-based agents using GRPO.

[📖 See all blog posts →](https://openpipe.ai/blog)

## Why Choose ART?

*   **Simplified Integration:** ART provides convenient wrappers for incorporating RL training into existing applications.
*   **Flexible Training Environments:** Run the ART client on your local machine and leverage ART's server for cloud-based GPU acceleration.
*   **Enhanced Observability:** Utilize integrations with platforms like W&B, Langfuse, and OpenPipe to simplify debugging.
*   **Intelligent Defaults:** Take advantage of optimized default configurations, or customize training parameters and inference engine configurations.

## 🛠️ Installation

Get started by installing the ART package:

```bash
pip install openpipe-art
```

## 🤖 ART•E Agent in Action

Explore a real-world use case with the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) blog post, demonstrating how Qwen 2.5 14B was trained to outperform o3 in email retrieval.

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## 🔁 Training Loop Deep Dive

ART operates with a client-server architecture. The client interacts with your LLM and the server handles the RL training.

**Key Components:**

1.  **Inference:**
    *   Your code uses the ART client to perform an agentic workflow.
    *   Completion requests are routed to the ART server.
    *   Each `system`, `user`, and `assistant` message is stored in a Trajectory.
    *   A `reward` is assigned to each Trajectory.
2.  **Training:**
    *   Trajectories are grouped and sent to the server.
    *   The server trains your model using GRPO.
    *   The server saves and loads the trained model.
    *   Inference resumes.

This training loop continues until the set iterations are complete.

## 🧩 Supported Models

ART is designed to be compatible with most vLLM/HuggingFace-transformers-compatible causal language models. For model support, refer to the [Unsloth](https://docs.unsloth.ai/get-started/all-our-models). Report any model incompatibility issues on our [Discord](https://discord.gg/zbBHRUpwf4) or through a [GitHub issue](https://github.com/openpipe/art/issues).

## 🤝 Contribute to ART

ART is an actively evolving project, and contributions are highly encouraged! Refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file for detailed information.

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

ART is licensed under the [Apache-2.0 License](LICENSE).

## 🙏 Acknowledgements

ART is built upon the work of numerous researchers and open-source projects. We are particularly grateful to the following:

*   [Unsloth](https://github.com/unslothai/unsloth)
*   [vLLM](https://github.com/vllm-project/vllm)
*   [trl](https://github.com/huggingface/trl)
*   [torchtune](https://github.com/pytorch/torchtune)
*   [SkyPilot](https://github.com/skypilot-org/skypilot)

We extend our thanks to our partners for their invaluable support in testing ART.
[pypi-url]: https://pypi.org/project/openpipe-art/
[contribute-url]: https://github.com/openpipe/art/blob/main/CONTRIBUTING.md
[contribute-image]: https://img.shields.io/badge/PRs-welcome-blue.svg