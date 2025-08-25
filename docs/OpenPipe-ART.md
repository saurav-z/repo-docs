<div align="center">
    <a href="https://art.openpipe.ai">
        <picture>
            <img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
        </picture>
    </a>
</div>

# Agent Reinforcement Trainer (ART): Train LLM Agents with Ease

**ART empowers you to train powerful AI agents for real-world tasks using Reinforcement Learning from Experience (RLfE).** Explore the [ART repository](https://github.com/OpenPipe/ART) for cutting-edge agent training!

## Key Features

*   **No Labeled Data Required:** ART leverages RLfE to learn from agent interactions, eliminating the need for pre-labeled datasets.
*   **General-Purpose Training:**  ART optimizes LLMs for a wide range of applications, adapting to various environments and tasks.
*   **State-of-the-Art Performance:** Achieve or surpass leading performance metrics in several benchmark tasks.
*   **Simple Integration:** Easily integrate ART into your projects without requiring extensive modifications to your existing systems.
*   **Modular Client-Server Architecture**: Easy integration into your applications with an OpenAI-compatible client and a server to run complex training tasks.

## Quickstart: Train Your Agent with MCP•RL

**MCP•RL** provides a simple way to train agents for complex tasks such as interacting with any Model Context Protocol (MCP) server with minimal setup. This includes automatic tool discovery, input task design, and LLM training using GRPO!

*   **Automatically Discover Server Tools:** Analyze server tools to design training scenarios.
*   **Generate Training Data:** Focus your efforts on the architecture, while ART generates tasks.
*   **Leverage GRPO (RLfE):** Optimize model performance via GRPO, enabling agents to learn from their experience.

**Example Training Code:**

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

## Core Capabilities of ART

ART is built on the foundation of Reinforcement Learning from Experience (RLfE), providing a robust framework for agent training:

*   **Ergonomic Framework:** An efficient and streamlined way to introduce RL training into any Python application.
*   **Train from Anywhere:** Run the ART client from your laptop and the ART server on the cloud.
*   **Built-in Integrations:** Integration with hosted platforms like W\&B, Langfuse, and OpenPipe for flexible observability and simplified debugging.
*   **Intelligent Defaults:** Start training your models with optimized parameters right out of the box.

##  Explore ART with Ready-to-Use Notebooks

Jumpstart your agent training with the following notebooks:

| Agent Task          | Example Notebook                                                                                                                    | Description                                      | Comparative Performance                                                                                                                                                                             |
| :------------------ | :---------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **MCP•RL**          | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/mcp-rl/mcp-rl.ipynb)              | Qwen 2.5 3B masters the NWS MCP server         | [Link coming soon]                                                                                                                                                                                  |
| **ART•E [RULER]**   | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/art-e.ipynb)                      | Qwen 2.5 7B learns to search emails using RULER | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/email_agent/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/art-e/art_e/evaluate/display_benchmarks.ipynb) |
| **2048**            | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/2048/2048.ipynb)                  | Qwen 2.5 3B learns to play 2048                | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/2048/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/2048/benchmark_2048.ipynb)                            |
| **Temporal Clue**   | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/temporal_clue/temporal-clue.ipynb) | Qwen 2.5 7B learns to solve Temporal Clue      | [Link coming soon]                                                                                                                                                                                  |
| **Tic Tac Toe**     | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb)    | Qwen 2.5 3B learns to play Tic Tac Toe         | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/tic-tac-toe-local/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/tic_tac_toe/benchmark_tic_tac_toe.ipynb) |
| **Codenames**       | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/codenames/Codenames_RL.ipynb)     | Qwen 2.5 3B learns to play Codenames           | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/codenames/win_rate_over_time.png" height="72"> [benchmarks](/examples/codenames/Codenames_RL.ipynb)                            |
| **AutoRL [RULER]**  | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/auto_rl.ipynb)                    | Train Qwen 2.5 7B to master any task           | [Link coming soon]                                                                                                                                                                                  |

## Stay Updated: ART News and Developments

Keep up-to-date with the latest advancements:

*   **[MCP•RL: Teach Your Model to Master Any MCP Server](https://x.com/corbtt/status/1953171838382817625)** - Automatically train models to effectively use MCP server tools through reinforcement learning.
*   **[AutoRL: Zero-Data Training for Any Task](https://x.com/mattshumer_/status/1950572449025650733)** - Train custom AI models without labeled data using automatic input generation and RULER evaluation.
*   **[RULER: Easy Mode for RL Rewards](https://openpipe.ai/blog/ruler-easy-mode-for-rl-rewards)** - Automatic reward generation in reinforcement learning.
*   **[ART•E: How We Built an Email Research Agent That Beats o3](https://openpipe.ai/blog/art-e-mail-agent)** - A Qwen 2.5 14B email agent outperforming OpenAI's o3.
*   **[ART Trainer: A New RL Trainer for Agents](https://openpipe.ai/blog/art-trainer)** - Easily train LLM-based agents using GRPO.

[📖 See all blog posts →](https://openpipe.ai/blog)

## Installation

Get started with ART by installing the openpipe-art package:

```bash
pip install openpipe-art
```

## ART•E Agent

Dive deeper into real-world agent training by exploring the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) blog post. Learn how Qwen 2.5 14B was trained to surpass o3 in email retrieval!

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## Training Loop Architecture

ART utilizes a client-server model, allowing you to separate the complex RL training from your core application logic:

1.  **Inference Phase:** The client sends requests to the ART server, which then runs the model's latest LoRA in vLLM.
2.  **Training Phase:** After rollouts finish, trajectories are sent to the server, where the GRPO training occurs.

## Supported Models

ART is designed to work with most vLLM/HuggingFace-transformers compatible causal language models. Ensure the models you're interested in using are compatible with [Unsloth](https://docs.unsloth.ai/get-started/all-our-models).

## Contributing

Contributions are greatly appreciated!  Review the [CONTRIBUTING.md](CONTRIBUTING.md) file for more information.

## Citation

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

## License

This project is available under the [Apache-2.0 License](LICENSE).

## Acknowledgements

ART is built with the help of these fantastic projects and contributors:

*   [Unsloth](https://github.com/unslothai/unsloth)
*   [vLLM](https://github.com/vllm-project/vllm)
*   [trl](https://github.com/huggingface/trl)
*   [torchtune](https://github.com/pytorch/torchtune)
*   [SkyPilot](https://github.com/skypilot-org/skypilot)

Thank you to our partners who have helped us test ART!