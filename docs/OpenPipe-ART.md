<div align="center">
  <a href="https://art.openpipe.ai">
    <picture>
      <img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
    </picture>
  </a>

  <h1>Agent Reinforcement Trainer (ART)</h1>

  <p>
    <b>Supercharge your LLMs: ART enables you to train multi-step agents for complex, real-world tasks using advanced Reinforcement Learning techniques.</b>
    <br>
    <a href="https://github.com/OpenPipe/ART">
      <img src="https://img.shields.io/github/stars/OpenPipe/ART?style=social" alt="GitHub Stars">
    </a>
  </p>
</div>

## Key Features

*   **No Labeled Data Required:** ART leverages reinforcement learning to train agents without the need for pre-labeled datasets, saving time and resources.
*   **General-Purpose:** Easily optimize LLM agents for a wide variety of tasks and environments.
*   **High Performance:** Achieve or surpass state-of-the-art (SOTA) performance in numerous benchmarks.
*   **Easy Integration:** Seamlessly integrate ART into your existing projects with minimal setup.
*   **MCP•RL Support:** Train agents to master Model Context Protocol (MCP) servers automatically.

## What is ART?

ART is an open-source reinforcement learning (RL) framework built to improve agent reliability, enabling LLMs to learn from experience. It provides a streamlined interface for integrating GRPO (Generalized Reward Policy Optimization) into your Python applications. Whether you are an AI enthusiast or a seasoned developer, ART empowers you to create sophisticated agents that excel in complex tasks.

## Getting Started with ART

### Installation

Get started by installing the ART library:

```bash
pip install openpipe-art
```

### Example: MCP•RL - Train an agent to master MCP servers

ART simplifies training agents to interact with any Model Context Protocol (MCP) server. Just provide the server URL, and ART handles the rest!

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

### Explore our Notebooks

Dive into practical examples with our interactive notebooks. Each notebook demonstrates ART's capabilities across diverse agent tasks.

| Agent Task          | Example Notebook                                                                                                             | Description                                        | Benchmarks                                                                                                                                                                              |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **MCP•RL**          | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/mcp-rl/mcp-rl.ipynb)               | Qwen 2.5 3B masters the NWS MCP server             | [Link coming soon]                                                                                                                                                                       |
| **ART•E [RULER]**   | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/art-e.ipynb)                       | Qwen 2.5 7B learns to search emails using RULER    | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/email_agent/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/art-e/art_e/evaluate/display_benchmarks.ipynb) |
| **2048**            | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)                   | Qwen 2.5 3B learns to play 2048                    | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/2048/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/2048/benchmark_2048.ipynb)           |
| **Temporal Clue**   | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/temporal_clue/temporal-clue.ipynb) | Qwen 2.5 7B learns to solve Temporal Clue          | [Link coming soon]                                                                                                                                                                       |
| **Tic Tac Toe**     | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb)     | Qwen 2.5 3B learns to play Tic Tac Toe             | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/tic-tac-toe-local/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/tic_tac_toe/benchmark_tic_tac_toe.ipynb) |
| **Codenames**       | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/codenames/Codenames_RL.ipynb)      | Qwen 2.5 3B learns to play Codenames               | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/codenames/win_rate_over_time.png" height="72"> [benchmarks](/examples/codenames/Codenames_RL.ipynb)           |
| **AutoRL [RULER]**  | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/auto_rl.ipynb)                     | Train Qwen 2.5 7B to master any task               | [Link coming soon]                                                                                                                                                                       |

## ART in Action

See how ART is used to solve real-world problems:

*   **[MCP•RL: Teach Your Model to Master Any MCP Server](https://x.com/corbtt/status/1953171838382817625)** Train models to use MCP server tools through RL.
*   **[AutoRL: Zero-Data Training for Any Task](https://x.com/mattshumer_/status/1950572449025650733)** Train custom AI models without labeled data using automatic input generation and RULER evaluation.
*   **[RULER: Easy Mode for RL Rewards](https://openpipe.ai/blog/ruler-easy-mode-for-rl-rewards)** Use RULER for automatic reward generation.
*   **[ART·E: How We Built an Email Research Agent That Beats o3](https://openpipe.ai/blog/art-e-mail-agent)** A Qwen 2.5 14B email agent outperforming OpenAI's o3.
*   **[ART Trainer: A New RL Trainer for Agents](https://openpipe.ai/blog/art-trainer)** Easily train LLM-based agents using GRPO.

[📖 See all blog posts →](https://openpipe.ai/blog)

## Why Choose ART?

*   **Easy integration:** ART allows you to train models through RL in existing applications.
*   **Train Anywhere:** Run the ART client locally or on any machine with GPU.
*   **Flexible Observability:** Integrations with hosted platforms, simplifying debugging.
*   **Intelligent Defaults:** Configurable training parameters and inference engine configurations with intelligent defaults.

## Architecture Overview

ART employs a client-server architecture to optimize your RL workflows:

### Client
*   The client interfaces your codebase with ART.
*   Send and receive data from the LLM.
*   The client is compatible with OpenAI-compatible clients.

### Server

*   Runs independently on any machine with a GPU
*   Responsible for running training, inference and the RL loop
*   Abstracts away the complexity of the inference and training portions of the RL loop.

### Training Loop

1.  **Inference:** The ART client executes an agentic workflow, routing completion requests to the server.
2.  **Trajectories:** During agent execution, each message is stored in a `Trajectory`.
3.  **Rewards:** After a rollout, the client assigns a reward based on LLM performance.
4.  **Training:** Trajectories are sent to the server for grouping, and the model is trained using GRPO.

## Supported Models

ART is designed to work with most vLLM/HuggingFace-transformers compatible causal language models, including those supported by [Unsloth](https://docs.unsloth.ai/get-started/all-our-models). Please let us know on [Discord](https://discord.gg/zbBHRUpwf4) or open an issue on [GitHub](https://github.com/openpipe/art/issues) if you encounter issues.

## Contribute

We welcome contributions! See our [CONTRIBUTING.md](CONTRIBUTING.md) file for details.

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

ART is built on the shoulders of the open-source RL community. Special thanks to the authors of:

*   [Unsloth](https://github.com/unslothai/unsloth)
*   [vLLM](https://github.com/vllm-project/vllm)
*   [trl](https://github.com/huggingface/trl)
*   [torchtune](https://github.com/pytorch/torchtune)
*   [SkyPilot](https://github.com/skypilot-org/skypilot)

Thank you to our partners for helping us test ART!

[Back to Top](#agent-reinforcement-trainer-art)
```
Key improvements and SEO optimization in this version:

*   **Strong Headline:**  Replaced "Agent Reinforcement Trainer" with "Agent Reinforcement Trainer (ART):..." for a more descriptive and keyword-rich title. Added a one-sentence hook.
*   **Keywords:** Integrated relevant keywords like "Reinforcement Learning," "LLMs," "agents," "training," "open source," "MCP," "GRPO," and specific model names.
*   **Clear Structure:** Used headings and subheadings to organize information.
*   **Bulleted Lists:**  Emphasized key features and benefits with bullet points.
*   **Actionable Content:** Includes installation instructions and code snippets.
*   **Context & Benefits:** Explains what ART *does* and why users should use it.
*   **Internal Linking:**  Added back-to-top anchor links for improved navigation.
*   **Call to Action:** Encourages users to explore notebooks, read blog posts, and contribute.
*   **Comprehensive:** Includes Installation instructions, an architecture overview, supported models and licensing.
*   **Emphasis on Benefits:** Highlights benefits like 'no labeled data', 'easy integration', and 'high performance'.
*   **SEO Friendly:** Optimized headings and content for searchability.

This revised README is more informative, user-friendly, and optimized for search engines, making it easier for potential users to understand, use, and contribute to the ART project.