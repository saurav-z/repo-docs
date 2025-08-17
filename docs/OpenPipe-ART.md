<div align="center">

<a href="https://art.openpipe.ai"><picture>
<img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
</picture></a>

<p align="center">
  <h1>Agent Reinforcement Trainer (ART)</h1>
</p>
</div>

**Supercharge your LLMs with ART: a powerful open-source framework for training multi-step agents using Reinforcement Learning!** ([View on GitHub](https://github.com/OpenPipe/ART))

## Key Features of ART

*   **Train Agents for Real-World Tasks:** ART enables you to train multi-step agents for complex tasks.
*   **GRPO-Based Training:** Leverages GRPO (Guided Policy Optimization) for efficient agent learning.
*   **Open-Source and Customizable:** Benefit from a flexible and open-source RL framework.
*   **Easy Integration:** Integrate ART into your existing Python applications with minimal setup.
*   **No Labeled Data Needed (for some applications):**  MCP•RL can learn from server tools, eliminating the need for labeled data.
*   **Pre-built examples:** Includes notebooks for MCP•RL, ART•E, 2048, Temporal Clue, Tic Tac Toe, Codenames, and AutoRL.
*   **Active Development & Community:**  Stay updated with the latest research and updates.
*   **Supports Many Models:** Designed to work with vLLM/HuggingFace-transformers compatible causal language models, and more.
*   **Flexible Training Loop:** Integrates a client and server architecture for flexible model training and evaluation.

## Getting Started with ART

ART provides an ergonomic harness for integrating GRPO into any python application.

```bash
pip install openpipe-art
```

### Example Usage: MCP•RL (Model Context Protocol Reinforcement Learning)

Learn how to train an agent to master any MCP (Model Context Protocol) server, without any labeled data. Just provide the server URL!

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

## Example Notebooks and Benchmarks

Explore pre-built notebooks to see ART in action:

| Agent Task         | Example Notebook                                                                                                             | Description                                     | Comparative Performance                                                                                                                                                                             |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **MCP•RL**         | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/mcp-rl/mcp-rl.ipynb)               | Qwen 2.5 3B masters the NWS MCP server          | [Link coming soon]                                                                                                                                                                                  |
| **ART•E [RULER]**  | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/art-e.ipynb)                       | Qwen 2.5 7B learns to search emails using RULER | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/email_agent/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/art-e/art_e/evaluate/display_benchmarks.ipynb) |
| **2048**           | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)                   | Qwen 2.5 3B learns to play 2048                 | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/2048/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/2048/benchmark_2048.ipynb)                            |
| **Temporal Clue**  | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/temporal_clue/temporal-clue.ipynb) | Qwen 2.5 7B learns to solve Temporal Clue       | [Link coming soon]                                                                                                                                                                                  |
| **Tic Tac Toe**    | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb)     | Qwen 2.5 3B learns to play Tic Tac Toe          | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/tic-tac-toe-local/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/tic_tac_toe/benchmark_tic_tac_toe.ipynb) |
| **Codenames**      | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/codenames/Codenames_RL.ipynb)      | Qwen 2.5 3B learns to play Codenames            | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/codenames/win_rate_over_time.png" height="72"> [benchmarks](/examples/codenames/Codenames_RL.ipynb)                            |
| **AutoRL [RULER]** | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/auto_rl.ipynb)                     | Train Qwen 2.5 7B to master any task            | [Link coming soon]                                                                                                                                                                                  |

## ART Training Loop Overview

ART's functionality is divided into a **client** and a **server**, allowing for a flexible training process:

1.  **Inference**: Your code uses the ART client to run agentic workflows. Completion requests route to the ART server, which runs the model.
2.  **Training**: Trajectories are grouped and sent to the server. The server trains your model using GRPO.

## 🤖 ART•E Agent

Learn more about how ART was used to train a Qwen 2.5 14B agent to beat o3 at email retrieval in the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) blog post!

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## 📰 ART News

Stay informed about the latest developments in ART and agent training:

*   🗞️ **[MCP•RL: Teach Your Model to Master Any MCP Server](https://x.com/corbtt/status/1953171838382817625)** - Automatically train models to effectively use MCP server tools through reinforcement learning.
*   🗞️ **[AutoRL: Zero-Data Training for Any Task](https://x.com/mattshumer_/status/1950572449025650733)** - Train custom AI models without labeled data using automatic input generation and RULER evaluation.
*   🗞️ **[RULER: Easy Mode for RL Rewards](https://openpipe.ai/blog/ruler-easy-mode-for-rl-rewards)** is now available for automatic reward generation in reinforcement learning.
*   🗞️ **[ART·E: How We Built an Email Research Agent That Beats o3](https://openpipe.ai/blog/art-e-mail-agent)** demonstrates a Qwen 2.5 14B email agent outperforming OpenAI's o3.
*   🗞️ **[ART Trainer: A New RL Trainer for Agents](https://openpipe.ai/blog/art-trainer)** enables easy training of LLM-based agents using GRPO.

[📖 See all blog posts →](https://openpipe.ai/blog)

## Why Choose ART?

*   **Simplified RL Integration:** ART provides convenient wrappers for incorporating RL training into your existing applications.
*   **Flexible Deployment:** Run the ART client on your laptop or a local GPU.
*   **Easy Debugging:** Integrations with hosted platforms such as W\&B, Langfuse, and OpenPipe offer flexible observability.
*   **Customizable and Efficient:** Configure training parameters and inference engine configurations to meet specific needs or leverage optimized defaults.

## 🤝 Contributing

ART is actively being developed and welcomes contributions!  Check out the [CONTRIBUTING.md](CONTRIBUTING.md) file for details.

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

ART is built upon the contributions of the open-source RL community.  We are especially grateful to the authors of the following projects:

*   [Unsloth](https://github.com/unslothai/unsloth)
*   [vLLM](https://github.com/vllm-project/vllm)
*   [trl](https://github.com/huggingface/trl)
*   [torchtune](https://github.com/pytorch/torchtune)
*   [SkyPilot](https://github.com/skypilot-org/skypilot)

Thank you to our partners who have helped us test ART in the wild!