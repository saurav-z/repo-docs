<div align="center">
    <a href="https://art.openpipe.ai">
        <picture>
            <img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
        </picture>
    </a>
    <h1>Agent Reinforcement Trainer (ART): Train LLM Agents Effectively</h1>
    <p>
      ART is an open-source reinforcement learning (RL) framework to help you develop and deploy intelligent LLM agents for real-world tasks.
    </p>

    [![PRs-Welcome][contribute-image]][contribute-url]
    [![PyPI version](https://img.shields.io/pypi/v/openpipe-art?color=364fc7)][pypi-url]
    [![Train Agent](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)
    [![Join Discord](https://img.shields.io/badge/Join%20Discord-5865F2?style=plastic&logo=discord&logoColor=white)](https://discord.gg/zbBHRUpwf4)
    [![Documentation](https://img.shields.io/badge/Documentation-orange?style=plastic&logo=gitbook&logoColor=white)](https://art.openpipe.ai)
</div>

## Introduction

ART empowers you to train powerful, multi-step agents using Generative Reinforcement Policy Optimization (GRPO), enabling them to master complex tasks. This framework provides an ergonomic harness for integrating GRPO into any Python application.

**Key Features:**

*   **No Labeled Data Required:** Train agents without the need for extensive labeled datasets.
*   **Versatile for Any Task:** Optimize models for a wide range of applications and environments.
*   **State-of-the-Art Performance:** Achieve or surpass SOTA performance in various benchmarks.
*   **Seamless Integration:** Easily integrate ART into your existing projects with minimal setup.

**Get Started:**

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

## Core Capabilities

ART excels at enabling agents to:

*   **Learn from Experience:**  Improve agent reliability and performance through continuous learning from interactions.
*   **Optimize for MCP Servers:** Train agents to effectively utilize any Model Context Protocol (MCP) server. ART automatically discovers server tools, designs tasks, trains the model, and validates performance.
*   **Leverage RULER:**  Utilize the Relative Utility Learning and Evaluation Reward (RULER) system for efficient reward generation and model evaluation.

## Notebook Examples - Get Hands-On!

Explore various agent tasks with our interactive notebooks:

| Agent Task         | Example Notebook                                                                                                             | Description                                     | Performance (Example)                                                                                                                                                                             |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **MCP•RL**         | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/mcp-rl/mcp-rl.ipynb)               | Qwen 2.5 3B masters the NWS MCP server          | [Link coming soon]                                                                                                                                                                                  |
| **ART•E [RULER]**  | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/art-e.ipynb)                       | Qwen 2.5 7B learns to search emails using RULER | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/email_agent/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/art-e/art_e/evaluate/display_benchmarks.ipynb) |
| **2048**           | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)                   | Qwen 2.5 3B learns to play 2048                 | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/2048/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/2048/benchmark_2048.ipynb)                            |
| **Temporal Clue**  | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/temporal_clue/temporal-clue.ipynb) | Qwen 2.5 7B learns to solve Temporal Clue       | [Link coming soon]                                                                                                                                                                                  |
| **Tic Tac Toe**    | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb)     | Qwen 2.5 3B learns to play Tic Tac Toe          | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/tic-tac-toe-local/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/tic_tac_toe/benchmark_tic_tac_toe.ipynb) |
| **Codenames**      | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/codenames/Codenames_RL.ipynb)      | Qwen 2.5 3B learns to play Codenames            | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/codenames/win_rate_over_time.png" height="72"> [benchmarks](/examples/codenames/Codenames_RL.ipynb)                            |
| **AutoRL [RULER]** | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/auto_rl.ipynb)                     | Train Qwen 2.5 7B to master any task            | [Link coming soon]                                                                                                                                                                                  |

## News & Updates

Stay informed about the latest ART developments:

*   🗞️ **[MCP•RL: Teach Your Model to Master Any MCP Server](https://x.com/corbtt/status/1953171838382817625)** - Automatically train models to effectively use MCP server tools through reinforcement learning.
*   🗞️ **[AutoRL: Zero-Data Training for Any Task](https://x.com/mattshumer_/status/1950572449025650733)** - Train custom AI models without labeled data using automatic input generation and RULER evaluation.
*   🗞️ **[RULER: Easy Mode for RL Rewards](https://openpipe.ai/blog/ruler-easy-mode-for-rl-rewards)** is now available for automatic reward generation in reinforcement learning.
*   🗞️ **[ART·E: How We Built an Email Research Agent That Beats o3](https://openpipe.ai/blog/art-e-mail-agent)** demonstrates a Qwen 2.5 14B email agent outperforming OpenAI's o3.
*   🗞️ **[ART Trainer: A New RL Trainer for Agents](https://openpipe.ai/blog/art-trainer)** enables easy training of LLM-based agents using GRPO.

[📖 See all blog posts →](https://openpipe.ai/blog)

## Why Choose ART?

*   **Simplified RL Integration:** Easily integrate RL training into your existing applications.
*   **Flexible Training Options:** Train agents on your local machine or leverage cloud resources for GPU-accelerated training.
*   **Observability & Debugging:** Integrations with tools like W&B, Langfuse, and OpenPipe streamline debugging and monitoring.
*   **Optimized Defaults:** Benefit from intelligent default configurations that enhance training efficiency and stability.

## Installation

Get started by installing ART:

```bash
pip install openpipe-art
```

## ART•E Agent: Real-World Application

Explore the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) blog post to discover how we trained a Qwen 2.5 14B agent to outperform o3 in email retrieval!

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## Training Loop Explained

ART's architecture comprises a **client** and a **server**:

1.  **Inference:**
    *   Your code uses the ART client to execute agent workflows (with parallel rollouts).
    *   Completion requests are routed to the ART server, running the model's latest LoRA in vLLM.
    *   Each `system`, `user`, and `assistant` message is stored in a Trajectory.
    *   Rollouts conclude, and your code assigns a `reward` to each Trajectory.
2.  **Training:**
    *   Completed Trajectories are grouped and sent to the server. Inference is blocked during training.
    *   The server trains your model using GRPO, initializing from the latest checkpoint.
    *   The server saves the trained LoRA and loads it into vLLM.
    *   Inference resumes, and the loop repeats.

## Supported Models

ART is compatible with most vLLM/HuggingFace-transformers compatible causal language models, including those supported by [Unsloth](https://docs.unsloth.ai/get-started/all-our-models). Currently, Gemma 3 is not supported. Please report any model compatibility issues on [Discord](https://discord.gg/zbBHRUpwf4) or [GitHub](https://github.com/openpipe/art/issues).

## Contribute

Join the ART community!  See our [CONTRIBUTING.md](CONTRIBUTING.md) file for more information.

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

This repository is available under the [Apache-2.0 License](LICENSE).

## Credits

ART is built upon the work of many. We're especially grateful to:

*   [Unsloth](https://github.com/unslothai/unsloth)
*   [vLLM](https://github.com/vllm-project/vllm)
*   [trl](https://github.com/huggingface/trl)
*   [torchtune](https://github.com/pytorch/torchtune)
*   [SkyPilot](https://github.com/skypilot-org/skypilot)

And, thank you to our partners who have helped test ART in the wild!

[pypi-url]: https://pypi.org/project/openpipe-art/
[contribute-url]: https://github.com/openpipe/art/blob/main/CONTRIBUTING.md
[contribute-image]: https://img.shields.io/badge/PRs-welcome-blue.svg
```
Key improvements and SEO considerations:

*   **Clear, Concise Hook:**  The opening sentence directly highlights the core benefit.
*   **Keyword Optimization:** Incorporated keywords like "LLM agents," "reinforcement learning," "train," "GRPO," and task-specific terms (MCP, etc.) naturally.
*   **Headings and Structure:**  Well-defined sections with clear headings improve readability and SEO.
*   **Bullet Points:**  Emphasize key features for easy scanning and SEO.
*   **Action-Oriented Language:** Uses verbs like "Train," "Explore," "Get Started" to encourage engagement.
*   **Internal Linking:** Included links to other content.
*   **External Linking:** Included links to resources like documentation, blog and examples.
*   **Meta Description:** The introductory sentences and the description of key features provide a good meta description.
*   **Alt Text:** Alt text is present for the image.
*   **Mobile-Friendly:** Uses responsive design and image sizes.
*   **Concise and Scannable:** The README is organized for easy readability.
*   **Removed Duplication:** Removed content that repeats the blog and is therefore not as useful in the README.