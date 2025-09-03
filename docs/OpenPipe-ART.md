<div align="center">

<a href="https://art.openpipe.ai"><picture>
<img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
</picture></a>

<p align="center">
  <h1>Agent Reinforcement Trainer (ART)</h1>
</p>
</div>

**Train AI agents for complex tasks with ART, a powerful open-source reinforcement learning framework.** Learn more and contribute at the [original repo](https://github.com/OpenPipe/ART).

[![PRs-Welcome](https://img.shields.io/badge/PRs-welcome-blue.svg)][contribute-url]
[![PyPI version](https://img.shields.io/pypi/v/openpipe-art?color=364fc7)][pypi-url]
[![Train Agent](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/2048/2048.ipynb)
[![Join Discord](https://img.shields.io/badge/Join%20Discord-5865F2?style=plastic&logo=discord&logoColor=white)](https://discord.gg/zbBHRUpwf4)
[![Documentation](https://img.shields.io/badge/Documentation-orange?style=plastic&logo=gitbook&logoColor=white)](https://art.openpipe.ai)

## Key Features

*   **Zero-Shot Reward Engineering with RULER:** Use LLMs to automatically score agent trajectories, eliminating the need for hand-crafted reward functions.
*   **Accelerated Development:** Develop agents 2-3x faster by skipping manual reward function creation.
*   **General-Purpose Application:** ART works across various tasks without modifications.
*   **High Performance:** Achieve results that match or surpass hand-crafted rewards in many benchmarks.
*   **Easy Integration:** Seamlessly replace manual reward functions with ART's solutions.
*   **Modular Architecture:** ART separates the training server from your codebase, improving maintainability.
*   **Flexible Training Options:** Train on your laptop or leverage GPU-enabled environments.
*   **Integrations:** Benefit from integrations with platforms like W&B, Langfuse, and OpenPipe.
*   **Customizable & Optimized Defaults:** Configure training parameters to meet specific requirements or utilize efficient default settings.

## RULER: Revolutionize Agent Reward Scoring

**RULER (Relative Universal LLM-Elicited Rewards)** simplifies agent training by using an LLM as a judge to automatically score agent trajectories, removing the need for manual reward functions, labeled data, or expert feedback.

```python
# Before: Hours of reward engineering
def complex_reward_function(trajectory):
    # 50+ lines of careful scoring logic...
    pass

# After: One line with RULER
judged_group = await ruler_score_group(group, "openai/o3")
```

[📖 Learn more about RULER →](https://art.openpipe.ai/fundamentals/ruler)

## ART: Train Multi-Step Agents with GRPO

ART is an open-source RL framework designed to enhance agent reliability by enabling LLMs to **learn from experience** using GRPO. ART provides a streamlined framework for integrating GRPO into any Python application.

## Example Notebooks

Explore ART's capabilities with these interactive notebooks:

| Agent Task          | Example Notebook                                                                                                                       | Description                                         | Comparative Performance                                                                                                                                                                                                     |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ART•E LangGraph** | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/langgraph/art-e-langgraph.ipynb)   | Qwen 2.5 7B learns to search emails using LangGraph | [Link coming soon]                                                                                                                                                                                                          |
| **MCP•RL**          | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/mcp-rl/mcp-rl.ipynb)               | Qwen 2.5 3B masters the NWS MCP server              | [Link coming soon]                                                                                                                                                                                                          |
| **ART•E [RULER]**   | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/art-e.ipynb)                       | Qwen 2.5 7B learns to search emails using RULER     | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/email_agent/accuracy-training-progress.svg" height="72"> [benchmarks](/dev/art-e/art_e/evaluate/display_benchmarks.ipynb)                              |
| **2048**            | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/2048/2048.ipynb)                   | Qwen 2.5 3B learns to play 2048                     | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/2048/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/2048/display_benchmarks.ipynb)                                                |
| **Temporal Clue**   | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/temporal_clue/temporal-clue.ipynb) | Qwen 2.5 7B learns to solve Temporal Clue           | [Link coming soon]                                                                                                                                                                                                          |
| **Tic Tac Toe**     | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb)     | Qwen 2.5 3B learns to play Tic Tac Toe              | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/tic-tac-toe-local/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/tic_tac_toe/display-benchmarks.ipynb)                            |
| **Codenames**       | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/codenames/Codenames_RL.ipynb)      | Qwen 2.5 3B learns to play Codenames                | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/codenames/win_rate_over_time.png" height="72"> [benchmarks](https://github.com/OpenPipe/art-notebooks/blob/main/examples/codenames/Codenames_RL.ipynb) |
| **AutoRL [RULER]**  | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/auto_rl.ipynb)                     | Train Qwen 2.5 7B to master any task                | [Link coming soon]                                                                                                                                                                                                          |

## ART News

Stay updated with the latest advancements and research from ART:

*   🗞️ **[ART now integrates seamlessly with LangGraph](https://art.openpipe.ai/integrations/langgraph-integration)** - Train your LangGraph agents with reinforcement learning for smarter multi-step reasoning and improved tool usage.
*   🗞️ **[MCP•RL: Teach Your Model to Master Any MCP Server](https://x.com/corbtt/status/1953171838382817625)** - Automatically train models to effectively use MCP server tools through reinforcement learning.
*   🗞️ **[AutoRL: Zero-Data Training for Any Task](https://x.com/mattshumer_/status/1950572449025650733)** - Train custom AI models without labeled data using automatic input generation and RULER evaluation.
*   🗞️ **[RULER: Easy Mode for RL Rewards](https://openpipe.ai/blog/ruler-easy-mode-for-rl-rewards)** is now available for automatic reward generation in reinforcement learning.
*   🗞️ **[ART·E: How We Built an Email Research Agent That Beats o3](https://openpipe.ai/blog/art-e-mail-agent)** demonstrates a Qwen 2.5 14B email agent outperforming OpenAI's o3.
*   🗞️ **[ART Trainer: A New RL Trainer for Agents](https://openpipe.ai/blog/art-trainer)** enables easy training of LLM-based agents using GRPO.

[📖 See all blog posts →](https://openpipe.ai/blog)

## Why Choose ART?

ART simplifies introducing RL training into existing applications by providing an ergonomic harness. ART offers:

*   **Versatile Training:** Train agents locally or in ephemeral GPU environments.
*   **Observability and Debugging:** Integrations with popular platforms like W&B, Langfuse, and OpenPipe.
*   **Customization with Defaults:**  Configure training parameters to meet specific needs, or leverage optimized default configurations.

## Installation

Install ART to your existing project with the following command:

```bash
pip install openpipe-art
```

## 🤖 ART•E Agent: Real-World Application

Learn how to use ART for real-world tasks by reading the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) blog post. It details training Qwen 2.5 14B to beat o3 at email retrieval.

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## 🔁 Training Loop Explained

ART's functionality is split into a **client** and a **server** to manage the RL process:

1.  **Inference:**
    *   Your code uses the ART client to perform agentic workflows (e.g., parallel rollouts).
    *   Completion requests are routed to the ART server.
    *   Each message is stored as a Trajectory.
    *   A `reward` is assigned to each Trajectory upon completion.
2.  **Training:**
    *   Trajectories are grouped and sent to the server.
    *   The server trains your model using GRPO.
    *   The server saves and loads the newly trained LoRA.
    *   Inference resumes.

The loop continues until the desired training iterations are complete.

## 🧩 Supported Models

ART is compatible with most vLLM/HuggingFace-transformers causal language models (e.g., those supported by [Unsloth](https://docs.unsloth.ai/get-started/all-our-models)). If a model is not working, please reach out on [Discord](https://discord.gg/zbBHRUpwf4) or open an issue on [GitHub](https://github.com/openpipe/art/issues).

## 🤝 Contributing

ART welcomes contributions! See the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidance.

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

The source code is licensed under the [Apache-2.0 License](LICENSE).

## 🙏 Credits

ART relies on the work of many in the open-source RL community. We're especially grateful to the authors of the following projects:

-   [Unsloth](https://github.com/unslothai/unsloth)
-   [vLLM](https://github.com/vllm-project/vllm)
-   [trl](https://github.com/huggingface/trl)
-   [torchtune](https://github.com/pytorch/torchtune)
-   [SkyPilot](https://github.com/skypilot-org/skypilot)

We thank our partners for helping us test ART.

[pypi-url]: https://pypi.org/project/openpipe-art/
[contribute-url]: https://github.com/openpipe/art/blob/main/CONTRIBUTING.md
[contribute-image]: https://img.shields.io/badge/PRs-welcome-blue.svg