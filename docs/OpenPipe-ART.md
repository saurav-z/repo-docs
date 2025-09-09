<div align="center">

<a href="https://art.openpipe.ai"><picture>
<img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
</picture></a>

<p align="center">
  <h1>Agent Reinforcement Trainer (ART)</h1>
</p>

<p>
  <b>Supercharge your LLM agents: Train multi-step agents for real-world tasks efficiently with Agent Reinforcement Trainer (ART) and the power of Generative Reinforcement Policy Optimization (GRPO).</b>
</p>

[![PRs-Welcome][contribute-image]][contribute-url]
[![PyPI version](https://img.shields.io/pypi/v/openpipe-art?color=364fc7)][pypi-url]
[![Train Agent](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/2048/2048.ipynb)

[![Join Discord](https://img.shields.io/badge/Join%20Discord-5865F2?style=plastic&logo=discord&logoColor=white)](https://discord.gg/zbBHRUpwf4)
[![Documentation](https://img.shields.io/badge/Documentation-orange?style=plastic&logo=gitbook&logoColor=white)](https://art.openpipe.ai)

</div>

## Key Features of ART

*   **Effortless Agent Training:** Quickly train LLM agents without complex reward engineering.
*   **Zero-Shot Reward with RULER:** Leverages LLMs for automatic reward scoring, eliminating the need for hand-crafted reward functions.
*   **GRPO Integration:** Employs Generative Reinforcement Policy Optimization for improved agent reliability and performance.
*   **Open Source and Customizable:** Benefit from an open-source framework with intelligent defaults and flexible configuration options.
*   **Seamless Integration:** Easy to integrate into existing Python applications.
*   **Cloud-Agnostic Training:** Train agents from anywhere, utilizing GPU-enabled environments on various platforms.
*   **Comprehensive Benchmarks & Examples:** Get started quickly with several example notebooks covering a range of tasks, including email search, 2048, and more.
*   **Flexible Observability:** Integrates with platforms like Weights & Biases, Langfuse, and OpenPipe for simplified debugging.

## RULER: Zero-Shot Agent Rewards

**RULER** (Relative Universal LLM-Elicited Rewards) revolutionizes agent training by using LLMs as judges to automatically score agent trajectories, eliminating the need for manual reward functions. Just define your task in the system prompt, and RULER handles the rest.

**Key Benefits of RULER:**

*   **Faster Development:** Reduce development time by skipping reward function engineering.
*   **General-Purpose:** Works seamlessly across various tasks without modifications.
*   **High Performance:** Achieves or surpasses hand-crafted rewards in several benchmarks.
*   **Simple Integration:** Easily integrates as a drop-in replacement for manual reward functions.

```python
# Before: Reward engineering
def complex_reward_function(trajectory):
    # 50+ lines of careful scoring logic...
    pass

# After: One line with RULER
judged_group = await ruler_score_group(group, "openai/o3")
```

[📖 Learn more about RULER →](https://art.openpipe.ai/fundamentals/ruler)

## ART Overview

ART is an open-source framework designed to enhance agent reliability by enabling LLMs to learn from experience, using GRPO. This robust framework provides a streamlined harness for incorporating GRPO into any Python application.

## Example Notebooks

Explore the following example notebooks to get started quickly:

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

## 📰 ART News

Stay updated with the latest advancements and research from ART:

*   🗞️ **[ART now integrates seamlessly with LangGraph](https://art.openpipe.ai/integrations/langgraph-integration)** - Train your LangGraph agents with reinforcement learning for smarter multi-step reasoning and improved tool usage.
*   🗞️ **[MCP•RL: Teach Your Model to Master Any MCP Server](https://x.com/corbtt/status/1953171838382817625)** - Automatically train models to effectively use MCP server tools through reinforcement learning.
*   🗞️ **[AutoRL: Zero-Data Training for Any Task](https://x.com/mattshumer_/status/1950572449025650733)** - Train custom AI models without labeled data using automatic input generation and RULER evaluation.
*   🗞️ **[RULER: Easy Mode for RL Rewards](https://openpipe.ai/blog/ruler-easy-mode-for-rl-rewards)** is now available for automatic reward generation in reinforcement learning.
*   🗞️ **[ART·E: How We Built an Email Research Agent That Beats o3](https://openpipe.ai/blog/art-e-mail-agent)** demonstrates a Qwen 2.5 14B email agent outperforming OpenAI's o3.
*   🗞️ **[ART Trainer: A New RL Trainer for Agents](https://openpipe.ai/blog/art-trainer)** enables easy training of LLM-based agents using GRPO.

[📖 See all blog posts →](https://openpipe.ai/blog)

## Why Choose ART?

*   **Ease of Integration:** ART simplifies the process of integrating RL training into your existing applications.
*   **Flexible Training:** Train agents on local machines or utilize ephemeral GPU-enabled environments.
*   **Improved Debugging & Observability:** Integrations with popular platforms like W&B and Langfuse provide enhanced monitoring and debugging capabilities.
*   **Optimized Defaults:** Benefit from intelligent defaults designed for training efficiency and stability.

## Installation

To add ART to your project, simply run:

```bash
pip install openpipe-art
```

## 🤖 ART•E Agent

Explore a real-world application of ART by reviewing the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) blog post, which details the training of Qwen 2.5 14B to outperform o3 in email retrieval.

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## 🔁 Training Loop Overview

ART leverages a client-server architecture. The client handles interactions with your codebase, while the server manages the inference and training aspects of the RL loop.

1.  **Inference:**
    *   Your code uses the ART client to perform an agentic workflow.
    *   Completion requests are routed to the ART server, running the model's latest LoRA in vLLM.
    *   Each message is stored in a Trajectory.
    *   When a rollout finishes, your code assigns a `reward`.

2.  **Training:**
    *   Trajectories are sent to the server.
    *   The server trains your model using GRPO.
    *   The server saves the trained LoRA.
    *   Inference resumes.

This loop continues until the specified number of iterations is complete.

## 🧩 Supported Models

ART supports most vLLM/HuggingFace-transformers compatible causal language models, including those compatible with [Unsloth](https://docs.unsloth.ai/get-started/all-our-models). If you encounter issues with a specific model, please reach out on [Discord](https://discord.gg/zbBHRUpwf4) or open an issue on [GitHub](https://github.com/openpipe/art/issues).

## 🤝 Contributing

Contributions to ART are welcome! Please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines.

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
}
```

## ⚖️ License

ART is licensed under the [Apache-2.0 License](LICENSE).

## 🙏 Credits

ART is built upon the work of many. We are especially grateful to the following projects:

*   [Unsloth](https://github.com/unslothai/unsloth)
*   [vLLM](https://github.com/vllm-project/vllm)
*   [trl](https://github.com/huggingface/trl)
*   [torchtune](https://github.com/pytorch/torchtune)
*   [SkyPilot](https://github.com/skypilot-org/skypilot)

Thank you to our partners for helping us test ART!

[pypi-url]: https://pypi.org/project/openpipe-art/
[contribute-url]: https://github.com/openpipe/art/blob/main/CONTRIBUTING.md
[contribute-image]: https://img.shields.io/badge/PRs-welcome-blue.svg
```

Key improvements and explanations:

*   **SEO Optimization:**
    *   Uses keywords like "Agent Reinforcement Trainer," "LLM agents," "GRPO," and "zero-shot rewards" throughout.
    *   Includes a clear title and a concise, engaging one-sentence hook at the beginning.
    *   Uses headings and subheadings to organize content, making it easier for search engines to understand the structure.
*   **Improved Readability and Structure:**
    *   Uses bullet points to highlight key features.
    *   Breaks down complex information into digestible chunks.
    *   Provides clear explanations of concepts like RULER and the training loop.
*   **Conciseness:**
    *   Removes redundant phrases and focuses on the most important information.
*   **Call to Action:**
    *   Encourages users to explore example notebooks and documentation.
*   **Emphasis on Benefits:**
    *   Highlights the key advantages of using ART and RULER.
*   **Clear Formatting:**
    *   Maintains the original markdown formatting for easy use.
*   **Link Back:**
    *   Includes the original repo link to encourage more visitors.