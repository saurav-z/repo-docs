<div align="center">

<a href="https://art.openpipe.ai"><picture>
<img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
</picture></a>

<p align="center">
  <h1>Agent Reinforcement Trainer (ART)</h1>
</p>

<p>
  **Unlock the power of reinforcement learning for your LLM agents with ART, an open-source framework that simplifies agent training and improves performance.**
</p>

[![PRs-Welcome][contribute-image]][contribute-url]
[![PyPI version](https://img.shields.io/pypi/v/openpipe-art?color=364fc7)][pypi-url]
[![Train Agent](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/2048/2048.ipynb)

[![Join Discord](https://img.shields.io/badge/Join%20Discord-5865F2?style=plastic&logo=discord&logoColor=white)](https://discord.gg/zbBHRUpwf4)
[![Documentation](https://img.shields.io/badge/Documentation-orange?style=plastic&logo=gitbook&logoColor=white)](https://art.openpipe.ai)

</div>

## Agent Reinforcement Trainer (ART): Train LLM Agents with Ease

ART is an open-source framework designed to streamline the training of multi-step LLM agents using GRPO.  **Stop manually engineering reward functions and effortlessly train powerful AI agents with ART's innovative features, like zero-shot reward generation.**

**Key Features:**

*   **Zero-Shot Reward Engineering with RULER:** Automate reward generation using LLMs, eliminating the need for hand-crafted reward functions.
*   **Faster Development:**  Significantly reduce development time (2-3x faster) by skipping reward function engineering.
*   **Versatile Application:** Train agents for any task without code modification.
*   **Superior Performance:** Achieve performance that matches or exceeds hand-crafted rewards in many benchmarks.
*   **Simple Integration:**  Seamlessly replace manual reward functions with ART's intuitive interface.
*   **GRPO Integration:** Take advantage of Generalized Reward Policy Optimization (GRPO).
*   **Modular Client-Server Architecture:** Train from anywhere without managing infrastructure.
*   **Built-in Integrations:** Out-of-the-box integrations with tools like W&B, Langfuse, and OpenPipe
*   **Customization Options:** Configure training parameters and inference settings to fit your needs.
*   **Broad Model Compatibility:** Compatible with most vLLM/HuggingFace-transformers compatible causal language models, including Qwen 2.5.

```python
# Before: Time-consuming reward engineering
def complex_reward_function(trajectory):
    # 50+ lines of careful scoring logic...
    pass

# After: Effortless reward generation with RULER
judged_group = await ruler_score_group(group, "openai/o3")
```

[📖 Learn more about RULER](https://art.openpipe.ai/fundamentals/ruler)

## ART Overview

ART simplifies Reinforcement Learning (RL) by providing an ergonomic harness for integrating GRPO into any Python application.  Easily train your agents and improve their reliability and performance. Run through our quickstart notebooks or read our documentation to get started.

## Training Agent Examples (Notebooks)

Explore practical examples and benchmark results to see how ART empowers you to train agents for diverse tasks:

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

## ART News and Updates

Stay informed about the latest advancements in ART and agent training:

*   🗞️ **[ART Integrates with LangGraph](https://art.openpipe.ai/integrations/langgraph-integration)** - Train your LangGraph agents with RL for improved multi-step reasoning.
*   🗞️ **[MCP•RL: Master Any MCP Server](https://x.com/corbtt/status/1953171838382817625)** - Automate the training of models using MCP server tools.
*   🗞️ **[AutoRL: Zero-Data Training for Any Task](https://x.com/mattshumer_/status/1950572449025650733)** - Train custom AI models without labeled data using automatic input generation and RULER evaluation.
*   🗞️ **[RULER: Easy Mode for RL Rewards](https://openpipe.ai/blog/ruler-easy-mode-for-rl-rewards)** is now available for automatic reward generation in reinforcement learning.
*   🗞️ **[ART·E: How We Built an Email Research Agent That Beats o3](https://openpipe.ai/blog/art-e-mail-agent)** demonstrates a Qwen 2.5 14B email agent outperforming OpenAI's o3.
*   🗞️ **[ART Trainer: A New RL Trainer for Agents](https://openpipe.ai/blog/art-trainer)** enables easy training of LLM-based agents using GRPO.

[📖 See all blog posts](https://openpipe.ai/blog)

## Why Choose ART?

ART provides a suite of features designed to make RL accessible and effective:

*   **Modular Client-Server Architecture:** Train from any location, eliminating the need to manage infrastructure.
*   **Flexible Training:** Run the ART client locally or on a GPU-enabled environment.
*   **Enhanced Observability:** Integrate with W&B, Langfuse, and OpenPipe for better debugging and monitoring.
*   **Intelligent Defaults:** Utilize optimized defaults, or customize training parameters to meet your specific needs.

## Installation

Get started quickly by installing the ART package in your Python project:

```bash
pip install openpipe-art
```

## 🤖 ART•E Agent

Explore the real-world application of ART.  Read the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) blog post to learn how we trained Qwen 2.5 14B to outperform o3 in email retrieval.

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## 🔁 Training Loop Explained

ART employs a client-server architecture to manage the RL training loop:

1.  **Inference:**
    1.  Your code uses the ART client to perform an agentic workflow.
    2.  Completion requests are routed to the ART server, which runs the model's latest LoRA in vLLM.
    3.  Each message is stored in a Trajectory.
    4.  A `reward` is assigned to each Trajectory.
2.  **Training:**
    1.  Trajectories are grouped and sent to the server.
    2.  The server trains your model using GRPO, starting from the latest checkpoint.
    3.  The server saves the LoRA and loads it into vLLM.
    4.  The loop repeats.

This cycle continues until the specified number of iterations is reached.

## 🧩 Supported Models

ART supports most vLLM/HuggingFace-transformers compatible causal language models.  Refer to [Unsloth](https://docs.unsloth.ai/get-started/all-our-models) for a list of compatible models. If you have any issues, please reach out on [Discord](https://discord.gg/zbBHRUpwf4) or open a GitHub issue!

## 🤝 Contributing

Join the ART community! Contributions are highly encouraged. See [CONTRIBUTING.md](https://github.com/OpenPipe/ART/blob/main/CONTRIBUTING.md) for guidelines.

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

This project is licensed under the [Apache-2.0 License](LICENSE).

## 🙏 Credits

ART is built on the work of many. We are grateful to the open-source RL community and especially to the developers of these projects:

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
Key improvements and changes:

*   **SEO-optimized title and introduction:** Using relevant keywords like "Agent Reinforcement Trainer", "LLM agents", and "reinforcement learning" in the title and first paragraph to improve search engine visibility.  A strong value proposition as the first line.
*   **Clear value proposition:**  Highlights the core benefit "unlock the power of reinforcement learning for your LLM agents".
*   **Concise bullet points:** Summarized key features to make them easier to read.
*   **More descriptive headings:**  Improved heading titles for better organization and readability.
*   **Emphasis on benefits:**  Focus on the advantages for the user (e.g., faster development, versatile application).
*   **Stronger call to action:**  Encourages users to explore notebooks and documentation.
*   **Keywords:** Included important keywords to optimize for search engines (e.g., "LLM agents," "reinforcement learning," "GRPO", "RULER").
*   **Updated Training Examples Section:** The formatting is now cleaner, with a descriptive table.
*   **Concise Summaries:** More concise and clear explanations throughout, making the README more scannable.
*   **Removed redundant information:** Streamlined the content to focus on key aspects of the project.
*   **Improved Formatting:** Enhanced overall formatting for better readability, including bolding, and spacing.
*   **Added link to the original repo:** Ensured that the user has a quick link back to the original repo.
*   **Improved Model Support:** Added specific detail on model support to give the user more info.