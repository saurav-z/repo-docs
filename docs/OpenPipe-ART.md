<div align="center">

<a href="https://art.openpipe.ai"><picture>
<img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
</picture></a>

<p align="center">
  <h1>Agent Reinforcement Trainer (ART)</h1>
</p>

<p>
  Train advanced multi-step AI agents for complex tasks with GRPO.
</p>

[![PRs-Welcome][contribute-image]][contribute-url]
[![PyPI version](https://img.shields.io/pypi/v/openpipe-art?color=364fc7)][pypi-url]
[![Train Agent](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/2048/2048.ipynb)

[![Join Discord](https://img.shields.io/badge/Join%20Discord-5865F2?style=plastic&logo=discord&logoColor=white)](https://discord.gg/zbBHRUpwf4)
[![Documentation](https://img.shields.io/badge/Documentation-orange?style=plastic&logo=gitbook&logoColor=white)](https://art.openpipe.ai)

</div>

## 🚀 Agent Reinforcement Trainer (ART): Train LLM Agents Faster and Smarter

**ART empowers you to train powerful, reliable AI agents capable of handling complex, multi-step tasks using the GRPO reinforcement learning algorithm.** ART provides an accessible, ergonomic environment for integrating GRPO into your Python applications. Say goodbye to complex reward function engineering with ART's innovative features, including the revolutionary **RULER** system.

**Key Features of ART:**

*   **Zero-Shot Rewards with RULER:** Leverage LLMs to automatically score agent trajectories, eliminating the need for hand-crafted reward functions.
*   **Faster Development:** Accelerate your development process by up to 3x by skipping reward function engineering.
*   **General-Purpose Applicability:** Works seamlessly across diverse tasks without requiring modifications.
*   **Performance that Excels:** Achieve results that match or surpass hand-crafted rewards in many benchmarks.
*   **Easy Integration:** Seamlessly integrate ART into your existing projects with a simple drop-in replacement for manual reward functions.
*   **Modular Client/Server Architecture:** Train agents from anywhere, abstracting training complexity with a modular client and server structure.
*   **Customizable & Optimized:** Configure training parameters and leverage intelligent defaults for efficiency and stability.

## 🎯 RULER: Zero-Shot Agent Rewards

**RULER (Relative Universal LLM-Elicited Rewards)** simplifies agent training by eliminating the need for manual reward functions. By using an LLM-as-judge, RULER automatically scores agent trajectories, dramatically speeding up development.

```python
# Before: Hours of reward engineering
def complex_reward_function(trajectory):
    # 50+ lines of careful scoring logic...
    pass

# After: One line with RULER
judged_group = await ruler_score_group(group, "openai/o3")
```

[📖 Learn more about RULER →](https://art.openpipe.ai/fundamentals/ruler)

## 💻 ART: Training Your AI Agents

ART simplifies the complexities of RL training, offering an intuitive experience. ART's core is designed to integrate with your applications with minimal changes. For a hands-on introduction, check out the provided notebooks below or explore the full documentation to delve deeper.

## 📒 Example Notebooks: Get Started Quickly

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

## 📰 ART News & Updates

Stay up-to-date with the latest research and advancements in building state-of-the-art AI agents.

*   🗞️ **[ART now integrates seamlessly with LangGraph](https://art.openpipe.ai/integrations/langgraph-integration)** - Train your LangGraph agents with reinforcement learning for smarter multi-step reasoning and improved tool usage.
*   🗞️ **[MCP•RL: Teach Your Model to Master Any MCP Server](https://x.com/corbtt/status/1953171838382817625)** - Automatically train models to effectively use MCP server tools through reinforcement learning.
*   🗞️ **[AutoRL: Zero-Data Training for Any Task](https://x.com/mattshumer_/status/1950572449025650733)** - Train custom AI models without labeled data using automatic input generation and RULER evaluation.
*   🗞️ **[RULER: Easy Mode for RL Rewards](https://openpipe.ai/blog/ruler-easy-mode-for-rl-rewards)** is now available for automatic reward generation in reinforcement learning.
*   🗞️ **[ART·E: How We Built an Email Research Agent That Beats o3](https://openpipe.ai/blog/art-e-mail-agent)** demonstrates a Qwen 2.5 14B email agent outperforming OpenAI's o3.
*   🗞️ **[ART Trainer: A New RL Trainer for Agents](https://openpipe.ai/blog/art-trainer)** enables easy training of LLM-based agents using GRPO.

[📖 See all blog posts →](https://openpipe.ai/blog)

## ⚙️ Why Choose ART?

ART offers several advantages to empower your agent development:

*   **Simplified Integration:** Easily introduce RL training into your existing applications.
*   **Flexible Training:** Train agents on your local machine or leverage cloud-based GPU resources.
*   **Enhanced Observability:** Integrate with platforms like W&B, Langfuse, and OpenPipe for easy debugging.
*   **Optimized Defaults:** Take advantage of intelligent defaults that have been optimized for training efficiency and stability.

## ⚙️ Installation: Get Started Now!

Get started training your agents with ART today! Simply install the `openpipe-art` package:

```bash
pip install openpipe-art
```

## 🤖 ART•E Agent: Real-World Example

Explore a practical application of ART. Discover how the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) was trained to excel at email retrieval with a Qwen 2.5 14B model, outperforming OpenAI's o3.

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## 🔄 Training Loop: How ART Works

ART's functionality is divided into a client and a server:

1.  **Inference:**
    *   Your code uses the ART client to perform agent workflows.
    *   Completion requests are routed to the ART server.
    *   Each message is stored in a Trajectory.
    *   When a rollout finishes, your code assigns a `reward`.
2.  **Training:**
    *   Trajectories are grouped and sent to the server.
    *   The server trains your model using GRPO.
    *   The server saves the trained LoRA and loads it into vLLM.
    *   Inference resumes.

This loop runs until a specified number of iterations is complete.

## 🧩 Supported Models

ART is designed to be compatible with most vLLM/HuggingFace-transformers compatible causal language models, or at least the ones supported by [Unsloth](https://docs.unsloth.ai/get-started/all-our-models).  If you encounter any compatibility issues, please reach out on [Discord](https://discord.gg/zbBHRUpwf4) or open an issue on [GitHub](https://github.com/openpipe/art/issues)!

## 🤝 Contributing

We welcome contributions to ART! Please review the [CONTRIBUTING.md](CONTRIBUTING.md) file to learn how to get involved.

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

We thank the following projects for their contributions to ART's development:

*   [Unsloth](https://github.com/unslothai/unsloth)
*   [vLLM](https://github.com/vllm-project/vllm)
*   [trl](https://github.com/huggingface/trl)
*   [torchtune](https://github.com/pytorch/torchtune)
*   [SkyPilot](https://github.com/skypilot-org/skypilot)

And a special thanks to our partners for helping us test ART!

[pypi-url]: https://pypi.org/project/openpipe-art/
[contribute-url]: https://github.com/openpipe/art/blob/main/CONTRIBUTING.md
[contribute-image]: https://img.shields.io/badge/PRs-welcome-blue.svg
```

Key improvements and explanations:

*   **SEO Optimization:**  Keywords like "Agent Reinforcement Trainer", "LLM Agents", "Reinforcement Learning", and "GRPO" are integrated naturally throughout the text.  Headings and subheadings are used to improve readability and SEO ranking.  The title is optimized.
*   **One-Sentence Hook:**  The first sentence serves as a compelling hook, immediately explaining the value proposition.
*   **Clear Structure:**  Uses well-defined headings (Key Features, RULER, Example Notebooks, Why Choose ART, etc.) to organize information and guide the reader.
*   **Bulleted Key Features:** Highlights the core benefits in a clear, scannable format.
*   **Concise Descriptions:** Provides brief, informative descriptions of each section and feature.
*   **Action-Oriented Language:**  Uses verbs like "train", "empowers", "accelerate", and "get started" to encourage engagement.
*   **Links Back to Repo:** Includes a prominent link back to the original repository at the top.
*   **Improved Formatting:**  Enhanced markdown for better readability, including bolding and the use of icons where appropriate.
*   **Focused on Value:**  Emphasizes the benefits of using ART.
*   **Removed Redundancy:** Streamlined descriptions and removed unnecessary words.
*   **Clearer Language:** Simplified some phrasing for better understanding.
*   **Expanded Model Support Information:**  Added a more specific sentence about supported models.
*   **Call to Action:**  Encourages the reader to get started and contribute.
*   **Emphasis on RULER:**  Gives RULER its own section.