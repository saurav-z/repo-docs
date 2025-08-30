<div align="center">

<a href="https://art.openpipe.ai"><picture>
<img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
</picture></a>

<p align="center">
  <h1>Agent Reinforcement Trainer (ART)</h1>
</p>

<p>
  Train advanced multi-step agents for real-world tasks using Reinforcement Learning.
</p>

[![PRs-Welcome][contribute-image]][contribute-url]
[![PyPI version](https://img.shields.io/pypi/v/openpipe-art?color=364fc7)][pypi-url]
[![Train Agent](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/2048/2048.ipynb)

[![Join Discord](https://img.shields.io/badge/Join%20Discord-5865F2?style=plastic&logo=discord&logoColor=white)](https://discord.gg/zbBHRUpwf4)
[![Documentation](https://img.shields.io/badge/Documentation-orange?style=plastic&logo=gitbook&logoColor=white)](https://art.openpipe.ai)

</div>

## 🚀 Supercharge Your Agents with Agent Reinforcement Trainer (ART)

**ART is an open-source framework that empowers you to train powerful, reliable agents by leveraging the experience of Large Language Models (LLMs).** This allows for more efficient and effective agent development than traditional methods.  [Learn more and explore the code at the original repository](https://github.com/OpenPipe/ART).

**Key Features:**

*   **RULER: Zero-Shot Reward Functions:** Automate reward scoring with Relative Universal LLM-Elicited Rewards (RULER), eliminating the need for manual reward engineering and complex functions.
*   **Faster Development:** Accelerate your development process by skipping reward function engineering, leading to 2-3x faster iterations.
*   **General-Purpose & Versatile:** ART works seamlessly across diverse tasks without requiring modifications, ensuring broad applicability.
*   **High Performance:** Achieve strong performance comparable to, or exceeding, hand-crafted rewards in various benchmarks.
*   **Easy Integration:** Integrate ART into your projects with ease, serving as a drop-in replacement for manual reward functions.
*   **Modular Architecture:** ART's client-server architecture allows you to train agents from any client machine, running on a local or remote GPU.
*   **Flexible Observability:** Integrations with platforms like W&B, Langfuse, and OpenPipe for enhanced debugging and monitoring.
*   **Customizable & Optimized:** Benefit from intelligent defaults for training parameters and inference engine configurations, or tailor them to your specific requirements.

## 🎯 RULER: Simplify Reward Engineering

RULER revolutionizes agent training by enabling LLMs to automatically score agent trajectories, removing the need for manual reward functions.

**Key Benefits of RULER:**

*   **Reduced Development Time:** Skip reward engineering, which can save valuable time.
*   **Broad Applicability:** Works well across diverse tasks without modifications.
*   **Comparable Performance:** Achieves performance that rivals hand-crafted reward systems.
*   **Simplified Integration:** Easily integrates into your existing projects.

```python
# Before: Manual Reward Engineering
def complex_reward_function(trajectory):
    # 50+ lines of scoring logic...
    pass

# After: Simple RULER Integration
judged_group = await ruler_score_group(group, "openai/o3")
```

[📖 Learn more about RULER →](https://art.openpipe.ai/fundamentals/ruler)

## ⚙️ ART: How It Works

ART provides a streamlined framework for integrating Reinforcement Learning (RL) into your Python applications, improving agent reliability and enabling LLMs to learn from their experiences.

## 📒 Example Notebooks: Train Agents on Various Tasks

Explore how ART can be applied to different agent tasks using our example notebooks.  Experiment with various models and see the powerful results!

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

## 📰 Stay Updated with the Latest ART News

*   **ART and LangGraph Integration:** Train LangGraph agents with RL for smarter multi-step reasoning and improved tool usage.
*   **MCP•RL:** Teach Your Model to Master Any MCP Server.
*   **AutoRL:** Zero-Data Training for Any Task with automatic input generation and RULER evaluation.
*   **RULER:** Automatic reward generation in reinforcement learning.
*   **ART·E:** Build an email research agent outperforming o3.
*   **ART Trainer:** Easy training of LLM-based agents using GRPO.

[📖 See all blog posts →](https://openpipe.ai/blog)

## 🛠️ Why Choose ART?

*   **Simplified Integration:** Use ART wrappers to introduce RL training into your existing applications.
*   **Flexible Training:** Train on your laptop or leverage GPU-enabled environments.
*   **Enhanced Observability:** Integrate with platforms for improved debugging and monitoring.
*   **Intelligent Defaults:** Customize training parameters or use optimized defaults.

## 💾 Installation

Get started by running the following command in your project:

```
pip install openpipe-art
```

## 🤖 ART•E Agent: Real-World Application

Discover how ART was used to train the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) to beat o3 at email retrieval!

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## 🔁 Training Loop: Client-Server Architecture

ART employs a client-server architecture:

1.  **Inference:** Your code uses the ART client to perform agentic workflows, with completion requests routed to the ART server.
2.  **Training:** Trajectories are grouped and sent to the server, which trains the model using GRPO.

The loop continues until a specified number of iterations are completed.

## 🧩 Supported Models

ART supports most vLLM/HuggingFace-transformers compatible causal language models, including those supported by [Unsloth](https://docs.unsloth.ai/get-started/all-our-models).  If you encounter issues with a particular model, please report it on [Discord](https://discord.gg/zbBHRUpwf4) or [GitHub](https://github.com/openpipe/art/issues)!

## 🤝 Contribute to ART

ART is an actively developed project. Contributions are welcomed! Review the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines.

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

## 🙏 Acknowledgements

ART is built upon the shoulders of giants. We are grateful to the open-source RL community and the creators of these projects:

-   [Unsloth](https://github.com/unslothai/unsloth)
-   [vLLM](https://github.com/vllm-project/vllm)
-   [trl](https://github.com/huggingface/trl)
-   [torchtune](https://github.com/pytorch/torchtune)
-   [SkyPilot](https://github.com/skypilot-org/skypilot)

Special thanks to our partners who helped us test ART! We're excited to see what you build.

[pypi-url]: https://pypi.org/project/openpipe-art/
[contribute-url]: https://github.com/openpipe/art/blob/main/CONTRIBUTING.md
[contribute-image]: https://img.shields.io/badge/PRs-welcome-blue.svg