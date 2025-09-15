<div align="center">
  <a href="https://art.openpipe.ai">
    <picture>
      <img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
    </picture>
  </a>

  <p align="center">
    <h1>Agent Reinforcement Trainer (ART)</h1>
  </p>

  <p>
    **Supercharge your LLM-powered agents with ART: the open-source framework for training multi-step agents using GRPO and zero-shot rewards.**
  </p>

  [![PRs-Welcome][contribute-image]][contribute-url]
  [![PyPI version](https://img.shields.io/pypi/v/openpipe-art?color=364fc7)][pypi-url]
  [![Train Agent](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/2048/2048.ipynb)

  [![Join Discord](https://img.shields.io/badge/Join%20Discord-5865F2?style=plastic&logo=discord&logoColor=white)](https://discord.gg/zbBHRUpwf4)
  [![Documentation](https://img.shields.io/badge/Documentation-orange?style=plastic&logo=gitbook&logoColor=white)](https://art.openpipe.ai)
</div>

## Key Features of ART

*   **Zero-Shot Reward Engineering with RULER:** Eliminate the need for hand-crafted reward functions by leveraging LLMs to automatically score agent actions, saving you time and effort.
    *   **Faster Development:** Reduce development time by 2-3x by skipping reward function engineering.
    *   **Universal Applicability:** ART works across various tasks without modification.
    *   **High Performance:** Achieve results that match or surpass hand-crafted reward functions in many benchmarks.
    *   **Easy Integration:** Seamlessly integrate RULER into your existing projects.
*   **Open-Source RL Framework:** ART provides a user-friendly framework for integrating GRPO into your Python applications.
*   **Flexible Training:** Train your agents from anywhere, with support for both local GPUs and cloud-based resources.
*   **Observability and Debugging:** Integrate with platforms like Weights & Biases, Langfuse, and OpenPipe for flexible monitoring and streamlined debugging.
*   **Intelligent Defaults:** Utilize optimized default settings or customize parameters to suit your specific needs.

## 📏 RULER: Zero-Shot Agent Rewards

RULER (Relative Universal LLM-Elicited Rewards) revolutionizes agent training by utilizing an LLM-as-judge to automatically assess agent trajectories. Simply define your task, and RULER handles the reward scoring—**no labeled data, expert feedback, or reward engineering needed.**

```python
# Before: Hours of reward engineering
def complex_reward_function(trajectory):
    # 50+ lines of careful scoring logic...
    pass

# After: One line with RULER
judged_group = await ruler_score_group(group, "openai/o3")
```

[📖 Learn more about RULER →](https://art.openpipe.ai/fundamentals/ruler)

## ART Overview

ART is an open-source Reinforcement Learning (RL) framework that allows LLMs to improve agents reliability by learning from experience. ART provides an ergonomic harness for integrating GRPO into any python application. For a quick hands-on introduction, run one of the notebooks below. When you're ready to learn more, check out the [docs](https://art.openpipe.ai).

## 📒 Example Notebooks - Train LLM Agents

Get started quickly with these ready-to-use notebooks demonstrating ART's capabilities:

| Agent Task           | Example Notebook                                                                                                                       | Description                                         | Comparative Performance                                                                                                                                                                                                    |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ART•E LangGraph**  | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/langgraph/art-e-langgraph.ipynb)   | Qwen 2.5 7B learns to search emails using LangGraph | [Link coming soon]                                                                                                                                                                                                         |
| **MCP•RL**           | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/mcp-rl/mcp-rl.ipynb)               | Qwen 2.5 3B masters the NWS MCP server              | [Link coming soon]                                                                                                                                                                                                         |
| **ART•E [RULER]**    | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/art-e.ipynb)                       | Qwen 2.5 7B learns to search emails using RULER     | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/email_agent/accuracy-training-progress.svg" height="72"> [benchmarks](/dev/art-e/art_e/evaluate/display_benchmarks.ipynb)                             |
| **2048**             | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/2048/2048.ipynb)                   | Qwen 2.5 3B learns to play 2048                     | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/2048/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/2048/display_benchmarks.ipynb)                                               |
| **Temporal Clue**    | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/temporal_clue/temporal-clue.ipynb) | Qwen 2.5 7B learns to solve Temporal Clue           | [Link coming soon]                                                                                                                                                                                                         |
| **Tic Tac Toe**      | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb)     | Qwen 2.5 3B learns to play Tic Tac Toe              | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/tic-tac-toe-local/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/tic_tac_toe/display-benchmarks.ipynb)                           |
| **Codenames**        | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/codenames/Codenames_RL.ipynb)      | Qwen 2.5 3B learns to play Codenames                | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/codenames/win_rate_over_time.png" height="72"> [benchmarks](https://github.com/OpenPipe/art-notebooks/blob/main/examples/codenames/Codenames_RL.ipynb) |
| **AutoRL [RULER]**   | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/auto_rl.ipynb)                     | Train Qwen 2.5 7B to master any task                | [Link coming soon]                                                                                                                                                                                                         |

## 📰 ART News

Stay up-to-date with the latest advancements in agent training:

*   🗞️  **[ART now integrates seamlessly with LangGraph](https://art.openpipe.ai/integrations/langgraph-integration)** - Train your LangGraph agents with reinforcement learning for smarter multi-step reasoning and improved tool usage.
*   🗞️  **[MCP•RL: Teach Your Model to Master Any MCP Server](https://x.com/corbtt/status/1953171838382817625)** - Automatically train models to effectively use MCP server tools through reinforcement learning.
*   🗞️  **[AutoRL: Zero-Data Training for Any Task](https://x.com/mattshumer_/status/1950572449025650733)** - Train custom AI models without labeled data using automatic input generation and RULER evaluation.
*   🗞️  **[RULER: Easy Mode for RL Rewards](https://openpipe.ai/blog/ruler-easy-mode-for-rl-rewards)** is now available for automatic reward generation in reinforcement learning.
*   🗞️  **[ART·E: How We Built an Email Research Agent That Beats o3](https://openpipe.ai/blog/art-e-mail-agent)** demonstrates a Qwen 2.5 14B email agent outperforming OpenAI's o3.
*   🗞️  **[ART Trainer: A New RL Trainer for Agents](https://openpipe.ai/blog/art-trainer)** enables easy training of LLM-based agents using GRPO.

[📖 See all blog posts →](https://openpipe.ai/blog)

## Why Choose ART?

*   **Simplified RL Integration:** ART provides easy-to-use wrappers for incorporating RL training into existing applications.
*   **Flexible Training Environment:** Run the ART client locally or utilize the ART server for GPU-accelerated training in the cloud.
*   **Enhanced Observability:** Integrate with popular platforms to monitor and debug your agent training process.
*   **Optimized Defaults & Customization:** Benefit from intelligent defaults or customize configurations to achieve optimal results.

## Installation

Get started with ART in your Python project with a simple `pip` command:

```bash
pip install openpipe-art
```

## 🤖 ART•E Agent

Learn how to apply ART to a real-world scenario with the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) blog post, where you'll discover how a Qwen 2.5 14B agent outperformed o3 in email retrieval!

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## 🔁 Training Loop Explained

ART utilizes a client-server architecture, dividing functionality into these two components:

1.  **Inference**
    1.  Your code uses the ART client to perform an agentic workflow.
    2.  Completion requests are routed to the ART server, which runs the model's latest LoRA in vLLM.
    3.  Each message is stored in a Trajectory.
    4.  When a rollout finishes, your code assigns a `reward` to its Trajectory.
2.  **Training**
    1.  Trajectories are grouped and sent to the server.
    2.  The server trains your model using GRPO.
    3.  The server saves the newly trained LoRA and loads it into vLLM.
    4.  Inference is unblocked and the loop resumes.

This process repeats until a specified number of iterations are completed.

## 🧩 Supported Models

ART is designed to work with most vLLM/HuggingFace-transformers compatible causal language models, especially those supported by [Unsloth](https://docs.unsloth.ai/get-started/all-our-models). Currently, Gemma 3 is not supported.  Please report any model compatibility issues on [Discord](https://discord.gg/zbBHRUpwf4) or [GitHub](https://github.com/openpipe/art/issues).

## 🤝 Contributing

We welcome contributions to ART! Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for details.

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

ART is built upon the work of many open-source contributors. We're particularly grateful to the authors of:

*   [Unsloth](https://github.com/unslothai/unsloth)
*   [vLLM](https://github.com/vllm-project/vllm)
*   [trl](https://github.com/huggingface/trl)
*   [torchtune](https://github.com/pytorch/torchtune)
*   [SkyPilot](https://github.com/skypilot-org/skypilot)

Thank you to our partners for their support and testing.  We can't wait to see what you build with ART!

[pypi-url]: https://pypi.org/project/openpipe-art/
[contribute-url]: https://github.com/openpipe/art/blob/main/CONTRIBUTING.md
[contribute-image]: https://img.shields.io/badge/PRs-welcome-blue.svg