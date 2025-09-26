<div align="center">
  <a href="https://art.openpipe.ai">
    <picture>
      <img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
    </picture>
  </a>
  <h1>Agent Reinforcement Trainer (ART): Train LLM Agents with Ease</h1>
  <p>
    <b>Supercharge your LLM agent development with ART, an open-source framework that simplifies training multi-step agents for real-world tasks.</b>
  </p>
  <p>
    <a href="https://github.com/openpipe/art">
      <img src="https://img.shields.io/github/stars/OpenPipe/ART?style=social" alt="GitHub Stars">
    </a>
    <a href="https://pypi.org/project/openpipe-art">
      <img src="https://img.shields.io/pypi/v/openpipe-art?color=364fc7" alt="PyPI version">
    </a>
    <a href="https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/2048/2048.ipynb">
      <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Train Agent on Colab">
    </a>
    <a href="https://discord.gg/zbBHRUpwf4">
      <img src="https://img.shields.io/badge/Join%20Discord-5865F2?style=plastic&logo=discord&logoColor=white" alt="Join Discord">
    </a>
    <a href="https://art.openpipe.ai">
      <img src="https://img.shields.io/badge/Documentation-orange?style=plastic&logo=gitbook&logoColor=white" alt="Documentation">
    </a>
  </p>
</div>

## Key Features

*   🚀 **Rapid Development:** Dramatically reduce development time by eliminating the need for manual reward function engineering.
*   🧠 **Zero-Shot Rewards with RULER:** Leverage RULER (Relative Universal LLM-Elicited Rewards) to automatically score agent trajectories using an LLM-as-judge. Simply define your task!
*   ⚙️ **General Purpose:** ART works seamlessly across various tasks without modification.
*   📈 **High Performance:** Achieve performance that matches or surpasses hand-crafted rewards.
*   💻 **Easy Integration:** Quickly integrate ART into your existing applications with a simple `pip install`.
*   🧪 **Open Source & Customizable:** Benefit from an open-source framework with intelligent defaults and easy customization options.
*   📚 **Comprehensive Examples:** Get started quickly with notebooks showcasing agent training for various tasks, including 2048, email search, and more.
*   ☁️ **Flexible Infrastructure:** Run on your laptop or leverage cloud-based GPU environments.

##  RULER: Zero-Shot Agent Rewards Explained

**RULER (Relative Universal LLM-Elicited Rewards)** is a groundbreaking feature that simplifies LLM agent development by eliminating the need for hand-crafted reward functions.  It uses an LLM-as-judge to automatically score agent trajectories. Define your task in the system prompt, and RULER handles the rest.

**Benefits of RULER:**

*   **Faster Development:** 2-3x faster development compared to manual reward engineering.
*   **Versatile:** Works across any task without code changes.
*   **Effective:** Delivers strong performance, often matching or exceeding hand-crafted rewards.
*   **Simple to Use:** Easy to integrate. Drop-in replacement for manual reward functions.

```python
# Before: Time-consuming reward engineering
def complex_reward_function(trajectory):
    # 50+ lines of careful scoring logic...
    pass

# After: Simple with RULER
judged_group = await ruler_score_group(group, "openai/o3")
```

[📖 Learn more about RULER →](https://art.openpipe.ai/fundamentals/ruler)

## ART Overview

ART is an open-source reinforcement learning (RL) framework designed to enhance agent reliability by enabling LLMs to learn from experience. It provides a streamlined system for integrating GRPO into Python applications.

## 📒 Notebooks: Train Agents on Various Tasks

Explore our interactive notebooks for hands-on experience with ART.

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

Stay up-to-date with the latest ART developments and research.

*   🗞️ **[ART now integrates seamlessly with LangGraph](https://art.openpipe.ai/integrations/langgraph-integration)** - Train your LangGraph agents with reinforcement learning for smarter multi-step reasoning and improved tool usage.
*   🗞️ **[MCP•RL: Teach Your Model to Master Any MCP Server](https://x.com/corbtt/status/1953171838382817625)** - Automatically train models to effectively use MCP server tools through reinforcement learning.
*   🗞️ **[AutoRL: Zero-Data Training for Any Task](https://x.com/mattshumer_/status/1950572449025650733)** - Train custom AI models without labeled data using automatic input generation and RULER evaluation.
*   🗞️ **[RULER: Easy Mode for RL Rewards](https://openpipe.ai/blog/ruler-easy-mode-for-rl-rewards)** is now available for automatic reward generation in reinforcement learning.
*   🗞️ **[ART·E: How We Built an Email Research Agent That Beats o3](https://openpipe.ai/blog/art-e-mail-agent)** demonstrates a Qwen 2.5 14B email agent outperforming OpenAI's o3.
*   🗞️ **[ART Trainer: A New RL Trainer for Agents](https://openpipe.ai/blog/art-trainer)** enables easy training of LLM-based agents using GRPO.

[📖 See all blog posts →](https://openpipe.ai/blog)

## Why Choose ART?

*   **Easy Integration:**  ART offers convenient wrappers for introducing RL training into your **existing applications**, abstracting the training server into a modular service.
*   **Flexible Training:** Train agents from anywhere - your laptop or cloud GPUs.
*   **Observability & Debugging:** Integrations with hosted platforms such as W&B, Langfuse, and OpenPipe.
*   **Optimized Defaults:** Benefit from optimized defaults for training efficiency and stability.

## Installation

Easily integrate ART into your project using pip:

```bash
pip install openpipe-art
```

## 🤖 ART•E Agent: Real-World Example

Learn how to use ART for a practical task with the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) blog post.  It details the process of training Qwen 2.5 14B to outperform o3 in email retrieval.

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700" alt="ART E Agent Performance">

## 🔁 Training Loop Overview

ART employs a client-server architecture for its training loop:

1.  **Inference:** The ART client interfaces with your code, running agentic workflows.  Completion requests are routed to the ART server (which uses vLLM). Each `system`, `user`, and `assistant` message is stored in a Trajectory. When a rollout is done, your code provides a `reward`.
2.  **Training:** Once all rollouts complete, Trajectories are sent to the server for training.  The server trains your model with GRPO, saving the new LoRA and loading it into vLLM. Inference then resumes.

The loop continues until the set number of inference and training iterations is finished.

## 🧩 Supported Models

ART is designed to work with most vLLM/HuggingFace-transformers compatible causal language models, including those supported by [Unsloth](https://docs.unsloth.ai/get-started/all-our-models).  If you encounter issues with a specific model, please reach out on [Discord](https://discord.gg/zbBHRUpwf4) or open an issue on [GitHub](https://github.com/openpipe/art/issues)!

## 🤝 Contributing

We welcome contributions!  See the [CONTRIBUTING.md](CONTRIBUTING.md) file for details.

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

This repository is licensed under the [Apache-2.0 License](LICENSE).

## 🙏 Credits

ART is built upon the work of many. We are especially grateful to the authors of the following projects:

*   [Unsloth](https://github.com/unslothai/unsloth)
*   [vLLM](https://github.com/vllm-project/vllm)
*   [trl](https://github.com/huggingface/trl)
*   [torchtune](https://github.com/pytorch/torchtune)
*   [SkyPilot](https://github.com/skypilot-org/skypilot)

Thanks also to our partners for helping us test ART.

[Return to the top of the page](#)