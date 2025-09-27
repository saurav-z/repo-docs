<div align="center">
  <a href="https://art.openpipe.ai">
    <picture>
      <img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
    </picture>
  </a>

  <h1>Agent Reinforcement Trainer (ART): Supercharge Your LLM Agents</h1>
  <p>
    <b>Train powerful, multi-step LLM agents for real-world tasks with ease using Agent Reinforcement Trainer (ART).</b>
  </p>

  [![PRs-Welcome][contribute-image]][contribute-url]
  [![PyPI version](https://img.shields.io/pypi/v/openpipe-art?color=364fc7)][pypi-url]
  [![Train Agent](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/2048/2048.ipynb)

  [![Join Discord](https://img.shields.io/badge/Join%20Discord-5865F2?style=plastic&logo=discord&logoColor=white)](https://discord.gg/zbBHRUpwf4)
  [![Documentation](https://img.shields.io/badge/Documentation-orange?style=plastic&logo=gitbook&logoColor=white)](https://art.openpipe.ai)
</div>

## Overview

Agent Reinforcement Trainer (ART) is an open-source framework designed to enhance the performance and reliability of LLM agents by enabling them to learn from experience through Reinforcement Learning (RL) methods. ART provides a streamlined approach for integrating GRPO into your Python applications.

**Key Features:**

*   **Simplified Training:** Train agents without complex reward function engineering using RULER.
*   **Zero-Shot Reward Functions with RULER:** Utilize LLMs to automatically score agent trajectories, eliminating the need for hand-crafted reward functions.
*   **Faster Development:** Accelerate your development process by skipping reward function engineering.
*   **General Purpose:** Apply ART across various tasks without modification.
*   **Strong Performance:** Achieve results that match or surpass hand-crafted rewards.
*   **Easy Integration:** Easily integrate ART into your existing projects.
*   **Modular Architecture:** The client-server design simplifies integration into existing applications.
*   **Flexible Training:** Run training on various infrastructures from local GPUs to cloud environments.

**Get Started:**

1.  **Install:** `pip install openpipe-art`
2.  **Explore Notebooks:** Dive into practical examples to see ART in action.

## RULER: Zero-Shot Agent Rewards

RULER (Relative Universal LLM-Elicited Rewards) revolutionizes agent training by employing an LLM-as-judge to automatically evaluate agent trajectories, eliminating the need for manual reward function creation. Simply define your task in the system prompt, and RULER handles the rest.

**Key Benefits of RULER:**

*   **2-3x Faster Development:** Significantly reduce development time by skipping reward function engineering.
*   **Versatile:** Works across diverse tasks without modification.
*   **High Performance:** Delivers results that match or outperform hand-crafted reward functions in numerous benchmarks.
*   **Seamless Integration:** Easily integrates into existing workflows.

```python
# Before: Complex reward engineering
def complex_reward_function(trajectory):
    # 50+ lines of careful scoring logic...
    pass

# After: One line with RULER
judged_group = await ruler_score_group(group, "openai/o3")
```

[📖 Learn more about RULER →](https://art.openpipe.ai/fundamentals/ruler)

## Notebooks & Examples

Explore example notebooks to quickly get started with ART and see it in action.

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

Stay informed about the latest advancements in ART and LLM agent development:

*   🗞️ **[ART now integrates seamlessly with LangGraph](https://art.openpipe.ai/integrations/langgraph-integration)** - Train your LangGraph agents with reinforcement learning for smarter multi-step reasoning and improved tool usage.
*   🗞️ **[MCP•RL: Teach Your Model to Master Any MCP Server](https://x.com/corbtt/status/1953171838382817625)** - Automatically train models to effectively use MCP server tools through reinforcement learning.
*   🗞️ **[AutoRL: Zero-Data Training for Any Task](https://x.com/mattshumer_/status/1950572449025650733)** - Train custom AI models without labeled data using automatic input generation and RULER evaluation.
*   🗞️ **[RULER: Easy Mode for RL Rewards](https://openpipe.ai/blog/ruler-easy-mode-for-rl-rewards)** is now available for automatic reward generation in reinforcement learning.
*   🗞️ **[ART·E: How We Built an Email Research Agent That Beats o3](https://openpipe.ai/blog/art-e-mail-agent)** demonstrates a Qwen 2.5 14B email agent outperforming OpenAI's o3.
*   🗞️ **[ART Trainer: A New RL Trainer for Agents](https://openpipe.ai/blog/art-trainer)** enables easy training of LLM-based agents using GRPO.

[📖 See all blog posts →](https://openpipe.ai/blog)

## Why ART?

ART simplifies integrating RL into your applications.

*   **Simple Integration**:  ART provides convenient wrappers for introducing RL training into existing applications. We abstract the training server into a modular service that your code doesn't need to interface with.
*   **Flexible Training Environment**: Train from anywhere with a laptop, a local GPU, or the cloud.  ART offers cloud-based GPU environments for training.
*   **Observability and Debugging:**  Integrations with platforms like W&B, Langfuse, and OpenPipe provide flexible observability to simplify the debugging process.
*   **Intelligent Defaults & Customization**: ART offers optimized defaults for efficient and stable training while allowing you to configure parameters and inference engine configurations.

## 🤖 ART•E Agent Example

Discover the power of ART with a real-world application: the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent), which successfully trained a Qwen 2.5 14B model to surpass o3 in email retrieval.

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## 🔁 Training Loop Explained

ART utilizes a client-server architecture.  The client interacts with your code, while the server handles the training.

1.  **Inference:**
    1.  Your code uses the ART client to perform agentic workflows.
    2.  Completions are routed to the ART server, using the model's latest LoRA in vLLM.
    3.  Messages are stored in a Trajectory.
    4.  When a rollout finishes, your code assigns a reward.

2.  **Training:**
    1.  Finished Trajectories are grouped and sent to the server.
    2.  The server trains your model using GRPO, initializing from the latest checkpoint.
    3.  The server saves the LoRA and loads it into vLLM.
    4.  Inference resumes, and the loop restarts.

This loop continues until the specified number of iterations is completed.

## 🧩 Supported Models

ART is compatible with most vLLM/HuggingFace-transformers compatible causal language models, and specifically those supported by [Unsloth](https://docs.unsloth.ai/get-started/all-our-models). If you experience any issues with a specific model, please report it on [Discord](https://discord.gg/zbBHRUpwf4) or [GitHub](https://github.com/openpipe/art/issues).

## 🤝 Contributing

ART is actively developed, and contributions are highly encouraged!  Refer to [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

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

ART is built upon the shoulders of giants in the open-source RL community. We are especially grateful to the authors of:

*   [Unsloth](https://github.com/unslothai/unsloth)
*   [vLLM](https://github.com/vllm-project/vllm)
*   [trl](https://github.com/huggingface/trl)
*   [torchtune](https://github.com/pytorch/torchtune)
*   [SkyPilot](https://github.com/skypilot-org/skypilot)

Thank you to our partners who've helped us test ART!

[pypi-url]: https://pypi.org/project/openpipe-art/
[contribute-url]: https://github.com/openpipe/art/blob/main/CONTRIBUTING.md
[contribute-image]: https://img.shields.io/badge/PRs-welcome-blue.svg

[**Back to the top**](https://github.com/OpenPipe/ART)