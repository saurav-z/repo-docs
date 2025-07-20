<div align="center">

<a href="https://art.openpipe.ai"><picture>
<img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
</picture></a>

<p align="center">
  <h1>Agent Reinforcement Trainer (ART)</h1>
</p>

<p>
  Train advanced multi-step agents with Large Language Models (LLMs) for real-world tasks using GRPO, and experience faster development and improved agent performance.
</p>

[![PRs-Welcome][contribute-image]][contribute-url]
[![Downloads][downloads-image]][pypi-url]
[![Train Agent](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)

[![Join Discord](https://img.shields.io/badge/Join%20Discord-5865F2?style=plastic&logo=discord&logoColor=white)](https://discord.gg/zbBHRUpwf4)
[![Documentation](https://img.shields.io/badge/Documentation-orange?style=plastic&logo=gitbook&logoColor=white)](https://art.openpipe.ai)

</div>

## Agent Reinforcement Trainer (ART): The Fastest Way to Train Agents with LLMs

ART is an open-source framework designed to **simplify and accelerate the training of sophisticated, multi-step agents powered by Large Language Models (LLMs).**  Leveraging the power of Generalized Reward Policy Optimization (GRPO), ART enables developers to quickly build and deploy AI agents capable of handling complex tasks.  [See the original repo here](https://github.com/OpenPipe/ART).

### Key Features

*   **Rapid Development:** ART streamlines the agent training process, enabling significantly faster development cycles.
*   **GRPO Integration:**  Utilizes GRPO for efficient agent training, optimizing performance and reliability.
*   **Open Source:**  Benefit from the collaborative nature of an open-source project.
*   **Modular Design:**  ART integrates easily into existing Python applications with a clean client-server architecture.
*   **Flexible Training:** Supports training on local machines or in cloud environments.
*   **Observability:** Integrates with popular platforms like W&B, Langfuse, and OpenPipe for comprehensive debugging and monitoring.
*   **Optimized Defaults:**  ART comes with intelligent defaults for training parameters and inference engine configurations, which have been optimized for efficiency and stability.

## 📏 RULER: Zero-Shot Agent Rewards

**RULER** (Relative Universal LLM-Elicited Rewards) eliminates the need for hand-crafted reward functions by using an LLM-as-judge to automatically score agent trajectories. Simply define your task in the system prompt, and RULER handles the rest—**no labeled data, expert feedback, or reward engineering required**.

✨ **Key Benefits:**

*   **2-3x faster development** - Skip reward function engineering entirely
*   **General-purpose** - Works across any task without modification
*   **Strong performance** - Matches or exceeds hand-crafted rewards in 3/4 benchmarks
*   **Easy integration** - Drop-in replacement for manual reward functions

```python
# Before: Hours of reward engineering
def complex_reward_function(trajectory):
    # 50+ lines of careful scoring logic...
    pass

# After: One line with RULER
judged_group = await ruler_score_group(group, "openai/o3")
```

[📖 Learn more about RULER →](https://art.openpipe.ai/fundamentals/ruler)

## 📒 Example Notebooks: Train Agents on Real-World Tasks

Explore ART's capabilities with these interactive notebooks, showcasing agent training across a variety of tasks:

| Agent Task        | Example Notebook                                                                                                             | Description                               | Comparative Performance                                                                                                                                     |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ART•E [RULER]**         | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/art-e/art-e.ipynb)               | Qwen 2.5 7B learns to search emails using RULER     | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/email_agent/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/art-e/art_e/evaluate/display_benchmarks.ipynb)                                                                                                                                          |
| **2048**          | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)                   | Qwen 2.5 3B learns to play 2048           | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/2048/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/2048/benchmark_2048.ipynb)                            |
| **Temporal Clue** | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/temporal_clue/temporal-clue.ipynb) | Qwen 2.5 7B learns to solve Temporal Clue | [Link coming soon]                                                                                                                                          |
| **Tic Tac Toe**   | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb)     | Qwen 2.5 3B learns to play Tic Tac Toe    | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/tic-tac-toe-local/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/tic_tac_toe/benchmark_tic_tac_toe.ipynb) |
| **Codenames**     | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/codenames/Codenames_RL.ipynb)      | Qwen 2.5 3B learns to play Codenames      | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/codenames/win_rate_over_time.png" height="72"> [benchmarks](/examples/codenames/Codenames_RL.ipynb)                            |

## Why Choose ART?

*   **Seamless Integration:** Easily incorporate RL training into existing applications with convenient wrappers.
*   **Flexible Deployment:** Run the ART client locally while leveraging a GPU-enabled server environment for training.
*   **Enhanced Observability:** Integrate with platforms like W&B, Langfuse, and OpenPipe for simplified debugging and monitoring.
*   **Customization:** Configure training parameters and inference engine configurations to meet specific needs, or use the defaults.

## Installation

Get started with ART by installing the package:

```bash
pip install openpipe-art
```

## 🤖 ART•E Agent

Interested in a real-world application?  See how the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) was trained with Qwen 2.5 14B to excel at email retrieval!

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## 🔁 Training Loop Overview

ART employs a client-server architecture for a streamlined training process:

1.  **Inference:** The ART client executes agent workflows, sending completion requests to the ART server.
2.  **Training:** The server trains the LLM using GRPO, saves the updated LoRA, and loads it into vLLM.
3.  The loop repeats, refining the agent with each iteration.

## 🧩 Supported Models

ART is designed to work with a wide array of causal language models compatible with vLLM and Hugging Face transformers, including those supported by [Unsloth](https://docs.unsloth.ai/get-started/all-our-models).  If you encounter any compatibility issues, please report them on [Discord](https://discord.gg/zbBHRUpwf4) or [GitHub](https://github.com/openpipe/art/issues).

## 🤝 Contributing

ART welcomes contributions! Refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file for details on how to contribute.

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

ART builds upon the work of many in the open-source RL community. Special thanks to the authors of:

*   [Unsloth](https://github.com/unslothai/unsloth)
*   [vLLM](https://github.com/vllm-project/vllm)
*   [trl](https://github.com/huggingface/trl)
*   [torchtune](https://github.com/pytorch/torchtune)
*   [SkyPilot](https://github.com/skypilot-org/skypilot)

We appreciate the support of our partners who have helped us test and refine ART!  We look forward to seeing your creations.

[pypi-url]: https://pypi.org/project/openpipe-art/
[contribute-url]: https://github.com/openpipe/art/blob/main/CONTRIBUTING.md
[contribute-image]: https://img.shields.io/badge/PRs-welcome-blue.svg
[downloads-image]: https://img.shields.io/pypi/dm/openpipe-art?color=364fc7&logoColor=364fc7
```
Key changes and improvements:

*   **SEO Optimization:** Added keywords like "Agent Reinforcement Trainer", "LLMs", "GRPO", "AI Agents", and "Open Source" throughout the README.  Used headings to structure the content for better readability.
*   **Clear Value Proposition:**  The introductory sentence clearly states the benefit of using ART ("simplify and accelerate the training of sophisticated, multi-step agents").
*   **Concise Bullet Points:**  Features are presented as bullet points for easy scanning.
*   **Stronger Section Headings:**  More descriptive and engaging headings.
*   **Call to Action:** The introduction now includes a clear call to action to train agents using LLMs.
*   **Improved Language:**  Used more active and engaging language throughout the document.
*   **Removed Redundancy:**  Combined information where possible to keep the document concise.
*   **Emphasis on Benefits:**  Highlighted the key benefits of ART.
*   **Simplified Installation Instructions:**  Kept installation simple.
*   **Enhanced Content:** The document provides a more compelling reason to explore and use ART.