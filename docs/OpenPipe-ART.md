<div align="center">

<a href="https://art.openpipe.ai"><picture>
<img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
</picture></a>

<p align="center">
  <h1>Agent Reinforcement Trainer (ART)</h1>
</p>

<p>
  **Supercharge your LLM agents with ART, an open-source framework for training multi-step agents using GRPO and RULER.**
</p>

[![PRs-Welcome][contribute-image]][contribute-url]
[![Downloads][downloads-image]][pypi-url]
[![Train Agent](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)

[![Join Discord](https://img.shields.io/badge/Join%20Discord-5865F2?style=plastic&logo=discord&logoColor=white)](https://discord.gg/zbBHRUpwf4)
[![Documentation](https://img.shields.io/badge/Documentation-orange?style=plastic&logo=gitbook&logoColor=white)](https://art.openpipe.ai)

</div>

## Agent Reinforcement Trainer (ART): Train Smarter Agents with Ease

ART empowers you to train robust, multi-step LLM agents for real-world tasks, rapidly improving their performance through reinforcement learning.  This framework provides an ergonomic harness for integrating GRPO into any python application.

**Key Features:**

*   **Rapid Development:**  Skip the tedious reward function engineering with **RULER**, our zero-shot agent rewards system.
*   **General-Purpose:**  ART works across diverse tasks without modification, making it a versatile solution for various agent applications.
*   **High Performance:**  Achieve performance that matches or surpasses hand-crafted rewards in many benchmarks.
*   **Seamless Integration:**  Easily integrate ART into your existing projects with a simple installation process and intuitive API.
*   **Open-Source & Customizable:** Leverage the flexibility of an open-source framework with intelligent defaults and configurable training parameters.
*   **Easy Debugging & Observability:** Integrations with hosted platforms like W&B, Langfuse, and OpenPipe provide flexible observability and simplify debugging.

## 📏 RULER: Zero-Shot Agent Rewards

**RULER** (Relative Universal LLM-Elicited Rewards) revolutionizes agent training by using an LLM-as-judge to automatically score agent trajectories.  Simply describe your task, and RULER handles the rest, eliminating the need for hand-crafted reward functions.

✨ **Benefits of RULER:**

*   **2-3x Faster Development:** Significantly reduce development time by bypassing reward function engineering.
*   **Versatile:**  Adaptable to any task without requiring modifications to your core logic.
*   **Effective Performance:**  Matches or surpasses the performance of hand-crafted reward functions in many scenarios.
*   **Simple Integration:**  Implement RULER as a drop-in replacement for manual reward functions.

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

ART is an open-source RL framework designed to elevate agent reliability by enabling LLMs to **learn from experience**. The framework makes it easy to integrate GRPO into your Python applications, providing convenient wrappers for adding RL training into your existing workflows.

## 📒 Example Notebooks

Quickly get hands-on with ART using our interactive notebooks. Explore examples of training agents for different tasks.

| Agent Task        | Example Notebook                                                                                                             | Description                                      | Performance                                                                                                                                         |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ART•E [RULER]**         | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/art-e/art-e.ipynb)               | Qwen 2.5 7B learns to search emails using RULER     | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/email_agent/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/art-e/art_e/evaluate/display_benchmarks.ipynb)                                                                                                                                          |
| **2048**          | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)                   | Qwen 2.5 3B learns to play 2048           | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/2048/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/2048/benchmark_2048.ipynb)                            |
| **Temporal Clue** | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/temporal_clue/temporal-clue.ipynb) | Qwen 2.5 7B learns to solve Temporal Clue | [Link coming soon]                                                                                                                                          |
| **Tic Tac Toe**   | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb)     | Qwen 2.5 3B learns to play Tic Tac Toe    | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/tic-tac-toe-local/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/tic_tac_toe/benchmark_tic_tac_toe.ipynb) |
| **Codenames**     | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/codenames/Codenames_RL.ipynb)      | Qwen 2.5 3B learns to play Codenames      | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/codenames/win_rate_over_time.png" height="72"> [benchmarks](/examples/codenames/Codenames_RL.ipynb)                            |

## Why Choose ART?

*   **Easy Integration:**  Seamlessly introduce RL into existing applications.
*   **Flexible Training:** Train locally or leverage GPU-enabled environments.
*   **Observability:** Integrations with platforms like W&B, Langfuse, and OpenPipe streamline debugging and monitoring.
*   **Intelligent Defaults:** Benefit from optimized default settings for efficiency and stability.

## Installation

Get started quickly by installing the ART package:

```bash
pip install openpipe-art
```

## 🤖 ART•E Agent

Dive deeper into a practical application with the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) blog post, where we detail training Qwen 2.5 14B to outperform o3 in email retrieval!

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## 🔁 Training Loop Explained

ART operates with a client-server architecture. The client interfaces with your codebase, and the server manages the training process.

1.  **Inference:**
    1.  Your code utilizes the ART client for agentic workflows.
    2.  Completion requests are directed to the ART server, which runs the latest LoRA in vLLM.
    3.  Each message (system, user, assistant) is stored in a Trajectory.
    4.  When a rollout concludes, your code assigns a `reward` to the Trajectory.

2.  **Training:**
    1.  Completed Trajectories are sent to the server for grouping.
    2.  The server trains the model using GRPO.
    3.  The server saves the trained LoRA and loads it into vLLM.
    4.  Inference resumes.

This loop repeats until a predefined number of iterations are completed.

## 🧩 Supported Models

ART supports most vLLM/HuggingFace-transformers compatible causal language models (and those supported by [Unsloth](https://docs.unsloth.ai/get-started/all-our-models)).  If a model isn't working, let us know on [Discord](https://discord.gg/zbBHRUpwf4) or open an issue on [GitHub](https://github.com/openpipe/art/issues)!

## 🤝 Contributing

Contributions are welcome! Review [CONTRIBUTING.md](CONTRIBUTING.md) for more information.

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

ART is built upon the shoulders of many. We're especially grateful to the authors of:

*   [Unsloth](https://github.com/unslothai/unsloth)
*   [vLLM](https://github.com/vllm-project/vllm)
*   [trl](https://github.com/huggingface/trl)
*   [torchtune](https://github.com/pytorch/torchtune)
*   [SkyPilot](https://github.com/skypilot-org/skypilot)

And a special thank you to our partners for helping us test ART!

[**Explore ART on GitHub**](https://github.com/OpenPipe/ART)
[pypi-url]: https://pypi.org/project/openpipe-art/
[contribute-url]: https://github.com/openpipe/art/blob/main/CONTRIBUTING.md
[contribute-image]: https://img.shields.io/badge/PRs-welcome-blue.svg
[downloads-image]: https://img.shields.io/pypi/dm/openpipe-art?color=364fc7&logoColor=364fc7
```

Key improvements and SEO optimizations:

*   **Clear Headline & Introduction:**  A concise, keyword-rich introduction sets the stage.
*   **Keyword Optimization:**  Uses relevant keywords like "LLM agents," "reinforcement learning," "GRPO," and "RULER" throughout the text.
*   **Bulleted Key Features:** Highlights the core benefits in an easy-to-scan format.
*   **Subheadings:**  Organizes the content logically for better readability.
*   **Strong Call to Action:**  Encourages exploration of the project with the link.
*   **Structured Data:** Headings and lists allow for better search engine understanding.
*   **Clearer language:** Avoids jargon where possible.
*   **Concise:**  Improved readability.
*   **Focus on Benefits:**  Emphasizes what users *gain* from using ART.
*   **Simplified Training Loop:** Easier to understand.
*   **Model Compatibility:** Mentioned model compatability.