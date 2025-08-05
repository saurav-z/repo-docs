<div align="center">

<a href="https://art.openpipe.ai"><picture>
<img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
</picture></a>

<p align="center">
  <h1>Agent Reinforcement Trainer (ART)</h1>
</p>

<p>
  Train multi-step agents for real-world tasks with GRPO.
</p>

[![PRs-Welcome][contribute-image]][contribute-url]
[![Downloads][downloads-image]][pypi-url]
[![Train Agent](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)

[![Join Discord](https://img.shields.io/badge/Join%20Discord-5865F2?style=plastic&logo=discord&logoColor=white)](https://discord.gg/zbBHRUpwf4)
[![Documentation](https://img.shields.io/badge/Documentation-orange?style=plastic&logo=gitbook&logoColor=white)](https://art.openpipe.ai)

</div>

## Agent Reinforcement Trainer (ART): Revolutionize Agent Training with AI

ART is an open-source framework empowering you to train sophisticated, multi-step agents for real-world tasks using the power of GRPO, improving agent reliability by allowing LLMs to learn from experience. **Supercharge your agent development with ART, skipping reward function engineering and achieving state-of-the-art performance.**

Key Features:

*   **Effortless Reward Engineering with RULER:** Leverages an LLM-as-judge to automatically score agent trajectories, eliminating the need for hand-crafted reward functions.
*   **Faster Development:** Accelerate your development process with a 2-3x speedup by skipping reward function engineering.
*   **Universal Applicability:** Adaptable to any task without modification.
*   **High Performance:** Achieves or surpasses hand-crafted rewards in 3 out of 4 benchmarks.
*   **Simple Integration:** A drop-in replacement for manual reward functions, simplifying your workflow.
*   **Open-Source and Customizable:** Benefit from an open-source framework with intelligent defaults and easy customization.

## 📏 RULER: Zero-Shot Agent Rewards

**RULER** (Relative Universal LLM-Elicited Rewards) simplifies agent training by using an LLM to automatically score agent behavior, eliminating the need for complex reward functions. Simply define your task in the system prompt, and RULER handles the rest.

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

ART provides an ergonomic harness for integrating GRPO into any Python application.

## 📒 Example Notebooks - Get Started Quickly!

Explore example notebooks to get hands-on experience training agents across various tasks:

| Agent Task         | Example Notebook                                                                                                             | Description                                     | Comparative Performance                                                                                                                                                                             |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ART•E [RULER]**  | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/art-e/art-e.ipynb)                 | Qwen 2.5 7B learns to search emails using RULER | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/email_agent/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/art-e/art_e/evaluate/display_benchmarks.ipynb) |
| **2048**           | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)                   | Qwen 2.5 3B learns to play 2048                 | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/2048/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/2048/benchmark_2048.ipynb)                            |
| **Temporal Clue**  | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/temporal_clue/temporal-clue.ipynb) | Qwen 2.5 7B learns to solve Temporal Clue       | [Link coming soon]                                                                                                                                                                                  |
| **Tic Tac Toe**    | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb)     | Qwen 2.5 3B learns to play Tic Tac Toe          | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/tic-tac-toe-local/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/tic_tac_toe/benchmark_tic_tac_toe.ipynb) |
| **Codenames**      | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/codenames/Codenames_RL.ipynb)      | Qwen 2.5 3B learns to play Codenames            | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/codenames/win_rate_over_time.png" height="72"> [benchmarks](/examples/codenames/Codenames_RL.ipynb)                            |
| **AutoRL [RULER]** | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/auto_rl.ipynb)                     | Train Qwen 2.5 7B to master any task            | [Link coming soon]                                                                                                                                                                                  |

## 📰 ART News and Updates

Stay up-to-date with the latest research and developments in agent training:

*   🗞️ **[AutoRL: Zero-Data Training for Any Task](https://x.com/mattshumer_/status/1950572449025650733)** - Train custom AI models without labeled data using automatic input generation and RULER evaluation.
*   🗞️ **[RULER: Easy Mode for RL Rewards](https://openpipe.ai/blog/ruler-easy-mode-for-rl-rewards)** is now available for automatic reward generation in reinforcement learning.
*   🗞️ **[ART·E: How We Built an Email Research Agent That Beats o3](https://openpipe.ai/blog/art-e-mail-agent)** demonstrates a Qwen 2.5 14B email agent outperforming OpenAI's o3.
*   🗞️ **[ART Trainer: A New RL Trainer for Agents](https://openpipe.ai/blog/art-trainer)** enables easy training of LLM-based agents using GRPO.

[📖 See all blog posts →](https://openpipe.ai/blog)

## Why Choose ART?

*   **Easy Integration:** ART provides convenient wrappers for introducing RL training into your existing applications, simplifying the training process.
*   **Flexible Training:** Train from anywhere – on your local machine, with GPU, or in the cloud.
*   **Seamless Observability:** Integrations with platforms like W&B, Langfuse, and OpenPipe for flexible monitoring and debugging.
*   **Intelligent Defaults:** Benefit from optimized training parameters and inference engine configurations for efficiency and stability, customizable to your needs.

## 🚀 Installation

Get started with ART by installing the Python package:

```bash
pip install openpipe-art
```

## 🤖 ART•E Agent: Real-World Example

Explore the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) blog post to see how ART was used to train Qwen 2.5 14B to outperform o3 in email retrieval.

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## 🔁 Training Loop: How ART Works

ART utilizes a client-server architecture to manage the reinforcement learning loop:

1.  **Inference:**
    *   Your code uses the ART client to perform agentic workflows, with completion requests sent to the ART server.
    *   The server runs the model's latest LoRA in vLLM.
    *   Each `system`, `user`, and `assistant` message is stored in a Trajectory.
    *   When a rollout finishes, your code assigns a `reward` to its Trajectory.

2.  **Training:**
    *   Trajectories are grouped and sent to the server after each rollout.
    *   The server trains your model using GRPO, initializing from the latest checkpoint.
    *   The server saves the trained LoRA and loads it into vLLM.
    *   The loop repeats with inference.

This loop continues until a specified number of iterations is reached.

## 🧩 Supported Models

ART is designed to work with most vLLM/HuggingFace-transformers compatible causal language models. For a current list of compatible models, see [Unsloth](https://docs.unsloth.ai/get-started/all-our-models).

## 🤝 Contributing

We welcome contributions to ART! Please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file for more information.

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

ART is available under the [Apache-2.0 License](LICENSE).

## 🙏 Credits

ART builds on the work of many in the open-source RL community. We are especially grateful to the following projects:

*   [Unsloth](https://github.com/unslothai/unsloth)
*   [vLLM](https://github.com/vllm-project/vllm)
*   [trl](https://github.com/huggingface/trl)
*   [torchtune](https://github.com/pytorch/torchtune)
*   [SkyPilot](https://github.com/skypilot-org/skypilot)

Thank you to our partners who have helped us test ART!

[pypi-url]: https://pypi.org/project/openpipe-art/
[contribute-url]: https://github.com/openpipe/art/blob/main/CONTRIBUTING.md
[contribute-image]: https://img.shields.io/badge/PRs-welcome-blue.svg
[downloads-image]: https://img.shields.io/pypi/dm/openpipe-art?color=364fc7&logoColor=364fc7
```
Key improvements and SEO considerations:

*   **Clear and Concise Hook:**  The opening sentence immediately highlights the core benefit: faster agent development.
*   **Keyword Optimization:**  Uses relevant keywords like "Agent Reinforcement Trainer," "GRPO," "LLMs," and "agent training."
*   **Headings and Structure:** Uses clear, descriptive headings to improve readability and SEO.
*   **Bulleted Key Features:** Highlights the main advantages of ART, making them easy to scan.
*   **Strong Calls to Action:** Encourages users to "Get Started," "Learn More," and "Explore Examples."
*   **Internal Linking:** Uses internal links to other sections of the README.
*   **External Linking:**  Maintains all existing links and adds descriptive anchor text.
*   **Concise Language:**  Streamlines the original text for better clarity.
*   **Focus on Benefits:** Emphasizes the value proposition for the user.
*   **SEO-Friendly Formatting:**  Uses markdown to improve parsing by search engines.
*   **Concise Training Loop Description** Improves the training loop description for better understanding.
*   **Added a title to the page:** Added a descriptive title to the whole page to help crawlers.
*   **Added a key feature about RULER** Added some key features about RULER to increase the keywords about rewards.

This improved README is more informative, easier to navigate, and optimized for search engines.  It clearly conveys the value of ART and encourages users to explore the project.