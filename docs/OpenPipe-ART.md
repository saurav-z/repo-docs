<div align="center">

<a href="https://art.openpipe.ai"><picture>
<img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
</picture></a>

<p align="center">
  <h1>Agent Reinforcement Trainer (ART)</h1>
</p>

<p>
Train multi-step agents for real-world tasks using GRPO.
</p>

[![PRs-Welcome][contribute-image]][contribute-url]
[![Downloads][downloads-image]][pypi-url]
[![Train Agent](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)

[![Join Discord](https://img.shields.io/badge/Join%20Discord-5865F2?style=plastic&logo=discord&logoColor=white)](https://discord.gg/zbBHRUpwf4)
[![Documentation](https://img.shields.io/badge/Documentation-orange?style=plastic&logo=gitbook&logoColor=white)](https://art.openpipe.ai)

</div>

## Agent Reinforcement Trainer: Revolutionize LLM Agent Training 🚀

ART is an open-source framework that empowers you to train powerful AI agents that learn from experience, eliminating the need for hand-crafted reward functions and simplifying the development process. Discover how you can leverage GRPO (Generalized Reinforcement Learning Optimization) to build more reliable and effective agents. [Learn more about ART on GitHub](https://github.com/OpenPipe/ART).

### Key Features

*   **RULER (Relative Universal LLM-Elicited Rewards):** Automate agent evaluation with an LLM-as-judge, **eliminating the need for manual reward engineering**.
*   **Simplified Training Loop:** Easily integrate GRPO into your Python applications with an ergonomic framework.
*   **Flexible Deployment:** Run training on your local machine or leverage cloud GPUs for efficient training.
*   **Integration with Observability Tools:** Supports integrations with platforms like W&B, Langfuse, and OpenPipe for enhanced debugging and monitoring.
*   **Customization:** Fine-tune training parameters and inference engine configurations to meet your specific project requirements.

## 📏 RULER: Zero-Shot Agent Rewards

**RULER** (Relative Universal LLM-Elicited Rewards) eliminates the need for hand-crafted reward functions by using an LLM-as-judge to automatically score agent trajectories. Simply define your task in the system prompt, and RULER handles the rest—**no labeled data, expert feedback, or reward engineering required**.

✨ **Key Benefits:**

*   **Faster Development:** 2-3x faster development, skipping reward function engineering entirely.
*   **General-Purpose:** Works across any task without modification.
*   **High Performance:** Matches or exceeds hand-crafted rewards in 3/4 benchmarks.
*   **Easy Integration:** Drop-in replacement for manual reward functions.

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

ART is an open-source RL framework that improves agent reliability by allowing LLMs to **learn from experience**. ART provides an ergonomic harness for integrating GRPO into any python application. For a quick hands-on introduction, run one of the notebooks below. When you're ready to learn more, check out the [docs](https://art.openpipe.ai).

## 📒 Notebooks - Train Agents on Real-World Tasks

Explore hands-on examples and benchmark results with ART.

| Agent Task         | Example Notebook                                                                                                             | Description                                     | Comparative Performance                                                                                                                                                                             |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ART•E [RULER]**  | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/art-e/art-e.ipynb)                 | Qwen 2.5 7B learns to search emails using RULER | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/email_agent/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/art-e/art_e/evaluate/display_benchmarks.ipynb) |
| **2048**           | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)                   | Qwen 2.5 3B learns to play 2048                 | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/2048/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/2048/benchmark_2048.ipynb)                            |
| **Temporal Clue**  | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/temporal_clue/temporal-clue.ipynb) | Qwen 2.5 7B learns to solve Temporal Clue       | [Link coming soon]                                                                                                                                                                                  |
| **Tic Tac Toe**    | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb)     | Qwen 2.5 3B learns to play Tic Tac Toe          | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/tic-tac-toe-local/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/tic_tac_toe/benchmark_tic_tac_toe.ipynb) |
| **Codenames**      | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/codenames/Codenames_RL.ipynb)      | Qwen 2.5 3B learns to play Codenames            | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/codenames/win_rate_over_time.png" height="72"> [benchmarks](/examples/codenames/Codenames_RL.ipynb)                            |
| **AutoRL [RULER]** | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/auto_rl.ipynb)                     | Train Qwen 2.5 7B to master any task            | [Link coming soon]                                                                                                                                                                                  |

## 📰 ART News

Stay up-to-date with the latest developments and research from the ART community.

*   🗞️ **[AutoRL: Zero-Data Training for Any Task](https://x.com/mattshumer_/status/1950572449025650733)** - Train custom AI models without labeled data using automatic input generation and RULER evaluation.
*   🗞️ **[RULER: Easy Mode for RL Rewards](https://openpipe.ai/blog/ruler-easy-mode-for-rl-rewards)** is now available for automatic reward generation in reinforcement learning.
*   🗞️ **[ART·E: How We Built an Email Research Agent That Beats o3](https://openpipe.ai/blog/art-e-mail-agent)** demonstrates a Qwen 2.5 14B email agent outperforming OpenAI's o3.
*   🗞️ **[ART Trainer: A New RL Trainer for Agents](https://openpipe.ai/blog/art-trainer)** enables easy training of LLM-based agents using GRPO.

[📖 See all blog posts →](https://openpipe.ai/blog)

## Why Choose ART?

*   **Seamless Integration:** Easily incorporate RL training into existing applications.
*   **Flexible Training:** Train from any machine with the ART client.
*   **Simplified Debugging:** Utilize integrations with platforms like W&B, Langfuse, and OpenPipe.
*   **Intelligent Defaults:** Start quickly with optimized default configurations.

## Installation

Get started with ART by installing the Python package.

```bash
pip install openpipe-art
```

## 🤖 ART•E Agent

Explore a real-world application of ART with the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent), which demonstrates how to train a Qwen 2.5 14B model to surpass o3 in email retrieval.

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## 🔁 Training Loop Explained

ART utilizes a client-server architecture for efficient training.

1.  **Inference:**
    *   Your code uses the ART client to run agent workflows.
    *   Requests are sent to the ART server, which runs the model's latest LoRA in vLLM.
    *   Each message is stored in a Trajectory.
    *   When a rollout is complete, a reward is assigned.

2.  **Training:**
    *   Trajectories are grouped and sent to the server.
    *   The server trains your model using GRPO.
    *   The server saves the newly trained LoRA.
    *   Inference resumes.

This loop continues until a specified number of iterations is reached.

## 🧩 Supported Models

ART is compatible with most vLLM/HuggingFace-transformers causal language models, including those supported by [Unsloth](https://docs.unsloth.ai/get-started/all-our-models). Note that Gemma 3 is currently unsupported. Reach out on [Discord](https://discord.gg/zbBHRUpwf4) or [GitHub](https://github.com/openpipe/art/issues) if you encounter issues with other models.

## 🤝 Contributing

ART is actively developed, and contributions are welcome! See the [CONTRIBUTING.md](CONTRIBUTING.md) file for details.

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

ART builds upon the work of many in the open-source RL community. Special thanks to the authors of the following projects:

*   [Unsloth](https://github.com/unslothai/unsloth)
*   [vLLM](https://github.com/vllm-project/vllm)
*   [trl](https://github.com/huggingface/trl)
*   [torchtune](https://github.com/pytorch/torchtune)
*   [SkyPilot](https://github.com/skypilot-org/skypilot)

We also thank our partners for helping us test ART.

[pypi-url]: https://pypi.org/project/openpipe-art/
[contribute-url]: https://github.com/openpipe/art/blob/main/CONTRIBUTING.md
[contribute-image]: https://img.shields.io/badge/PRs-welcome-blue.svg
[downloads-image]: https://img.shields.io/pypi/dm/openpipe-art?color=364fc7&logoColor=364fc7
```
Key improvements and explanations:

*   **SEO-Optimized Title and Introduction:**  The title is made more descriptive ("Agent Reinforcement Trainer"). The introductory sentence includes a key phrase "Revolutionize LLM agent training" to capture search intent. It also links back to the repo immediately.
*   **Clear Headings:** Uses standard markdown headings (##, ###) for better organization and readability.
*   **Bulleted Key Features:** Uses bullet points for a clear, concise presentation of core benefits.
*   **Emphasis on Benefits:**  Highlights *why* users should care with clear benefits of using the framework.
*   **Clearer Explanations:** Improved descriptions, especially for RULER and the training loop.
*   **Concise Language:** Streamlined the text for better readability and understanding.
*   **Call to Actions:** Encourages users to explore notebooks, documentation, and blog posts with explicit calls to action.
*   **Improved Formatting:**  Consistent markdown formatting (bold, italics, code blocks).
*   **Contextual Links:** Links are placed within the flow of the text, guiding the user to related resources.
*   **Comprehensive Coverage:**  Includes all the original content but organizes it more effectively.
*   **Keywords:** Includes relevant keywords like "Reinforcement Learning", "LLM", "Agent", "Training", "GRPO", "RULER" to help with searchability.
*   **Conciseness:** While being comprehensive, it cuts out unnecessary words to make it more appealing.