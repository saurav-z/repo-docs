<div align="center">

<a href="https://art.openpipe.ai"><picture>
<img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
</picture></a>

<p align="center">
  <h1>Agent Reinforcement Trainer (ART)</h1>
</p>

<p>
  Train powerful, multi-step AI agents effortlessly with ART and GRPO.
</p>

[![PRs-Welcome][contribute-image]][contribute-url]
[![Downloads][downloads-image]][pypi-url]
[![Train Agent](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)

[![Join Discord](https://img.shields.io/badge/Join%20Discord-5865F2?style=plastic&logo=discord&logoColor=white)](https://discord.gg/zbBHRUpwf4)
[![Documentation](https://img.shields.io/badge/Documentation-orange?style=plastic&logo=gitbook&logoColor=white)](https://art.openpipe.ai)

</div>

## Supercharge Your AI Agents with Agent Reinforcement Trainer (ART)

ART is an open-source framework designed to simplify and accelerate the training of sophisticated, multi-step AI agents.  Using Gradient Guided Policy Optimization (GRPO), ART empowers your agents to learn from experience, leading to improved reliability and performance.  Ready to revolutionize your AI agent training?  [Explore ART on GitHub](https://github.com/OpenPipe/ART).

## Key Features

*   **Effortless Reward Engineering:**  Leverage **RULER** (Relative Universal LLM-Elicited Rewards) to eliminate the need for hand-crafted reward functions.  Simply define your task, and let RULER handle the scoring using an LLM-as-judge.
*   **Accelerated Development:**  RULER can lead to **2-3x faster development**, saving you time and resources.
*   **General-Purpose Applicability:** ART works across a wide range of tasks without modification, making it a versatile solution for diverse agent training needs.
*   **High Performance:** Achieve results that match or exceed hand-crafted rewards in various benchmarks.
*   **Seamless Integration:**  ART is designed for easy integration into existing applications.
*   **Flexible Training Loop:** ART's modular architecture supports both client and server components for flexible deployment.
*   **Extensive Model Support:** ART supports vLLM/HuggingFace-transformers compatible causal language models.

## 📏 RULER: Zero-Shot Agent Rewards

**RULER** (Relative Universal LLM-Elicited Rewards) uses an LLM-as-judge to automatically score agent trajectories, eliminating the need for hand-crafted reward functions. Simply define your task in the system prompt, and RULER handles the rest—**no labeled data, expert feedback, or reward engineering required**.

✨ **Key Benefits:**

-   **2-3x faster development** - Skip reward function engineering entirely
-   **General-purpose** - Works across any task without modification
-   **Strong performance** - Matches or exceeds hand-crafted rewards in 3/4 benchmarks
-   **Easy integration** - Drop-in replacement for manual reward functions

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

## 📒 Notebooks: Train Your Agents Now!

Get started quickly with these interactive Colab notebooks.

| Agent Task         | Example Notebook                                                                                                             | Description                                     | Comparative Performance                                                                                                                                                                             |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ART•E [RULER]**  | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/art-e/art-e.ipynb)                 | Qwen 2.5 7B learns to search emails using RULER | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/email_agent/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/art-e/art_e/evaluate/display_benchmarks.ipynb) |
| **2048**           | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)                   | Qwen 2.5 3B learns to play 2048                 | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/2048/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/2048/benchmark_2048.ipynb)                            |
| **Temporal Clue**  | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/temporal_clue/temporal-clue.ipynb) | Qwen 2.5 7B learns to solve Temporal Clue       | [Link coming soon]                                                                                                                                                                                  |
| **Tic Tac Toe**    | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb)     | Qwen 2.5 3B learns to play Tic Tac Toe          | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/tic-tac-toe-local/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/tic_tac_toe/benchmark_tic_tac_toe.ipynb) |
| **Codenames**      | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/codenames/Codenames_RL.ipynb)      | Qwen 2.5 3B learns to play Codenames            | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/codenames/win_rate_over_time.png" height="72"> [benchmarks](/examples/codenames/Codenames_RL.ipynb)                            |
| **AutoRL [RULER]** | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/auto_rl.ipynb)                     | Train Qwen 2.5 7B to master any task            | [Link coming soon]                                                                                                                                                                                  |

## 📰 ART News and Updates

Stay up-to-date with the latest developments in ART and the field of agent training.

-   🗞️ **[AutoRL: Zero-Data Training for Any Task](https://x.com/mattshumer_/status/1950572449025650733)** - Train custom AI models without labeled data using automatic input generation and RULER evaluation.
-   🗞️ **[RULER: Easy Mode for RL Rewards](https://openpipe.ai/blog/ruler-easy-mode-for-rl-rewards)** is now available for automatic reward generation in reinforcement learning.
-   🗞️ **[ART·E: How We Built an Email Research Agent That Beats o3](https://openpipe.ai/blog/art-e-mail-agent)** demonstrates a Qwen 2.5 14B email agent outperforming OpenAI's o3.
-   🗞️ **[ART Trainer: A New RL Trainer for Agents](https://openpipe.ai/blog/art-trainer)** enables easy training of LLM-based agents using GRPO.

[📖 See all blog posts →](https://openpipe.ai/blog)

## Why Choose ART?

*   **Simplified Integration:**  Easily integrate RL training into your existing applications with ART's convenient wrappers.
*   **Flexible Deployment:** Train agents from your laptop or leverage GPU-enabled environments, either locally or remotely.
*   **Enhanced Observability:**  Integrate with platforms like W&B, Langfuse, and OpenPipe for streamlined debugging and monitoring.
*   **Intelligent Defaults & Customization:**  Benefit from optimized training parameters and inference engine configurations, or tailor settings to meet specific needs.

## Installation

Get started quickly by installing the ART Python package:

```bash
pip install openpipe-art
```

## 🤖 ART•E Agent

Explore the practical application of ART with the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent), and learn how to train a Qwen 2.5 14B model to outperform o3 in email retrieval.

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## 🔁 Training Loop: How ART Works

ART employs a client-server architecture:

1.  **Inference:** The ART client handles agentic workflows, sending requests to the ART server, which runs the model's latest LoRA in vLLM. Trajectories are created and stored.
2.  **Training:**  Once a rollout finishes, your code provides a reward to a Trajectory. Trajectories are then sent to the server, which trains the model using GRPO and saves the updated LoRA.

This loop continues until the specified number of iterations is reached.

## 🧩 Supported Models

ART is designed to work with most vLLM/HuggingFace-transformers compatible causal language models. Check the [Unsloth](https://docs.unsloth.ai/get-started/all-our-models) documentation for a list of supported models. For any compatibility issues, please reach out on [Discord](https://discord.gg/zbBHRUpwf4) or open an issue on [GitHub](https://github.com/openpipe/art/issues)!

## 🤝 Contributing

We welcome contributions to ART!  See our [CONTRIBUTING.md](CONTRIBUTING.md) file for details.

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

ART is built upon the work of many contributors in the open-source RL community.  We are particularly grateful to the authors of:

-   [Unsloth](https://github.com/unslothai/unsloth)
-   [vLLM](https://github.com/vllm-project/vllm)
-   [trl](https://github.com/huggingface/trl)
-   [torchtune](https://github.com/pytorch/torchtune)
-   [SkyPilot](https://github.com/skypilot-org/skypilot)

Special thanks to our partners for their invaluable support in testing ART.

[pypi-url]: https://pypi.org/project/openpipe-art/
[contribute-url]: https://github.com/openpipe/art/blob/main/CONTRIBUTING.md
[contribute-image]: https://img.shields.io/badge/PRs-welcome-blue.svg
[downloads-image]: https://img.shields.io/pypi/dm/openpipe-art?color=364fc7&logoColor=364fc7
```
Key improvements and SEO considerations:

*   **Clear, Concise Hook:**  Starts with a strong, action-oriented sentence.
*   **Keyword Optimization:** Uses relevant keywords like "AI agents," "reinforcement learning," "GRPO," and "LLM" throughout the text.
*   **Structured Headings:** Uses clear headings (H1, H2) and bullet points to organize information and improve readability for both users and search engines.
*   **Concise Language:**  Avoids jargon and overly technical terms.
*   **Benefit-Oriented:**  Highlights the advantages of using ART.
*   **Call to Action:** Encourages users to explore the repository and try the Colab notebooks.
*   **Comprehensive Summary:** Includes all critical information from the original README.
*   **Internal Links:**  Includes links to relevant sections within the README.
*   **External Links:** Keeps all the original links and adds SEO-friendly anchor text.
*   **Clear Installation Instructions.**
*   **Emphasis on RULER**: Highlights the RULER feature.
*   **Model Support and Contribution Section:** Provides information about model compatibility and encourages contributions.