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

## Agent Reinforcement Trainer (ART): Revolutionize LLM Agent Training with OpenPipe's Framework

**ART empowers developers to easily train and refine LLM agents for complex tasks by leveraging the power of Reinforcement Learning from Human Preferences (GRPO), resulting in more reliable and effective AI agents.** Explore the official [ART repository](https://github.com/OpenPipe/ART) for the source code and more information.

**Key Features:**

*   🚀 **Rapid Development:**  Accelerate your agent training workflow with RULER, saving you valuable time and resources.
*   🧠 **Zero-Shot Rewards with RULER:** Use the power of LLMs to generate automatic scoring for agent actions, eliminating the need for handcrafted reward functions, labels, or expert feedback.
*   🛠️ **Simplified Integration:** Seamlessly integrate ART into your existing Python applications, and easily deploy using the provided server and client tools.
*   ⚙️ **Customization & Optimization:** Customize training parameters, leverage intelligent defaults optimized for efficient training.
*   🌍 **Versatile & Adaptable:** ART works across a wide range of tasks without modification, ensuring broad applicability.
*   💻 **Flexible Training:** Train your agents from any local client or leverage hosted platforms.
*   ✅ **Observability Integrations:** Monitor your agent's performance using W&B, Langfuse, and OpenPipe.

## 📏 RULER: Zero-Shot Agent Rewards

**RULER (Relative Universal LLM-Elicited Rewards)** eliminates the need for hand-crafted reward functions by using an LLM-as-judge to automatically score agent trajectories. Simply define your task in the system prompt, and RULER handles the rest—**no labeled data, expert feedback, or reward engineering required**.

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

## ART Overview

ART is an open-source RL framework that improves agent reliability by allowing LLMs to **learn from experience**. ART provides an ergonomic harness for integrating GRPO into any python application. For a quick hands-on introduction, run one of the notebooks below. When you're ready to learn more, check out the [docs](https://art.openpipe.ai).

## 📒 Example Notebooks

| Agent Task        | Example Notebook                                                                                                             | Description                               | Comparative Performance                                                                                                                                     |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ART•E [RULER]**         | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/art-e/art-e.ipynb)               | Qwen 2.5 7B learns to search emails using RULER     | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/email_agent/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/art-e/art_e/evaluate/display_benchmarks.ipynb)                                                                                                                                          |
| **2048**          | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)                   | Qwen 2.5 3B learns to play 2048           | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/2048/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/2048/benchmark_2048.ipynb)                            |
| **Temporal Clue** | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/temporal_clue/temporal-clue.ipynb) | Qwen 2.5 7B learns to solve Temporal Clue | [Link coming soon]                                                                                                                                          |
| **Tic Tac Toe**   | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb)     | Qwen 2.5 3B learns to play Tic Tac Toe    | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/tic-tac-toe-local/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/tic_tac_toe/benchmark_tic_tac_toe.ipynb) |
| **Codenames**     | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/codenames/Codenames_RL.ipynb)      | Qwen 2.5 3B learns to play Codenames      | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/codenames/win_rate_over_time.png" height="72"> [benchmarks](/examples/codenames/Codenames_RL.ipynb)                            |

## Why ART?

- ART provides convenient wrappers for introducing RL training into **existing applications**. We abstract the training server into a modular service that your code doesn't need to interface with.
- **Train from anywhere.** Run the ART client on your laptop and let the ART server kick off an ephemeral GPU-enabled environment, or run on a local GPU.
- Integrations with hosted platforms like W&B, Langfuse, and OpenPipe provide flexible observability and **simplify debugging**.
- ART is customizable with **intelligent defaults**. You can configure training parameters and inference engine configurations to meet specific needs, or take advantage of the defaults, which have been optimized for training efficiency and stability.

## Installation

ART agents can be trained from any client machine that runs python. To add to an existing project, run this command:

```
pip install openpipe-art
```

## 🤖 ART•E Agent

Curious about how to use ART for a real-world task? Check out the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) blog post, where we detail how we trained Qwen 2.5 14B to beat o3 at email retrieval!

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## 🔁 Training Loop Overview

ART's functionality is divided into a **client** and a **server**. The OpenAI-compatible client is responsible for interfacing between ART and your codebase. Using the client, you can pass messages and get completions from your LLM as it improves. The server runs independently on any machine with a GPU. It abstracts away the complexity of the inference and training portions of the RL loop while allowing for some custom configuration. An outline of the training loop is shown below:

1.  **Inference**

    1.  Your code uses the ART client to perform an agentic workflow (usually executing several rollouts in parallel to gather data faster).
    2.  Completion requests are routed to the ART server, which runs the model's latest LoRA in vLLM.
    3.  As the agent executes, each `system`, `user`, and `assistant` message is stored in a Trajectory.
    4.  When a rollout finishes, your code assigns a `reward` to its Trajectory, indicating the performance of the LLM.
2.  **Training**
    1.  When each rollout has finished, Trajectories are grouped and sent to the server. Inference is blocked while training executes.
    2.  The server trains your model using GRPO, initializing from the latest checkpoint (or an empty LoRA on the first iteration).
    3.  The server saves the newly trained LoRA to a local directory and loads it into vLLM.
    4.  Inference is unblocked and the loop resumes at step 1.

This training loop runs until a specified number of inference and training iterations have completed.

## 🧩 Supported Models

ART should work with most vLLM/HuggingFace-transformers compatible causal language models, or at least the ones supported by [Unsloth](https://docs.unsloth.ai/get-started/all-our-models). Gemma 3 does not appear to be supported for the time being. If any other model isn't working for you, please let us know on [Discord](https://discord.gg/zbBHRUpwf4) or open an issue on [GitHub](https://github.com/openpipe/art/issues)!

## 🤝 Contributing

ART is in active development, and contributions are most welcome! Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for more information.

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

ART stands on the shoulders of giants. While we owe many of the ideas and early experiments that led to ART's development to the open source RL community at large, we're especially grateful to the authors of the following projects:

-   [Unsloth](https://github.com/unslothai/unsloth)
-   [vLLM](https://github.com/vllm-project/vllm)
-   [trl](https://github.com/huggingface/trl)
-   [torchtune](https://github.com/pytorch/torchtune)
-   [SkyPilot](https://github.com/skypilot-org/skypilot)

Finally, thank you to our partners who've helped us test ART in the wild! We're excited to see what you all build with it.

[pypi-url]: https://pypi.org/project/openpipe-art/
[contribute-url]: https://github.com/openpipe/art/blob/main/CONTRIBUTING.md
[contribute-image]: https://img.shields.io/badge/PRs-welcome-blue.svg
[downloads-image]: https://img.shields.io/pypi/dm/openpipe-art?color=364fc7&logoColor=364fc7
```
Key improvements and SEO considerations:

*   **Clear Title & Hook:** Starts with a compelling one-sentence description that highlights ART's core benefit.
*   **Keyword Optimization:** Includes relevant keywords throughout (e.g., "LLM agent training," "reinforcement learning," "GRPO," "open source," "RULER," "zero-shot rewards").
*   **Structured Headings:** Uses clear headings to organize content for readability and SEO.
*   **Bulleted Lists:**  Highlights key features for easy scanning and understanding.
*   **Emphasis on Benefits:**  Focuses on the value proposition of ART (e.g., faster development, ease of use, improved agent performance).
*   **Internal Linking:** Includes links to the project's GitHub repository, documentation, and relevant examples to encourage exploration.
*   **Call to Actions:** Encourages users to join the Discord, learn more, and contribute.
*   **Installation Guide:** Provides clear instructions for installing the package.
*   **Simplified Overview:** Clarified training loop for better comprehension.
*   **Model Support & Contact:** Highlights supported models and provides a way to get support.
*   **Citation, License, and Credits:** Includes essential information for ethical use and attribution.