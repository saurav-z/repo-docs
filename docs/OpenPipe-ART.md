<div align="center">

<a href="https://art.openpipe.ai"><picture>
<img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
</picture></a>

<p align="center">
  <h1>Agent Reinforcement Trainer (ART): Train LLM Agents for Real-World Tasks</h1>
</p>

<p>
ART empowers you to train sophisticated multi-step agents, providing an ergonomic harness for integrating GRPO into any python application.
</p>

[![PRs-Welcome][contribute-image]][contribute-url]
[![Downloads][downloads-image]][pypi-url]
[![Train Agent](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)

[![Join Discord](https://img.shields.io/badge/Join%20Discord-5865F2?style=plastic&logo=discord&logoColor=white)](https://discord.gg/zbBHRUpwf4)
[![Documentation](https://img.shields.io/badge/Documentation-orange?style=plastic&logo=gitbook&logoColor=white)](https://art.openpipe.ai)

</div>

## Key Features of Agent Reinforcement Trainer (ART)

ART is an open-source reinforcement learning framework that simplifies training LLM-powered agents.  

*   **Effortless Reward Engineering with RULER:** Utilizes RULER (Relative Universal LLM-Elicited Rewards) to automatically score agent trajectories using an LLM-as-judge, eliminating the need for manual reward functions.
*   **Accelerated Development:** RULER can speed up development by 2-3x compared to traditional methods by eliminating the time spent creating custom reward functions.
*   **Broad Applicability:** Works across various tasks without modification, offering a general-purpose solution for agent training.
*   **Strong Performance:** Achieves competitive results, matching or exceeding hand-crafted rewards in many benchmarks.
*   **Easy Integration:** Provides a simple drop-in replacement for your existing reward functions.
*   **Modular Architecture:** Separates the training loop into a client and server for flexible deployment and management.
*   **Flexible Training:** Supports training from any client machine with a GPU or by using ephemeral GPU-enabled environments for scalable training.
*   **Observability & Debugging:** Integrates with platforms like W&B, Langfuse, and OpenPipe to simplify debugging and monitoring.
*   **Optimized Defaults:** Comes with intelligent defaults that are optimized for training efficiency and stability, providing a smooth user experience.

[**Check out the ART Repo on GitHub!**](https://github.com/OpenPipe/ART)

## 📏 RULER: Zero-Shot Agent Rewards

**RULER** (Relative Universal LLM-Elicited Rewards) eliminates the need for hand-crafted reward functions by using an LLM-as-judge to automatically score agent trajectories. Simply define your task in the system prompt, and RULER handles the rest—**no labeled data, expert feedback, or reward engineering required**.

✨ **Key Benefits:**
- **2-3x faster development** - Skip reward function engineering entirely
- **General-purpose** - Works across any task without modification
- **Strong performance** - Matches or exceeds hand-crafted rewards in 3/4 benchmarks
- **Easy integration** - Drop-in replacement for manual reward functions

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

## 📒 Example Notebooks: Train Agents on Diverse Tasks

Quickly get started with ART using our example notebooks.  Each notebook demonstrates training an agent for a specific task, along with performance benchmarks.

| Agent Task        | Example Notebook                                                                                                             | Description                               | Comparative Performance                                                                                                                                     |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ART•E [RULER]**         | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/art-e/art-e.ipynb)               | Qwen 2.5 7B learns to search emails using RULER     | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/email_agent/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/art-e/art_e/evaluate/display_benchmarks.ipynb)                                                                                                                                          |
| **2048**          | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)                   | Qwen 2.5 3B learns to play 2048           | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/2048/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/2048/benchmark_2048.ipynb)                            |
| **Temporal Clue** | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/temporal_clue/temporal-clue.ipynb) | Qwen 2.5 7B learns to solve Temporal Clue | [Link coming soon]                                                                                                                                          |
| **Tic Tac Toe**   | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb)     | Qwen 2.5 3B learns to play Tic Tac Toe    | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/tic-tac-toe-local/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/tic_tac_toe/benchmark_tic_tac_toe.ipynb) |
| **Codenames**     | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/codenames/Codenames_RL.ipynb)      | Qwen 2.5 3B learns to play Codenames      | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/codenames/win_rate_over_time.png" height="72"> [benchmarks](/examples/codenames/Codenames_RL.ipynb)                            |

## Why Use ART?

ART simplifies the integration of Reinforcement Learning into your existing applications. 

*   **Simplified Integration:** ART provides convenient wrappers to add RL training to your existing applications.
*   **Flexible Training Environments:** Train your agents locally or leverage cloud-based GPU environments for scalable training.
*   **Enhanced Observability:** Integrates with popular platforms for easy monitoring and debugging.
*   **Intelligent Defaults:** Enjoy optimized default settings that promote training efficiency and stability.

## Installation

Install the ART library using pip:

```bash
pip install openpipe-art
```

## 🤖 ART•E Agent: Real-World Example

Learn how to apply ART to a practical task: training an agent to retrieve emails. Check out the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) blog post for details on training a Qwen 2.5 14B model to beat o3 at email retrieval!

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## 🔁 Training Loop Overview

ART utilizes a client-server architecture for efficient training. Here's an outline of the training loop:

1.  **Inference:**
    1.  Your code interacts with the ART client to execute agentic workflows.
    2.  Completion requests are routed to the ART server, which runs the model's latest LoRA in vLLM.
    3.  Each interaction is stored in a Trajectory.
    4.  When a rollout is complete, the trajectory is assigned a reward.

2.  **Training:**
    1.  Trajectories are grouped and sent to the server for training.
    2.  The server trains your model using GRPO, starting from the latest checkpoint.
    3.  The server saves the trained LoRA and loads it into vLLM.
    4.  Inference resumes.

The loop continues until the specified number of iterations is complete.

## 🧩 Supported Models

ART is compatible with a wide range of causal language models that are supported by vLLM/HuggingFace-transformers, including models supported by [Unsloth](https://docs.unsloth.ai/get-started/all-our-models). If you encounter issues with a specific model, please report them on [Discord](https://discord.gg/zbBHRUpwf4) or [GitHub](https://github.com/openpipe/art/issues).

## 🤝 Contributing

Contributions to ART are highly encouraged! Please consult the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines.

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

ART leverages the work of numerous contributors and projects within the open-source RL community.  Special thanks to the authors of:

-   [Unsloth](https://github.com/unslothai/unsloth)
-   [vLLM](https://github.com/vllm-project/vllm)
-   [trl](https://github.com/huggingface/trl)
-   [torchtune](https://github.com/pytorch/torchtune)
-   [SkyPilot](https://github.com/skypilot-org/skypilot)

Thank you to our partners for helping test ART!

[pypi-url]: https://pypi.org/project/openpipe-art/
[contribute-url]: https://github.com/openpipe/art/blob/main/CONTRIBUTING.md
[contribute-image]: https://img.shields.io/badge/PRs-welcome-blue.svg
[downloads-image]: https://img.shields.io/pypi/dm/openpipe-art?color=364fc7&logoColor=364fc7
```
Key improvements and SEO considerations:

*   **Clear Title and Hook:** The title is optimized and includes the keyword "Agent Reinforcement Trainer (ART)" along with a one-sentence hook to grab attention.
*   **Keyword Optimization:** Keywords like "Agent Reinforcement Trainer," "LLM Agents," "Reinforcement Learning," and "RULER" are used throughout the README.
*   **Headings and Structure:** Uses clear headings and subheadings to improve readability and organization, which is good for SEO.
*   **Bulleted Lists:** Key features are presented in bulleted lists for easy scanning and understanding.
*   **Concise Language:** Uses clear, concise language to describe the functionality and benefits.
*   **Call to Action:** Includes clear calls to action such as "Check out the ART Repo on GitHub!"
*   **Link to Original Repo:**  A link back to the original GitHub repo is included.
*   **Alt Text for Images:** Added alt text to images for better accessibility and SEO.
*   **Expanded Overview:** The "Why ART?" section and Training Loop section have been expanded to give more context to the user.
*   **Focus on Benefits:** The benefits of using ART are emphasized throughout the document.
*   **Consistent Formatting:**  Uses consistent Markdown formatting for readability.
*   **Contextual Benchmarks:**  Added to the example notebooks section to draw attention to the agent performance.
*   **More comprehensive list of credits.**