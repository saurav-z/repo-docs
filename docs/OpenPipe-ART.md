<div align="center">
<a href="https://art.openpipe.ai">
<picture>
  <img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
</picture>
</a>
<p align="center">
  <h1>Agent Reinforcement Trainer (ART)</h1>
</p>
</div>

**Train powerful, multi-step LLM agents for complex tasks with Agent Reinforcement Trainer (ART), an open-source framework for building reliable AI agents.** ART allows you to quickly develop and deploy AI agents using Reinforcement Learning from Human Feedback (RLHF) techniques. For more details, please visit the original repository: [OpenPipe/ART](https://github.com/OpenPipe/ART).

## Key Features:

*   **🤖 Train Agents with RL:** Quickly build and deploy AI agents for a variety of real-world tasks using advanced reinforcement learning techniques.
*   **📏 RULER for Zero-Shot Reward Engineering:** Leverage RULER (Relative Universal LLM-Elicited Rewards) to automatically score agent trajectories with no need for hand-crafted reward functions, expert feedback, or labeled data.
    *   **2-3x faster development**
    *   **General-purpose** - Works across any task without modification
    *   **Strong performance** - Matches or exceeds hand-crafted rewards in 3/4 benchmarks
    *   **Easy integration** - Drop-in replacement for manual reward functions
*   **🚀 Flexible and Customizable:** ART provides an ergonomic harness for integrating GRPO into any python application. You can configure training parameters and inference engine configurations to meet specific needs, or take advantage of the defaults, which have been optimized for training efficiency and stability.
*   **💻 Train from Anywhere:** Run the ART client on your laptop, or use a local or remote GPU.
*   **📊 Integrations:** Easily integrate with platforms like W&B, Langfuse, and OpenPipe for flexible observability and simplified debugging.
*   **📒 Ready-to-Use Notebooks:** Get started quickly with pre-built notebooks showcasing ART's capabilities across various agent tasks.

## Getting Started with ART

### Installation

Install the ART client to integrate into your existing project.

```bash
pip install openpipe-art
```

### Examples and Benchmarks

Explore ART's capabilities with example notebooks:

| Agent Task        | Example Notebook                                                                                                             | Description                               | Comparative Performance                                                                                                                                     |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ART•E [RULER]**         | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/art-e/art-e.ipynb)               | Qwen 2.5 7B learns to search emails using RULER     | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/email_agent/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/art-e/art_e/evaluate/display_benchmarks.ipynb)                                                                                                                                          |
| **2048**          | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)                   | Qwen 2.5 3B learns to play 2048           | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/2048/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/2048/benchmark_2048.ipynb)                            |
| **Temporal Clue** | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/temporal_clue/temporal-clue.ipynb) | Qwen 2.5 7B learns to solve Temporal Clue | [Link coming soon]                                                                                                                                          |
| **Tic Tac Toe**   | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb)     | Qwen 2.5 3B learns to play Tic Tac Toe    | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/tic-tac-toe-local/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/tic_tac_toe/benchmark_tic_tac_toe.ipynb) |
| **Codenames**     | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/codenames/Codenames_RL.ipynb)      | Qwen 2.5 3B learns to play Codenames      | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/codenames/win_rate_over_time.png" height="72"> [benchmarks](/examples/codenames/Codenames_RL.ipynb)                            |

### ART•E Agent

Learn more about ART with the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) blog post.

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## ART Training Loop Overview

ART leverages a client-server architecture to streamline the training process:

1.  **Inference (Client-Side):**
    *   Your code uses the ART client to perform agentic workflows.
    *   Completion requests are routed to the ART server, running the model's latest LoRA in vLLM.
    *   `system`, `user`, and `assistant` messages are stored in a Trajectory.
    *   When a rollout finishes, your code assigns a `reward` to its Trajectory.
2.  **Training (Server-Side):**
    *   Trajectories are grouped and sent to the server. Inference is blocked while training executes.
    *   The server trains your model using GRPO.
    *   The server saves the newly trained LoRA and loads it into vLLM.
    *   Inference is unblocked and the loop resumes at step 1.

## Supported Models

ART supports most vLLM/HuggingFace-transformers compatible causal language models, or at least the ones supported by [Unsloth](https://docs.unsloth.ai/get-started/all-our-models). Gemma 3 does not appear to be supported for the time being.

## Contributing

Contributions are welcome! Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for more information.

## Citation

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

## License

This project is licensed under the [Apache-2.0 License](LICENSE).

## Acknowledgements

ART is built upon the work of several open-source projects. Special thanks to:

*   [Unsloth](https://github.com/unslothai/unsloth)
*   [vLLM](https://github.com/vllm-project/vllm)
*   [trl](https://github.com/huggingface/trl)
*   [torchtune](https://github.com/pytorch/torchtune)
*   [SkyPilot](https://github.com/skypilot-org/skypilot)

Thank you to our partners who've helped us test ART in the wild!