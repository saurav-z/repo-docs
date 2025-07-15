<div align="center">

<a href="https://art.openpipe.ai"><picture>
<img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
</picture></a>

<p align="center">
  <h1>Agent Reinforcement Trainer (ART)</h1>
</p>

<p>
  <a href="https://github.com/openpipe/art">
    <img alt="GitHub Repo" src="https://img.shields.io/github/stars/openpipe/art?style=social" />
  </a>
  <a href="https://pypi.org/project/openpipe-art/">
    <img alt="PyPI Downloads" src="https://img.shields.io/pypi/dm/openpipe-art?color=364fc7&logoColor=364fc7" />
  </a>
    <a href="https://discord.gg/zbBHRUpwf4">
    <img alt="Join Discord" src="https://img.shields.io/badge/Join%20Discord-5865F2?style=plastic&logo=discord&logoColor=white"/>
  </a>
   <a href="https://art.openpipe.ai">
    <img alt="Documentation" src="https://img.shields.io/badge/Documentation-orange?style=plastic&logo=gitbook&logoColor=white"/>
  </a>
</p>

</div>

## Agent Reinforcement Trainer (ART): Train LLM Agents with Ease

ART is an open-source framework that simplifies the training of multi-step LLM agents for real-world tasks using GRPO, enabling you to build more reliable and capable AI agents by leveraging the power of reinforcement learning.  ([See the original repo](https://github.com/OpenPipe/ART))

**Key Features:**

*   **RULER: Zero-Shot Agent Rewards:**  Eliminate reward function engineering with RULER (Relative Universal LLM-Elicited Rewards), which uses an LLM-as-judge to automatically score agent trajectories.
    *   **Faster Development:** Reduce development time by 2-3x by skipping manual reward engineering.
    *   **Versatile:** Works across various tasks without modifications.
    *   **Strong Performance:** Achieves performance that matches or surpasses hand-crafted rewards in most benchmarks.
    *   **Easy Integration:**  Drop-in replacement for manual reward functions.
*   **Ergonomic RL Framework:** ART provides a straightforward way to integrate GRPO into your existing Python applications.
*   **Flexible Training:** Train agents on your local machine or leverage cloud resources for GPU-accelerated training.
*   **Observability:** Integrate with platforms like Weights & Biases (W&B), Langfuse, and OpenPipe for improved debugging and monitoring.
*   **Intelligent Defaults:** Benefit from optimized default settings, or customize parameters to meet your specific needs.

## Getting Started

Install ART using pip:

```bash
pip install openpipe-art
```

## Notebook Examples

Get hands-on experience with ART by exploring these example notebooks:

| Agent Task        | Example Notebook                                                                                                             | Description                               | Comparative Performance                                                                                                                                     |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **2048**          | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)                   | Qwen 2.5 3B learns to play 2048           | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/2048/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/2048/benchmark_2048.ipynb)                            |
| **Temporal Clue** | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/temporal_clue/temporal-clue.ipynb) | Qwen 2.5 7B learns to solve Temporal Clue | [Link coming soon]                                                                                                                                          |
| **Tic Tac Toe**   | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb)     | Qwen 2.5 3B learns to play Tic Tac Toe    | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/tic-tac-toe-local/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/tic_tac_toe/benchmark_tic_tac_toe.ipynb) |
| **Codenames**     | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/codenames/Codenames_RL.ipynb)      | Qwen 2.5 3B learns to play Codenames      | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/codenames/win_rate_over_time.png" width=180> [benchmarks](/examples/codenames/Codenames_RL.ipynb)                            |

## ART in Action

Check out the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) blog post to see how ART was used to train Qwen 2.5 14B to beat o3 at email retrieval!

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## Training Loop Overview

ART employs a client-server architecture for training:

1.  **Inference:**
    *   Your code utilizes the ART client to perform agentic workflows.
    *   Requests are routed to the ART server, which runs the model with the latest LoRA in vLLM.
    *   Each message is stored in a Trajectory.
    *   When a rollout finishes, your code assigns a reward.

2.  **Training:**
    *   Trajectories are grouped and sent to the server after rollouts.
    *   The server trains your model using GRPO, initializing from the latest checkpoint.
    *   The server saves the trained LoRA and loads it into vLLM.
    *   The loop restarts at step 1.

## Supported Models

ART is designed to work with most vLLM/HuggingFace-transformers compatible causal language models, including those supported by [Unsloth](https://docs.unsloth.ai/get-started/all-our-models). For any model compatibility issues, please reach out on [Discord](https://discord.gg/zbBHRUpwf4) or create an issue on [GitHub](https://github.com/openpipe/art/issues).

## Contributing

Contributions are highly encouraged!  See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

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

ART leverages the work of these amazing projects:

*   [Unsloth](https://github.com/unslothai/unsloth)
*   [vLLM](https://github.com/vllm-project/vllm)
*   [trl](https://github.com/huggingface/trl)
*   [torchtune](https://github.com/pytorch/torchtune)
*   [SkyPilot](https://github.com/skypilot-org/skypilot)

We are also grateful to our partners who helped test ART!