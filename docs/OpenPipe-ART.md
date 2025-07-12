<div align="center">

<a href="https://art.openpipe.ai"><picture>
<img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
</picture></a>

<p align="center">
  <h1>Agent Reinforcement Trainer (ART)</h1>
</p>
</div>

## Train and Enhance LLM Agents with ART

ART is an open-source framework that empowers you to train and refine large language model (LLM) agents to perform complex, real-world tasks with the power of reinforcement learning.  Leverage the capabilities of GRPO, and eliminate the need for complex reward functions with **RULER**, our LLM-as-judge solution. Explore ART on [GitHub](https://github.com/OpenPipe/ART).

**Key Features:**

*   **RULER: Zero-Shot Agent Rewards:** Automate reward scoring using LLMs, eliminating manual reward function engineering.
    *   2-3x faster development compared to hand-crafted reward functions.
    *   Works across any task without modification.
    *   Matches or exceeds hand-crafted rewards in 3/4 benchmarks.
    *   Easy integration with a drop-in replacement for manual reward functions.
*   **Ergonomic GRPO Integration:** ART provides an accessible interface for incorporating reinforcement learning into your applications.
*   **Flexible Training:** Train agents from any location and platform. ART seamlessly integrates with hosted platforms like W&B, Langfuse, and OpenPipe, or you can run it locally.
*   **Intelligent Defaults:** Configure training parameters to meet your specific needs, or leverage optimized default settings.
*   **Modular Architecture:** ART's client-server setup simplifies integrating RL into your existing codebase.

## Quickstart Notebooks

Experiment with ART using our hands-on notebooks:

| Agent Task        | Example Notebook                                                                                                             | Description                               | Comparative Performance                                                                                                                                     |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **2048**          | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)                   | Qwen 2.5 3B learns to play 2048           | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/2048/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/2048/benchmark_2048.ipynb)                            |
| **Temporal Clue** | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/temporal_clue/temporal-clue.ipynb) | Qwen 2.5 7B learns to solve Temporal Clue | [Link coming soon]                                                                                                                                          |
| **Tic Tac Toe**   | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb)     | Qwen 2.5 3B learns to play Tic Tac Toe    | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/tic-tac-toe-local/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/tic_tac_toe/benchmark_tic_tac_toe.ipynb) |
| **Codenames**     | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/codenames/Codenames_RL.ipynb)      | Qwen 2.5 3B learns to play Codenames      | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/codenames/win_rate_over_time.png" height="72"> [benchmarks](/examples/codenames/Codenames_RL.ipynb)                            |

## Why Choose ART?

*   **Simplified Integration:** ART is designed to seamlessly integrate RL training into your existing applications.
*   **Versatile Training Environment:** Run the ART client locally and leverage remote GPU resources.
*   **Enhanced Observability:** Integrate with popular platforms like W&B, Langfuse, and OpenPipe for easier debugging.
*   **Optimized Defaults:** Quickly get started with intelligent default settings, or customize to meet specific project requirements.

## Installation

Get started by installing the ART client:

```bash
pip install openpipe-art
```

## ART•E Agent: Real-World Application

See how ART was used to train an email agent! Check out the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) blog post to learn how we trained Qwen 2.5 14B to outperform o3 at email retrieval.

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## Training Loop Overview

ART employs a client-server architecture:

1.  **Inference:**
    *   Your code uses the ART client to perform an agentic workflow.
    *   Requests are routed to the ART server, running the model's latest LoRA in vLLM.
    *   Each message is stored in a Trajectory.
    *   Rollouts are assigned a reward.
2.  **Training:**
    *   Trajectories are grouped and sent to the server.
    *   The server trains your model using GRPO.
    *   The server saves the LoRA and loads it into vLLM.
    *   The loop resumes with inference.

## Supported Models

ART supports most vLLM/HuggingFace-transformers compatible causal language models, including those supported by [Unsloth](https://docs.unsloth.ai/get-started/all-our-models).  For any model compatibility issues, please let us know on [Discord](https://discord.gg/zbBHRUpwf4) or open an issue on [GitHub](https://github.com/openpipe/art/issues).

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

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

[Apache-2.0 License](LICENSE)

## Acknowledgements

ART builds upon the work of several open-source projects:

*   [Unsloth](https://github.com/unslothai/unsloth)
*   [vLLM](https://github.com/vllm-project/vllm)
*   [trl](https://github.com/huggingface/trl)
*   [torchtune](https://github.com/pytorch/torchtune)
*   [SkyPilot](https://github.com/skypilot-org/skypilot)

Thank you to our partners for helping us test ART.