<div align="center">
  <a href="https://art.openpipe.ai">
    <picture>
      <img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
    </picture>
  </a>
  <h1>Agent Reinforcement Trainer (ART)</h1>
  <p>
    <b>Supercharge your LLM agents with ART, an open-source framework for training multi-step agents using GRPO and LLM-as-judge rewards.</b>
  </p>

  [![PRs-Welcome][contribute-image]][contribute-url]
  [![PyPI version](https://img.shields.io/pypi/v/openpipe-art?color=364fc7)][pypi-url]
  [![Train Agent](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/2048/2048.ipynb)

  [![Join Discord](https://img.shields.io/badge/Join%20Discord-5865F2?style=plastic&logo=discord&logoColor=white)](https://discord.gg/zbBHRUpwf4)
  [![Documentation](https://img.shields.io/badge/Documentation-orange?style=plastic&logo=gitbook&logoColor=white)](https://art.openpipe.ai)
</div>

## Key Features of ART

*   **Zero-Shot Reward Engineering with RULER:** Leverage LLMs to automatically score agent performance, eliminating the need for handcrafted reward functions.
    *   **2-3x Faster Development:** Significantly reduce development time by skipping reward function engineering.
    *   **General-Purpose:** Works across diverse tasks without modification.
    *   **Superior Performance:** Achieves or surpasses hand-crafted rewards in many benchmarks.
    *   **Simple Integration:** Easily integrates into existing projects as a drop-in replacement for manual reward functions.

```python
# Before: Hours of reward engineering
def complex_reward_function(trajectory):
    # 50+ lines of careful scoring logic...
    pass

# After: One line with RULER
judged_group = await ruler_score_group(group, "openai/o3")
```

*   **Open-Source RL Framework for LLMs:**  ART simplifies integrating GRPO into your Python applications.
*   **Simplified Training Loop:**
    *   ART is divided into a client and server, abstracting away the complexity of the inference and training portions of the RL loop.
    *   ART can be trained from anywhere, either on a local GPU or a remote environment.
*   **Flexible Observability and Debugging:** Integrations with platforms such as W&B, Langfuse, and OpenPipe provide easy observability.
*   **Customizable with Intelligent Defaults:** Configure training parameters and inference engine configurations to meet your specific needs.
*   **Seamless LangGraph Integration:** ART now integrates seamlessly with LangGraph.

## Getting Started

Install ART using pip:

```bash
pip install openpipe-art
```

Run the [2048](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/2048/2048.ipynb) example on Colab to get a quick hands-on introduction.

## ART Examples & Benchmarks

Explore pre-built agents and performance benchmarks:

| Agent Task           | Example Notebook                                                                                                                       | Description                                         | Comparative Performance                                                                                                                                                                                                     |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ART•E LangGraph**  | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/langgraph/art-e-langgraph.ipynb)   | Qwen 2.5 7B learns to search emails using LangGraph | [Link coming soon]                                                                                                                                                                                                          |
| **MCP•RL**           | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/mcp-rl/mcp-rl.ipynb)               | Qwen 2.5 3B masters the NWS MCP server              | [Link coming soon]                                                                                                                                                                                                          |
| **ART•E [RULER]**    | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/art-e.ipynb)                       | Qwen 2.5 7B learns to search emails using RULER     | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/email_agent/accuracy-training-progress.svg" height="72"> [benchmarks](/dev/art-e/art_e/evaluate/display_benchmarks.ipynb)                              |
| **2048**             | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/2048/2048.ipynb)                   | Qwen 2.5 3B learns to play 2048                     | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/2048/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/2048/display_benchmarks.ipynb)                                                |
| **Temporal Clue**    | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/temporal_clue/temporal-clue.ipynb) | Qwen 2.5 7B learns to solve Temporal Clue           | [Link coming soon]                                                                                                                                                                                                          |
| **Tic Tac Toe**      | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb)     | Qwen 2.5 3B learns to play Tic Tac Toe              | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/tic-tac-toe-local/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/tic_tac_toe/display-benchmarks.ipynb)                            |
| **Codenames**        | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/codenames/Codenames_RL.ipynb)      | Qwen 2.5 3B learns to play Codenames                | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/codenames/win_rate_over_time.png" height="72"> [benchmarks](https://github.com/OpenPipe/art-notebooks/blob/main/examples/codenames/Codenames_RL.ipynb) |
| **AutoRL [RULER]**   | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/auto_rl.ipynb)                     | Train Qwen 2.5 7B to master any task                | [Link coming soon]                                                                                                                                                                                                          |

## Learn More

*   **ART•E Agent:** Learn how we trained an email agent that beats o3 - [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent).

    <img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

*   **ART News:** Stay updated with the latest research and developments.
    *   [ART now integrates seamlessly with LangGraph](https://art.openpipe.ai/integrations/langgraph-integration)
    *   [MCP•RL: Teach Your Model to Master Any MCP Server](https://x.com/corbtt/status/1953171838382817625)
    *   [AutoRL: Zero-Data Training for Any Task](https://x.com/mattshumer_/status/1950572449025650733)
    *   [RULER: Easy Mode for RL Rewards](https://openpipe.ai/blog/ruler-easy-mode-for-rl-rewards)
    *   [ART·E: How We Built an Email Research Agent That Beats o3](https://openpipe.ai/blog/art-e-mail-agent)
    *   [ART Trainer: A New RL Trainer for Agents](https://openpipe.ai/blog/art-trainer)

*   **ART Blog:**  [See all blog posts →](https://openpipe.ai/blog)
*   **Documentation:** Access detailed documentation for comprehensive guidance.

## Supported Models

ART is designed to work with most vLLM/HuggingFace-transformers compatible causal language models.

## Contributing

Contributions are welcome!  See the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines.

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

## Credits

ART is built upon the work of many. We're especially grateful to:

*   [Unsloth](https://github.com/unslothai/unsloth)
*   [vLLM](https://github.com/vllm-project/vllm)
*   [trl](https://github.com/huggingface/trl)
*   [torchtune](https://github.com/pytorch/torchtune)
*   [SkyPilot](https://github.com/skypilot-org/skypilot)

[pypi-url]: https://pypi.org/project/openpipe-art/
[contribute-url]: https://github.com/openpipe/art/blob/main/CONTRIBUTING.md
[contribute-image]: https://img.shields.io/badge/PRs-welcome-blue.svg