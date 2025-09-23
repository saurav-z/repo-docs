<div align="center">

<a href="https://art.openpipe.ai"><picture>
<img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
</picture></a>

</div>

# Agent Reinforcement Trainer (ART): Train LLM Agents for Real-World Tasks

**ART empowers you to train multi-step LLM agents more effectively using GRPO and RULER, the innovative zero-shot reward system.**

[View on GitHub](https://github.com/OpenPipe/ART) | [Documentation](https://art.openpipe.ai) | [Join Discord](https://discord.gg/zbBHRUpwf4)

## Key Features

*   **Effortless Training:** Train agents using GRPO (Guided Policy Optimization), simplifying complex training pipelines.
*   **Zero-Shot Rewards with RULER:** Automatically score agent trajectories using an LLM as a judge, eliminating the need for hand-crafted reward functions and labeled data.
*   **Faster Development:** Accelerate your development process by skipping reward function engineering, potentially achieving 2-3x faster development.
*   **Versatile & General-Purpose:** Applicable across various tasks without requiring modification.
*   **Performance Boost:** Achieve performance that matches or surpasses hand-crafted rewards in various benchmarks.
*   **Easy Integration:** Seamlessly integrate into your existing applications as a drop-in replacement for manual reward functions.
*   **Notebook Examples:** Get started quickly with pre-built notebooks for tasks like email search, 2048, and more.

## RULER: Zero-Shot Agent Rewards

**RULER (Relative Universal LLM-Elicited Rewards) revolutionizes agent training by eliminating hand-crafted reward functions, providing automatic scoring using LLMs.** Simply define your task in the system prompt, and RULER handles the rest. This innovative method requires no labeled data, expert feedback, or reward engineering.

✨ **Key Benefits of RULER:**

*   **Accelerated Development:** Significantly reduces development time.
*   **Broad Applicability:** Works seamlessly across a wide array of tasks.
*   **Performance Excellence:** Achieves high performance, rivaling hand-crafted rewards in many scenarios.
*   **Simplified Integration:** Easy to integrate as a direct replacement for manual reward functions.

```python
# From hours of reward engineering...
def complex_reward_function(trajectory):
    # 50+ lines of careful scoring logic...
    pass

# ...to one line with RULER
judged_group = await ruler_score_group(group, "openai/o3")
```

[📖 Learn more about RULER →](https://art.openpipe.ai/fundamentals/ruler)

## ART: An Open-Source RL Framework

ART provides an open-source RL (Reinforcement Learning) framework, enhancing agent reliability by leveraging LLMs to learn from experience. This framework provides an ergonomic harness, making it easy to integrate GRPO into your Python applications. Explore the [docs](https://art.openpipe.ai) to learn more.

## 📒 Notebooks: Get Started Quickly

| Agent Task          | Example Notebook                                                                                                                       | Description                                                                | Comparative Performance                                                                                                                                                                                                     |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ART•E LangGraph** | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/langgraph/art-e-langgraph.ipynb)   | Qwen 2.5 7B learns to search emails using LangGraph                    | [Link coming soon]                                                                                                                                                                                                          |
| **MCP•RL**          | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/mcp-rl/mcp-rl.ipynb)               | Qwen 2.5 3B masters the NWS MCP server                                    | [Link coming soon]                                                                                                                                                                                                          |
| **ART•E [RULER]**   | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/art-e.ipynb)                       | Qwen 2.5 7B learns to search emails using RULER                           | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/email_agent/accuracy-training-progress.svg" height="72"> [benchmarks](/dev/art-e/art_e/evaluate/display_benchmarks.ipynb)                              |
| **2048**            | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/2048/2048.ipynb)                   | Qwen 2.5 3B learns to play 2048                                         | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/2048/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/2048/display_benchmarks.ipynb)                                                |
| **Temporal Clue**   | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/temporal_clue/temporal-clue.ipynb) | Qwen 2.5 7B learns to solve Temporal Clue                                | [Link coming soon]                                                                                                                                                                                                          |
| **Tic Tac Toe**     | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb)     | Qwen 2.5 3B learns to play Tic Tac Toe                                  | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/tic-tac-toe-local/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/tic_tac_toe/display-benchmarks.ipynb)                            |
| **Codenames**       | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/codenames/Codenames_RL.ipynb)      | Qwen 2.5 3B learns to play Codenames                                      | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/codenames/win_rate_over_time.png" height="72"> [benchmarks](https://github.com/OpenPipe/art-notebooks/blob/main/examples/codenames/Codenames_RL.ipynb) |
| **AutoRL [RULER]**  | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/auto_rl.ipynb)                     | Train Qwen 2.5 7B to master any task                                       | [Link coming soon]                                                                                                                                                                                                          |

## 📰 ART News

*   **[ART now integrates seamlessly with LangGraph](https://art.openpipe.ai/integrations/langgraph-integration)** - Train your LangGraph agents with reinforcement learning for smarter multi-step reasoning and improved tool usage.
*   **[MCP•RL: Teach Your Model to Master Any MCP Server](https://x.com/corbtt/status/1953171838382817625)** - Automatically train models to effectively use MCP server tools through reinforcement learning.
*   **[AutoRL: Zero-Data Training for Any Task](https://x.com/mattshumer_/status/1950572449025650733)** - Train custom AI models without labeled data using automatic input generation and RULER evaluation.
*   **[RULER: Easy Mode for RL Rewards](https://openpipe.ai/blog/ruler-easy-mode-for-rl-rewards)** is now available for automatic reward generation in reinforcement learning.
*   **[ART·E: How We Built an Email Research Agent That Beats o3](https://openpipe.ai/blog/art-e-mail-agent)** demonstrates a Qwen 2.5 14B email agent outperforming OpenAI's o3.
*   **[ART Trainer: A New RL Trainer for Agents](https://openpipe.ai/blog/art-trainer)** enables easy training of LLM-based agents using GRPO.

[📖 See all blog posts →](https://openpipe.ai/blog)

## Why Use ART?

*   **Simplified Integration:** Easily incorporate RL training into your existing applications with convenient wrappers.
*   **Flexible Training:** Train from anywhere - on your laptop or with a GPU-enabled environment via the ART server.
*   **Enhanced Observability:** Leverage integrations with platforms like W&B, Langfuse, and OpenPipe for improved debugging.
*   **Intelligent Defaults:** Customize training parameters and inference engine configurations or use the optimized defaults for efficiency and stability.

## Installation

Quickly install ART to start training:

```bash
pip install openpipe-art
```

## 🤖 ART•E Agent

Learn how ART can be applied to real-world tasks by exploring the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) blog post.

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## 🔁 Training Loop Overview

ART’s functionality is split into a **client** and a **server**. The client (compatible with the OpenAI API) interfaces between ART and your codebase. The server, which runs independently on any machine with a GPU, abstracts the complexity of the inference and training portions of the RL loop, and provides for custom configuration.

1.  **Inference:**

    *   Your code uses the ART client to perform an agentic workflow (usually executing several rollouts in parallel to gather data faster).
    *   Completion requests are routed to the ART server, which runs the model's latest LoRA in vLLM.
    *   As the agent executes, each `system`, `user`, and `assistant` message is stored in a Trajectory.
    *   When a rollout finishes, your code assigns a `reward` to its Trajectory, indicating the performance of the LLM.

2.  **Training:**

    *   When each rollout has finished, Trajectories are grouped and sent to the server. Inference is blocked while training executes.
    *   The server trains your model using GRPO, initializing from the latest checkpoint (or an empty LoRA on the first iteration).
    *   The server saves the newly trained LoRA to a local directory and loads it into vLLM.
    *   Inference is unblocked and the loop resumes at step 1.

The training loop runs until a specified number of inference and training iterations are complete.

## 🧩 Supported Models

ART is designed to work with most vLLM/HuggingFace-transformers compatible causal language models, including those supported by [Unsloth](https://docs.unsloth.ai/get-started/all-our-models). For support of a specific model, please contact us on [Discord](https://discord.gg/zbBHRUpwf4) or open an issue on [GitHub](https://github.com/openpipe/art/issues)!

## 🤝 Contributing

Contributions are welcome! See the [CONTRIBUTING.md](CONTRIBUTING.md) file for details.

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

Available under the [Apache-2.0 License](LICENSE).

## 🙏 Credits

ART is built on the work of many. We are especially grateful to the following projects:

*   [Unsloth](https://github.com/unslothai/unsloth)
*   [vLLM](https://github.com/vllm-project/vllm)
*   [trl](https://github.com/huggingface/trl)
*   [torchtune](https://github.com/pytorch/torchtune)
*   [SkyPilot](https://github.com/skypilot-org/skypilot)

Thank you to our partners for their help testing ART!