<div align="center">
  <a href="https://art.openpipe.ai">
    <picture>
      <img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
    </picture>
  </a>
  <h1>Agent Reinforcement Trainer (ART)</h1>
</div>

**ART empowers you to train intelligent agents for complex tasks with reinforcement learning, no labeled data required!**  ([See the original repo](https://github.com/OpenPipe/ART))

## Key Features of ART

*   **Train Agents for Real-World Tasks:** ART enables you to train multi-step agents using the GRPO (Generalized Reward Propagation Optimization) framework.
*   **No Labeled Data Required:**  Leverage your server's tools to create scenarios and let ART learn from experience, optimizing models for any Model Context Protocol (MCP) server.
*   **General-Purpose & Adaptable:** ART is designed to optimize models for any MCP server and is suitable for a wide range of applications.
*   **Superior Performance:** Achieve state-of-the-art results, outperforming existing solutions in several benchmarks.
*   **Easy Integration:** Easily integrate ART into your existing projects; no modifications to your MCP server are needed.

## 🔌 MCP•RL: Master MCP Servers with AI Agents

<img src="assets/MCP_RL_diagram.svg" width="7000" alt="MCP•RL Diagram">

**MCP•RL** streamlines the training of agents to effectively utilize any MCP (Model Context Protocol) server with minimal setup. Simply provide the server URL, and MCP•RL will automatically:

1.  Discover server tools.
2.  Generate input tasks that use those tools.
3.  Train the model to improve performance on the MCP server using RULER.
4.  Validate the trained model through testing on new tasks.

```python
from art.rewards import ruler_score_group

# Specialize a model for NWS MCP server
MCP_SERVER_URL = "https://server.smithery.ai/@smithery-ai/national-weather-service/mcp"

# Generate training scenarios based on MCP tools
scenarios = await generate_scenarios(
    num_scenarios=24,
    server_url=MCP_SERVER_URL,
)

# ...run the agent...

# Use RULER to assign relative scores to each trajectory
scored_groups = []
for group in groups:
    judged_group = await ruler_score_group(group)
    scored_groups.append(judged_group)

# Train the model to improve performance on the MCP server
await model.train(scored_groups)
```

## ART Overview: Learn from Experience

ART is an open-source reinforcement learning (RL) framework that enhances agent reliability by enabling Large Language Models (LLMs) to learn from experience.  It provides a straightforward way to integrate GRPO into your Python applications.  Check out the [docs](https://art.openpipe.ai) to learn more.

## 📒 Training Examples (Notebooks)

Explore various agent tasks and their performance through our example notebooks:

| Agent Task         | Example Notebook                                                                                                             | Description                                     | Comparative Performance                                                                                                                                                                             |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **MCP•RL**         | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/mcp-rl/mcp-rl.ipynb)            | Qwen 2.5 3B masters the NWS MCP server          | [Link coming soon]                                                                                                                                                                                  |
| **ART•E [RULER]**  | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/art-e/art-e.ipynb)                 | Qwen 2.5 7B learns to search emails using RULER | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/email_agent/accuracy-training-progress.svg" height="72" alt="Email Agent Benchmark"> [benchmarks](/examples/art-e/art_e/evaluate/display_benchmarks.ipynb) |
| **2048**           | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)                   | Qwen 2.5 3B learns to play 2048                 | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/2048/accuracy-training-progress.svg" height="72" alt="2048 Agent Benchmark"> [benchmarks](/examples/2048/benchmark_2048.ipynb)                            |
| **Temporal Clue**  | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/temporal_clue/temporal-clue.ipynb) | Qwen 2.5 7B learns to solve Temporal Clue       | [Link coming soon]                                                                                                                                                                                  |
| **Tic Tac Toe**    | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb)     | Qwen 2.5 3B learns to play Tic Tac Toe          | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/tic-tac-toe-local/accuracy-training-progress.svg" height="72" alt="Tic Tac Toe Agent Benchmark"> [benchmarks](/examples/tic_tac_toe/benchmark_tic_tac_toe.ipynb) |
| **Codenames**      | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/codenames/Codenames_RL.ipynb)      | Qwen 2.5 3B learns to play Codenames            | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/codenames/win_rate_over_time.png" height="72" alt="Codenames Agent Benchmark"> [benchmarks](/examples/codenames/Codenames_RL.ipynb)                            |
| **AutoRL [RULER]** | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/auto_rl.ipynb)                     | Train Qwen 2.5 7B to master any task            | [Link coming soon]                                                                                                                                                                                  |

## 📰 ART News and Updates

Stay informed about the latest research and developments in agent creation with ART:

*   🗞️ **[AutoRL: Zero-Data Training for Any Task](https://x.com/mattshumer_/status/1950572449025650733)**: Train custom AI models without labeled data through automatic input generation and RULER evaluation.
*   🗞️ **[RULER: Easy Mode for RL Rewards](https://openpipe.ai/blog/ruler-easy-mode-for-rl-rewards)**: Learn how RULER simplifies reward generation in reinforcement learning.
*   🗞️ **[ART·E: How We Built an Email Research Agent That Beats o3](https://openpipe.ai/blog/art-e-mail-agent)**: Discover how Qwen 2.5 14B email agent outperforms OpenAI's o3.
*   🗞️ **[ART Trainer: A New RL Trainer for Agents](https://openpipe.ai/blog/art-trainer)**: Easily train LLM-based agents using GRPO with the new ART Trainer.

[📖 See all blog posts →](https://openpipe.ai/blog)

## Why Choose ART?

*   **Simplify RL Integration:**  ART provides convenient wrappers for introducing RL training into existing applications, abstracting the training server into a modular service.
*   **Train from Anywhere:** Run the ART client locally, or leverage the ART server for an ephemeral GPU-enabled environment, providing flexibility in training.
*   **Enhanced Observability & Debugging:**  Integrations with platforms like W&B, Langfuse, and OpenPipe offer flexible observability and simplified debugging.
*   **Intelligent Defaults & Customization:** Utilize optimized defaults or configure training parameters and inference engine settings to suit your specific needs.

## Installation

Easily integrate ART into your project with the following command:

```bash
pip install openpipe-art
```

## 🤖 ART•E Agent: Real-World Application

Explore the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) blog post to learn how we trained Qwen 2.5 14B to surpass o3 in email retrieval!

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700" alt="ART-E Agent Performance Graph">

## 🔁 Training Loop Explained

ART uses a client-server architecture, dividing functionality between a client and a server. The client facilitates interaction between ART and your codebase, while the server handles the complexity of inference and training:

1.  **Inference:**

    1.  Your code uses the ART client to execute an agentic workflow.
    2.  Completion requests are sent to the ART server, which runs the model's latest LoRA in vLLM.
    3.  Each `system`, `user`, and `assistant` message is saved in a Trajectory.
    4.  When a rollout finishes, your code assigns a `reward` to its Trajectory.

2.  **Training:**

    1.  After each rollout, Trajectories are grouped and sent to the server. Inference is paused while training runs.
    2.  The server trains your model using GRPO, starting from the latest checkpoint.
    3.  The server saves the trained LoRA and loads it into vLLM.
    4.  Inference resumes, and the loop repeats.

This training loop continues until a predetermined number of inference and training iterations are complete.

## 🧩 Supported Models

ART is designed to be compatible with a wide array of causal language models supported by vLLM/HuggingFace-transformers, and at least those that are supported by [Unsloth](https://docs.unsloth.ai/get-started/all-our-models). If you find any model that is not working, please report it on [Discord](https://discord.gg/zbBHRUpwf4) or [GitHub](https://github.com/openpipe/art/issues)!

## 🤝 Contributing

ART is actively developed, and we welcome contributions!  See the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines.

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

## 🙏 Acknowledgements

ART is built upon the work of many.  We are especially grateful to the authors of the following projects:

-   [Unsloth](https://github.com/unslothai/unsloth)
-   [vLLM](https://github.com/vllm-project/vllm)
-   [trl](https://github.com/huggingface/trl)
-   [torchtune](https://github.com/pytorch/torchtune)
-   [SkyPilot](https://github.com/skypilot-org/skypilot)

We also appreciate the support of our partners who have helped us test ART.