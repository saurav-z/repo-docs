<div align="center">
  <a href="https://art.openpipe.ai">
    <picture>
      <img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
    </picture>
  </a>
  <h1>Agent Reinforcement Trainer (ART)</h1>
</div>

## Supercharge Your Agents with ART: Train LLMs to Achieve Real-World Tasks!

ART is an open-source framework that empowers you to train multi-step agents for complex tasks, enabling LLMs to learn from experience through reinforcement learning. Integrate GRPO into your Python applications effortlessly and watch your agents evolve!

*   [**GitHub Repository**](https://github.com/OpenPipe/ART)
*   [**Documentation**](https://art.openpipe.ai)
*   [**Join Discord**](https://discord.gg/zbBHRUpwf4)
*   [**Try the 2048 Notebook**](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/2048/2048.ipynb)

**Key Features:**

*   **LangGraph Integration:** Easily integrate ART with LangGraph to build smart, adaptable agents.
    *   **Automatic behavior improvement:** Train agents to get better at multi-step reasoning
    *   **Tool usage optimization:** Learn when and how to use tools more effectively
    *   **Seamless integration:** Drop-in replacement for LangGraph's LLM initialization
    *   **RULER compatibility:** Train without hand-crafted reward functions
*   **Simplified Training:**  Provides an ergonomic harness for integrating GRPO into any python application.
*   **RULER Compatibility:** Train agents without needing to engineer reward functions.
*   **Flexible Deployment:** Train agents on your laptop or leverage GPU-enabled environments in the cloud.
*   **Observability & Debugging:** Integrations with hosted platforms like W&B, Langfuse, and OpenPipe.
*   **Customizable & Optimized:** Utilize intelligent defaults while being able to configure training parameters.

## Getting Started with ART

Install ART with a simple pip command:

```bash
pip install openpipe-art
```

## ART & LangGraph: Build Smarter Agents

ART seamlessly integrates with LangGraph, allowing you to build and train ReAct-style agents that improve through reinforcement learning.

```python
import art
from art.langgraph import wrap_rollout, init_chat_model
from langgraph import create_react_agent

# Your existing tools
tools = [search_inbox, read_email, return_final_answer]

@wrap_rollout(model)
async def run_agent(scenario: str) -> art.Trajectory:
    # Create LangGraph agent with ART's LLM wrapper
    agent = create_react_agent(init_chat_model(), tools)

    result = await agent.ainvoke({"messages": [("user", scenario)]})
    return art.Trajectory()  # Automatically captured

# Train with RULER - no reward engineering needed!
await art.train(model, reward_function="ruler")
```

[📖 Learn more about LangGraph integration →](https://art.openpipe.ai/integrations/langgraph-integration) | [🏋️ Try the notebook →](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/langgraph/art-e-langgraph.ipynb)

## Example Notebooks:  Hands-on Training with ART

Explore various use cases and see ART in action with our comprehensive notebooks:

| Agent Task          | Example Notebook                                                                                                                       | Description                                         | Comparative Performance                                                                                                                                                                             |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ART•E LangGraph** | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/langgraph/art-e-langgraph.ipynb)   | Qwen 2.5 7B learns to search emails using LangGraph | [Link coming soon]                                                                                                                                                                                  |
| **MCP•RL**          | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/mcp-rl/mcp-rl.ipynb)               | Qwen 2.5 3B masters the NWS MCP server              | [Link coming soon]                                                                                                                                                                                  |
| **ART•E [RULER]**   | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/art-e.ipynb)                       | Qwen 2.5 7B learns to search emails using RULER     | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/email_agent/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/art-e/art_e/evaluate/display_benchmarks.ipynb) |
| **2048**            | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/2048/2048.ipynb)                   | Qwen 2.5 3B learns to play 2048                     | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/2048/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/2048/benchmark_2048.ipynb)                            |
| **Temporal Clue**   | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/temporal_clue/temporal-clue.ipynb) | Qwen 2.5 7B learns to solve Temporal Clue           | [Link coming soon]                                                                                                                                                                                  |
| **Tic Tac Toe**     | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb)     | Qwen 2.5 3B learns to play Tic Tac Toe              | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/tic-tac-toe-local/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/tic_tac_toe/benchmark_tic_tac_toe.ipynb) |
| **Codenames**       | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/codenames/Codenames_RL.ipynb)      | Qwen 2.5 3B learns to play Codenames                | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/codenames/win_rate_over_time.png" height="72"> [benchmarks](/examples/codenames/Codenames_RL.ipynb)                            |
| **AutoRL [RULER]**  | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/auto_rl.ipynb)                     | Train Qwen 2.5 7B to master any task                | [Link coming soon]                                                                                                                                                                                  |

## ART News & Research

Stay up-to-date with the latest advancements in agent training with ART:

*   🗞️ **[MCP•RL: Teach Your Model to Master Any MCP Server](https://x.com/corbtt/status/1953171838382817625)** - Automatically train models to effectively use MCP server tools through reinforcement learning.
*   🗞️ **[AutoRL: Zero-Data Training for Any Task](https://x.com/mattshumer_/status/1950572449025650733)** - Train custom AI models without labeled data using automatic input generation and RULER evaluation.
*   🗞️ **[RULER: Easy Mode for RL Rewards](https://openpipe.ai/blog/ruler-easy-mode-for-rl-rewards)** is now available for automatic reward generation in reinforcement learning.
*   🗞️ **[ART·E: How We Built an Email Research Agent That Beats o3](https://openpipe.ai/blog/art-e-mail-agent)** demonstrates a Qwen 2.5 14B email agent outperforming OpenAI's o3.
*   🗞️ **[ART Trainer: A New RL Trainer for Agents](https://openpipe.ai/blog/art-trainer)** enables easy training of LLM-based agents using GRPO.

[📖 See all blog posts →](https://openpipe.ai/blog)

## Why Choose ART?

*   **Rapid Integration:** ART offers convenient wrappers for integrating RL training into **existing applications**.
*   **Versatile Training:** Train locally or leverage cloud environments.
*   **Enhanced Debugging:**  Integrations with W&B, Langfuse, and OpenPipe for simplified debugging.
*   **Intelligent Defaults:**  Get started quickly with optimized training parameters and inference configurations, or customize to your specific needs.

## 🤖 ART•E Agent: A Real-World Example

Learn how ART was used to train a Qwen 2.5 14B agent that outperforms o3 at email retrieval in the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) blog post.

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## 🔁 The ART Training Loop Explained

ART's functionality is divided into a **client** and a **server**, following these steps:

1.  **Inference**

    *   Your code uses the ART client to perform an agentic workflow (usually executing several rollouts in parallel to gather data faster).
    *   Completion requests are routed to the ART server, which runs the model's latest LoRA in vLLM.
    *   As the agent executes, each `system`, `user`, and `assistant` message is stored in a Trajectory.
    *   When a rollout finishes, your code assigns a `reward` to its Trajectory, indicating the performance of the LLM.

2.  **Training**

    *   When each rollout has finished, Trajectories are grouped and sent to the server. Inference is blocked while training executes.
    *   The server trains your model using GRPO, initializing from the latest checkpoint (or an empty LoRA on the first iteration).
    *   The server saves the newly trained LoRA to a local directory and loads it into vLLM.
    *   Inference is unblocked and the loop resumes at step 1.

This loop runs until a specified number of iterations have completed.

## 🧩 Supported Models

ART is designed to work with most vLLM/HuggingFace-transformers compatible causal language models, particularly those supported by [Unsloth](https://docs.unsloth.ai/get-started/all-our-models). Note: Gemma 3 is not currently supported.  If you experience issues with other models, please reach out on [Discord](https://discord.gg/zbBHRUpwf4) or open an issue on [GitHub](https://github.com/openpipe/art/issues)!

## 🤝 Contribute

We welcome contributions!  Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

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

ART is built on the foundation of amazing open-source projects:

*   [Unsloth](https://github.com/unslothai/unsloth)
*   [vLLM](https://github.com/vllm-project/vllm)
*   [trl](https://github.com/huggingface/trl)
*   [torchtune](https://github.com/pytorch/torchtune)
*   [SkyPilot](https://github.com/skypilot-org/skypilot)

We thank our partners for helping us test ART!