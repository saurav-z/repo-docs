<div align="center">

<a href="https://art.openpipe.ai"><picture>
<img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
</picture></a>

<p align="center">
  <h1>Agent Reinforcement Trainer (ART): Train LLM Agents Effectively</h1>
</p>

<p>
  Use Agent Reinforcement Trainer (ART) to train multi-step agents for real-world tasks using GRPO.
</p>

[![PRs-Welcome][contribute-image]][contribute-url]
[![PyPI version](https://img.shields.io/pypi/v/openpipe-art?color=364fc7)][pypi-url]
[![Train Agent](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/2048/2048.ipynb)

[![Join Discord](https://img.shields.io/badge/Join%20Discord-5865F2?style=plastic&logo=discord&logoColor=white)](https://discord.gg/zbBHRUpwf4)
[![Documentation](https://img.shields.io/badge/Documentation-orange?style=plastic&logo=gitbook&logoColor=white)](https://art.openpipe.ai)

</div>

## **Agent Reinforcement Trainer (ART): Train LLM Agents for Complex Tasks**

ART is an open-source RL framework designed to improve the reliability and performance of LLM agents by enabling them to learn from experience. ART provides a user-friendly system for integrating GRPO into any Python application, enabling you to create smarter, more capable agents.

**Key Features:**

*   **GRPO-Powered Training:** Leverage the power of GRPO to train your agents and improve their performance on complex, multi-step tasks.
*   **Open-Source & Customizable:** ART is open-source, offering flexibility and customization for your specific needs.
*   **Easy Integration:**  Seamlessly integrate ART into your existing Python projects with a simple `pip install openpipe-art`.
*   **Modular Client-Server Architecture:** ART's architecture separates the client (your code) from the server (training and inference), allowing for easy setup and scaling.
*   **Built-in Observability:** Integrations with popular platforms like W&B, Langfuse, and OpenPipe simplify debugging and provide valuable insights.
*   **Optimized Defaults:** Benefit from intelligent default settings optimized for training efficiency and stability.

### **RULER: Zero-Shot Agent Rewards**

**RULER** (Relative Universal LLM-Elicited Rewards) dramatically simplifies reward function creation by using an LLM-as-judge to automatically score agent trajectories.  This eliminates the need for hand-crafted reward functions.

**Key Benefits of RULER:**

*   **Faster Development:** Reduce development time by skipping reward function engineering.
*   **General Purpose:** Works across any task without modification.
*   **Strong Performance:** Achieves or exceeds the performance of hand-crafted rewards in many benchmarks.
*   **Easy to Use:**  Simple one-line implementation.

```python
# Before: Hours of reward engineering
def complex_reward_function(trajectory):
    # 50+ lines of careful scoring logic...
    pass

# After: One line with RULER
judged_group = await ruler_score_group(group, "openai/o3")
```

[📖 Learn more about RULER →](https://art.openpipe.ai/fundamentals/ruler)

### **ART Overview**

ART provides an ergonomic harness for integrating GRPO into any python application.

### **Getting Started with ART**

1.  **Installation:** Install ART using pip:

```bash
pip install openpipe-art
```

2.  **Explore Notebooks:** Quickly get hands-on experience with our example notebooks to explore ART's capabilities. When you're ready to learn more, check out the [docs](https://art.openpipe.ai).

### **📒 Example Notebooks**

| Agent Task          | Example Notebook                                                                                                                       | Description                                         | Comparative Performance                                                                                                                                                                                                     |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ART•E LangGraph** | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/langgraph/art-e-langgraph.ipynb)   | Qwen 2.5 7B learns to search emails using LangGraph | [Link coming soon]                                                                                                                                                                                                          |
| **MCP•RL**          | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/mcp-rl/mcp-rl.ipynb)               | Qwen 2.5 3B masters the NWS MCP server              | [Link coming soon]                                                                                                                                                                                                          |
| **ART•E [RULER]**   | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/art-e.ipynb)                       | Qwen 2.5 7B learns to search emails using RULER     | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/email_agent/accuracy-training-progress.svg" height="72"> [benchmarks](/dev/art-e/art_e/evaluate/display_benchmarks.ipynb)                              |
| **2048**            | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/2048/2048.ipynb)                   | Qwen 2.5 3B learns to play 2048                     | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/2048/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/2048/display_benchmarks.ipynb)                                                |
| **Temporal Clue**   | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/temporal_clue/temporal-clue.ipynb) | Qwen 2.5 7B learns to solve Temporal Clue           | [Link coming soon]                                                                                                                                                                                                          |
| **Tic Tac Toe**     | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb)     | Qwen 2.5 3B learns to play Tic Tac Toe              | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/tic-tac-toe-local/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/tic_tac_toe/display-benchmarks.ipynb)                            |
| **Codenames**       | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/codenames/Codenames_RL.ipynb)      | Qwen 2.5 3B learns to play Codenames                | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/codenames/win_rate_over_time.png" height="72"> [benchmarks](https://github.com/OpenPipe/art-notebooks/blob/main/examples/codenames/Codenames_RL.ipynb) |
| **AutoRL [RULER]**  | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/auto_rl.ipynb)                     | Train Qwen 2.5 7B to master any task                | [Link coming soon]                                                                                                                                                                                                          |

### **📰 ART News**

*   **[ART now integrates seamlessly with LangGraph](https://art.openpipe.ai/integrations/langgraph-integration)** - Train your LangGraph agents with reinforcement learning for smarter multi-step reasoning and improved tool usage.
*   **[MCP•RL: Teach Your Model to Master Any MCP Server](https://x.com/corbtt/status/1953171838382817625)** - Automatically train models to effectively use MCP server tools through reinforcement learning.
*   **[AutoRL: Zero-Data Training for Any Task](https://x.com/mattshumer_/status/1950572449025650733)** - Train custom AI models without labeled data using automatic input generation and RULER evaluation.
*   **[RULER: Easy Mode for RL Rewards](https://openpipe.ai/blog/ruler-easy-mode-for-rl-rewards)** is now available for automatic reward generation in reinforcement learning.
*   **[ART·E: How We Built an Email Research Agent That Beats o3](https://openpipe.ai/blog/art-e-mail-agent)** demonstrates a Qwen 2.5 14B email agent outperforming OpenAI's o3.
*   **[ART Trainer: A New RL Trainer for Agents](https://openpipe.ai/blog/art-trainer)** enables easy training of LLM-based agents using GRPO.

[📖 See all blog posts →](https://openpipe.ai/blog)

### **Why Choose ART?**

*   **Simplified RL for Existing Applications:** ART provides convenient wrappers for integrating RL training into your existing projects.
*   **Flexible Training:** Train agents from anywhere - on your laptop, a local GPU, or a cloud environment.
*   **Enhanced Observability:** Integrate with platforms like W&B, Langfuse, and OpenPipe for improved debugging.
*   **Optimized Defaults:** ART provides intelligent defaults that have been optimized for training efficiency and stability.

### **🤖 ART•E Agent - Real-World Example**

Discover how to use ART for real-world tasks by exploring the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) blog post, which details how we trained Qwen 2.5 14B to beat o3 at email retrieval!

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

### **🔁 Training Loop Explained**

ART's architecture uses a client and server setup, allowing you to easily train and deploy LLM agents:

1.  **Inference:**
    *   Your code uses the ART client to perform agentic workflows.
    *   Completions are routed to the ART server, which runs the model's latest LoRA in vLLM.
    *   Each message is stored in a Trajectory.
    *   Your code assigns a `reward` when a rollout finishes.
2.  **Training:**
    *   Trajectories are sent to the server after each rollout.
    *   The server trains your model using GRPO.
    *   The server saves the new LoRA.
    *   Inference resumes.

This loop continues until the training completes.

### **🧩 Supported Models**

ART should work with most vLLM/HuggingFace-transformers compatible causal language models, or at least the ones supported by [Unsloth](https://docs.unsloth.ai/get-started/all-our-models). Gemma 3 does not appear to be supported for the time being. If any other model isn't working for you, please let us know on [Discord](https://discord.gg/zbBHRUpwf4) or open an issue on [GitHub](https://github.com/openpipe/art/issues)!

### **🤝 Contributing**

ART is under active development, and welcomes contributions!  See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

### **📖 Citation**

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

### **⚖️ License**

This project is licensed under the [Apache-2.0 License](LICENSE).

### **🙏 Credits**

ART relies on the contributions of the open-source RL community, and we're especially grateful to the authors of:

*   [Unsloth](https://github.com/unslothai/unsloth)
*   [vLLM](https://github.com/vllm-project/vllm)
*   [trl](https://github.com/huggingface/trl)
*   [torchtune](https://github.com/pytorch/torchtune)
*   [SkyPilot](https://github.com/skypilot-org/skypilot)

We are also grateful to our partners who helped us test ART.

[pypi-url]: https://pypi.org/project/openpipe-art/
[contribute-url]: https://github.com/openpipe/art/blob/main/CONTRIBUTING.md
[contribute-image]: https://img.shields.io/badge/PRs-welcome-blue.svg
```
Key changes and improvements:

*   **SEO Optimization:**  Incorporated relevant keywords (Agent Reinforcement Trainer, LLM Agents, GRPO, RULER) throughout the text.
*   **Concise Hook:**  Added a clear, compelling one-sentence description to grab attention.
*   **Structured Headings:** Used proper headings (H1, H2, H3) for improved readability and SEO.
*   **Bulleted Lists:** Used bullet points to highlight key features and benefits, making the information easily scannable.
*   **Clear Call to Action:** Encourages users to explore the example notebooks.
*   **Simplified Explanations:**  Made the descriptions of RULER and the training loop more concise and easier to understand.
*   **Comprehensive Overview:**  Provided a more complete overview of ART's capabilities and benefits.
*   **Focus on User Benefits:**  Emphasized the advantages of using ART, such as faster development, ease of use, and improved agent performance.
*   **Improved Formatting:** Better use of Markdown to improve readability (e.g., bolding, emphasis).
*   **Direct links to code:** Used more direct links, such as to the example notebooks.

This improved README is more user-friendly, SEO-optimized, and effectively communicates the value proposition of ART.