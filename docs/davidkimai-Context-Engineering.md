# Context Engineering: Master the Art of the LLM Context Window (and Beyond)

**Tired of prompt engineering limitations?** Discover Context Engineering, a revolutionary approach to LLMs that focuses on designing, orchestrating, and optimizing the *entire* context window for unparalleled performance and results.  [Explore the original repository on GitHub](https://github.com/davidkimai/Context-Engineering)

## Key Features & Benefits:

*   **First-Principles Approach:** Learn context engineering from the ground up, starting with fundamental concepts and progressively building towards advanced techniques.
*   **Comprehensive Course in Progress:** A detailed curriculum is under development to guide you from beginner to expert, with a focus on operationalizing the latest research.
*   **Hands-On Examples:**  Experiment with practical, runnable code examples and templates.
*   **Research-Driven:**  Dive deep into cutting-edge research papers, including studies from IBM Zurich, ICML Princeton, and more, to understand the latest advancements in the field.
*   **Biological Metaphor:**  Context Engineering is presented as a progressive field analogous to biological structures: atoms → molecules → cells → organs → neural systems → neural & semantic field theory 
*   **Active Community:** Join the Discord server to discuss context engineering, share your work, and learn from others.

## What is Context Engineering?

Context Engineering goes beyond prompt engineering by focusing on everything *else* the model sees. It's about crafting the perfect information payload for your LLM at inference time, including examples, memory, retrieval, tools, and control flow.

## Core Concepts You'll Master:

*   **Token Budget Optimization:**  Maximize efficiency by carefully managing token usage to reduce costs and improve speed.
*   **Few-Shot Learning & In-Context Learning:** Leverage the power of example-driven learning to guide LLM behavior.
*   **Memory Systems:** Build stateful, coherent interactions that allow LLMs to remember and learn from past interactions.
*   **Retrieval Augmentation (RAG):** Enhance responses with factual data, reducing hallucinations and increasing accuracy.
*   **Cognitive Tools & Prompt Programming:** Develop custom tools and templates to extend capabilities and create new layers for context engineering.
*   **Neural Field Theory:** Model context as a dynamic neural field to allow for iterative context updating.

## Learning Path and Structure

A structured learning path is provided, starting with foundations and progressing through practical guides and examples to advanced research and community contributions.

## Recent Research & Insights:
This repository is based on cutting-edge research including:
*   **MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents**
*   **Eliciting Reasoning in Language Models with Cognitive Tools - IBM Zurich**
*   **Emergent Symbolic Mechanisms Support Abstract Reasoning in Large Language Models - ICML Princeton**

## Quick Start:

1.  **Start with [`00_foundations/01_atoms_prompting.md`]** (5 min): Understand why prompts alone often underperform
2.  **Run [`10_guides_zero_to_hero/01_min_prompt.py`]** (Jupyter Notebook style): Experiment with a minimal working example
3.  **Explore [`20_templates/minimal_context.yaml`]**: Copy/paste a template into your own project
4.  **Study [`30_examples/00_toy_chatbot/`]**: See a complete implementation with context management

## Community & Resources:

*   **Discord:** [Join the Discord Chat](https://discord.gg/JeFENHNNNQ)
*   **DeepWiki:**  [DeepWiki for Context Engineering](https://deepwiki.com/davidkimai/Context-Engineering)
*   **DeepGraph:** [DeepGraph for Context Engineering](https://www.deepgraph.co/davidkimai/Context-Engineering)
*   **NotebookLM + Podcast Deep Dive:** [Context Engineering with NotebookLM](https://notebooklm.google.com/notebook/0c6e4dc6-9c30-4f53-8e1a-05cc9ff3bc7e)

## Contribute:

We welcome contributions! Check out [.github/CONTRIBUTING.md](.github/CONTRIBUTING.md) for guidelines.

## License:

[MIT License](LICENSE)

## Citation:

```bibtex
@misc{context-engineering,
  author = {Context Engineering Contributors},
  title = {Context Engineering: Beyond Prompt Engineering},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/davidkimai/context-engineering}
}
```

## Acknowledgements:

-   [Andrej Karpathy](https://x.com/karpathy/status/1937902205765607626) for coining "context engineering" and inspiring this repo
-   All contributors and the open source community