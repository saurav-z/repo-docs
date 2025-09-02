# Context Engineering: Go Beyond Prompt Engineering with This Comprehensive Guide

**Unlock the full potential of Large Language Models (LLMs) by mastering Context Engineering, the art and science of crafting the perfect information payload for optimal results.  Dive into the core concepts and techniques at the original repository: [davidkimai/Context-Engineering](https://github.com/davidkimai/Context-Engineering).**

This repository provides a comprehensive, first-principles approach to context engineering, moving beyond simple prompt crafting.

## Key Features

*   **First-Principles Approach:** Learn context engineering from the ground up, starting with fundamental concepts and building to advanced techniques.
*   **Practical Examples:**  Hands-on tutorials, code snippets, and real-world examples to help you apply what you learn.
*   **Visualizations:**  Clear diagrams, analogies, and conceptual models that explain complex ideas with visual aids.
*   **Research-Driven:** Stay ahead of the curve with insights from cutting-edge research papers and publications, including ongoing work from ICML, NeurIPS, and IBM Zurich.
*   **Community Collaboration:**  Contribute to the project and learn from other context engineering enthusiasts.

## What is Context Engineering?

> "Context is not just the single prompt users send to an LLM. Context is the complete information payload provided to a LLM at inference time, encompassing all structured informational components that the model needs to plausibly accomplish a given task." - *Definition of Context Engineering from A Systematic Analysis of Over 1400 Research Papers*

Context Engineering is the process of designing, organizing, and optimizing the information provided to LLMs at inference time.  It focuses on everything *beyond* the prompt, including:

*   **Examples and Few-Shot Learning:** Providing the model with relevant demonstrations to guide its behavior.
*   **Memory and State Management:** Enabling the model to remember past interactions and maintain context over time.
*   **Retrieval and Knowledge Integration:** Augmenting the model with access to external knowledge sources.
*   **Tool Integration:** Empowering the model with the ability to use external tools and APIs.
*   **Agent Systems and Orchestration:** Breaking down complex tasks into smaller, manageable steps and coordinating multiple agents.

## Learning Path

This repository offers a structured learning path:

1.  **Foundations:** (00_foundations/) Covering the core concepts of prompt engineering and the limitations of prompts
2.  **Guides:** (10_guides_zero_to_one/) Practical hands-on tutorials that illustrate the principles of context engineering
3.  **Templates:** (20_templates/)  Copy-paste snippets for the most important parts of the code, ready to be used in any project
4.  **Examples:** (30_examples/) Real projects, which become increasingly complex with each step
5.  **Deep Dives and Evaluation:** (40_reference/) Go deeper with advanced topics and research insights
6.  **Community contributions:** (50_contrib/) Contribute to the project

## Research & Resources

Explore key research papers driving innovation in the field:

*   **Context Engineering Survey-Review of 1400 Research Papers**
*   **MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents**
*   **Eliciting Reasoning in Language Models with Cognitive Tools**
*   **Emergent Symbolic Mechanisms Support Abstract Reasoning in Large Language Models**

You can find more insights in these resources as well:

*   [DeepWiki](https://deepwiki.com/davidkimai/Context-Engineering)
*   [DeepGraph](https://www.deepgraph.co/davidkimai/Context-Engineering)
*   [Chat with NotebookLM + Podcast Deep Dive](https://notebooklm.google.com/notebook/0c6e4dc6-9c30-4f53-8e1a-05cc9ff3bc7e)
*   [Discord](https://discord.gg/JeFENHNNNQ)

## Quick Start

1.  **Start with the Basics:** Read [`00_foundations/01_atoms_prompting.md`](00_foundations/01_atoms_prompting.md) (5 min) to understand why prompts alone often underperform
2.  **Run a Simple Example:** Experiment with a minimal working example in [`10_guides_zero_to_hero/01_min_prompt.py`](10_guides_zero_to_hero/01_min_prompt.py) (Jupyter Notebook style)
3.  **Use Templates:** Copy/paste a template into your own project using [`20_templates/minimal_context.yaml`](20_templates/minimal_context.yaml)
4.  **Explore a Complete Implementation:** See a complete implementation with context management in [`30_examples/00_toy_chatbot/`](30_examples/00_toy_chatbot/)

## Contributing

We welcome contributions! Check out our [CONTRIBUTING.md](.github/CONTRIBUTING.md) file for guidelines.

## License

[MIT License](LICENSE)

## Citation

```bibtex
@misc{context-engineering,
  author = {Context Engineering Contributors},
  title = {Context Engineering: Beyond Prompt Engineering},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/davidkimai/context-engineering}
}
```

## Acknowledgements

*   [Andrej Karpathy](https://x.com/karpathy/status/1937902205765607626) for coining "context engineering" and inspiring this repo
*   All contributors and the open-source community