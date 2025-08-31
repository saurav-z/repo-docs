# Context Engineering: Master the Art and Science of Context Design (🚀 Explore the Future of LLMs)

> **Move beyond prompt engineering and unlock the full potential of Large Language Models (LLMs) by mastering context design, orchestration, and optimization.**

**[Explore the original repository](https://github.com/davidkimai/Context-Engineering)**

This repository provides a comprehensive, first-principles approach to **Context Engineering**, guiding you through the core concepts, practical techniques, and cutting-edge research that define this rapidly evolving field. It's designed for learners of all levels, from beginners to experienced AI practitioners.

## Key Features

*   💡 **First-Principles Approach:** Learn the foundational principles of context design.
*   🛠️ **Practical Examples:** Hands-on tutorials with runnable code examples.
*   📚 **Comprehensive Course:** A structured learning path with foundations, system implementation, integration, and frontier concepts.
*   🧠 **Cutting-Edge Research:** Dive into the latest findings from ICML, NeurIPS, and more.
*   🤝 **Community Focused:** Join the Discord community to discuss, collaborate, and contribute.
*   🔗 **Links to Resources:** Easily access relevant tools and projects.

## What is Context Engineering?

Context Engineering goes beyond simple prompt engineering. It's about designing the entire information payload that an LLM receives during inference. This includes:

*   Prompts
*   Examples (Few-Shot Learning)
*   Memory (Persistent Information)
*   Retrieval (Document Augmentation)
*   Tools
*   State Management
*   Control Flow

**Think of it this way:** While prompt engineering focuses on "What you say" (the single instruction), Context Engineering encompasses "Everything else the model sees" (examples, memory, retrieval, tools, state, and control flow).

## Why Context Engineering Matters

As Andrej Karpathy noted, "Context engineering is the delicate art and science of filling the context window with just the right information for the next step." ([Source](https://x.com/karpathy/status/1937902205765607626)).

By mastering context engineering, you can:

*   Improve LLM performance and accuracy
*   Reduce token costs and latency
*   Build more complex and capable AI systems
*   Unlock new possibilities in LLM applications

## Learning Path

The repository offers a structured learning path, taking you from foundational concepts to advanced techniques:

*   **00\_foundations/**: Theory & core concepts.
*   **10\_guides\_zero\_to\_one/**: Hands-on walkthroughs.
*   **20\_templates/**: Copy-paste snippets.
*   **30\_examples/**: Real projects, progressively complex.
*   **40\_reference/**: Deep dives & evaluation cookbook.
*   **50\_contrib/**: Community contributions

## Quick Start

1.  **Start Here:** Begin with [`00_foundations/01_atoms_prompting.md`](00_foundations/01_atoms_prompting.md) (5 min) to understand why prompts alone are often insufficient.
2.  **Run a Simple Example:** Experiment with [`10_guides_zero_to_hero/01_min_prompt.py`](10_guides_zero_to_hero/01_min_prompt.py) (Jupyter Notebook style).
3.  **Explore Templates:** Copy and paste templates from [`20_templates/minimal_context.yaml`](20_templates/minimal_context.yaml) into your own projects.
4.  **Examine a Complete Implementation:** Study a complete context management implementation in [`30_examples/00_toy_chatbot/`](30_examples/00_toy_chatbot/).

## Key Concepts to Master

*   **Token Budget:** Optimizing token usage.
*   **Few-Shot Learning:** Learning by example.
*   **Memory Systems:** Persisting information.
*   **Retrieval Augmentation:** Injecting relevant documents.
*   **Control Flow:** Breaking down tasks.
*   **Context Pruning:** Removing irrelevant data.
*   **Metrics & Evaluation:** Measuring effectiveness.
*   **Cognitive Tools & Prompt Programming:** Building custom tools.
*   **Neural Field Theory:** Dynamic context modeling.
*   **Symbolic Mechanisms:** Enhancing reasoning with symbolic architectures.
*   **Quantum Semantics:** Designing systems with superpositional techniques.

## Research Spotlight

This repository is informed by the latest research. Some key papers:

*   **MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents** - [Singapore-MIT June 2025](https://www.arxiv.org/pdf/2506.15841)
*   **Eliciting Reasoning in Language Models with Cognitive Tools** - [IBM Zurich June 2025](https://www.arxiv.org/pdf/2506.12115)
*   **Emergent Symbolic Mechanisms Support Abstract Reasoning in Large Language Models** - [ICML Princeton June 18, 2025](https://openreview.net/forum?id=y1SnRPDWx4)

## Join the Community

*   **Discord:** [Join our Discord](https://discord.gg/JeFENHNNNQ) to connect with other context engineering enthusiasts.

## Contribute

We welcome contributions! See the [CONTRIBUTING.md](.github/CONTRIBUTING.md) file for guidelines.

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