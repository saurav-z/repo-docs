# Context Engineering: Mastering the Art and Science of LLM Context ([Original Repo](https://github.com/davidkimai/Context-Engineering))

**Unlock the full potential of Large Language Models (LLMs) by mastering context design, orchestration, and optimization—going beyond prompt engineering to achieve groundbreaking results.**

Context engineering is the key to unlocking advanced LLM capabilities. This repository serves as a first-principles handbook for moving beyond prompt engineering and mastering the art of designing, orchestrating, and optimizing the context window. Inspired by the work of Andrej Karpathy and the latest research, this guide helps you build powerful LLM applications by understanding and leveraging the "everything else the model sees."

## Key Features

*   **Comprehensive Course:** A structured, first-principles learning path from foundational concepts to frontier techniques.
*   **Practical Guides:** Hands-on walkthroughs and code examples.
*   **Real-World Examples:** Complete, runnable implementations that you can adapt to your own projects.
*   **Research-Backed:** Detailed exploration of cutting-edge research papers and methodologies, including IBM Zurich, ICML Princeton, and more.
*   **Visual Learning:** Utilizing a Karpathy + 3Blue1Brown inspired style.
*   **Community Driven:** Contribute and collaborate with other developers through our Discord community.

## Core Concepts

*   **Token Budget Optimization:** Learn to manage and optimize your token usage for cost efficiency and performance.
*   **Few-Shot Learning:** Master the art of providing examples to guide LLM behavior.
*   **Memory Systems:** Understand how to implement stateful interactions and persistent information.
*   **Retrieval Augmentation (RAG):** Integrate external knowledge to improve accuracy and reduce hallucinations.
*   **Control Flow:** Break down complex tasks into manageable steps.
*   **Context Pruning:** Learn how to remove irrelevant information for better performance.
*   **Metrics and Evaluation:** Discover how to measure and optimize the effectiveness of your context engineering strategies.
*   **Cognitive Tools & Prompt Programming:** Explore the power of prompt programming and building custom tools and templates.
*   **Neural Field Theory:** Use context as a neural field to allow for iterative context updating.
*   **Symbolic Mechanisms:** Enable higher-order reasoning through symbolic architectures.
*   **Quantum Semantics:** Leverage superpositional techniques in your context design.

## Learning Path

The learning path is structured around these core phases:

1.  **Foundations:** Core concepts and theory.
2.  **Guides:** Hands-on walkthroughs and practical advice.
3.  **Templates:** Ready-to-use code snippets and examples.
4.  **Examples:** Complete projects with progressively more complex context management.
5.  **Reference:** Deep dives into advanced concepts.
6.  **Contribute:** Community contributions and collaboration.

## Why Context Engineering Matters

Context engineering goes beyond prompt engineering by focusing on the "everything else" that shapes an LLM's output. This includes examples, memory, retrieval mechanisms, tools, state management, and control flow. The goal is to provide the model with all the necessary structured informational components at inference time to help it plausibly accomplish a given task.

### Research Highlights

*   **MEM1:** Learn to Synergize Memory and Reasoning for Efficient Long-Horizon Agents.
*   **IBM Zurich:** Eliciting Reasoning in Language Models with Cognitive Tools.
*   **Emergent Symbolic Mechanisms Support Abstract Reasoning in Large Language Models:** Unlock the ability of LLMs to work with abstract variables.

## Quick Start

1.  **Foundations:** Start by reading [`00_foundations/01_atoms_prompting.md`](00_foundations/01_atoms_prompting.md) (5 min) to understand the limitations of prompts.
2.  **Minimal Example:** Run [`10_guides_zero_to_hero/01_min_prompt.py`](10_guides_zero_to_hero/01_min_prompt.py) (Jupyter Notebook style) for a basic experiment.
3.  **Templates:** Explore [`20_templates/minimal_context.yaml`](20_templates/minimal_context.yaml) for a ready-to-use template.
4.  **Full Implementation:** Study the complete implementation with context management in [`30_examples/00_toy_chatbot/`](30_examples/00_toy_chatbot/).

## Contributing

We welcome your contributions! Please see [CONTRIBUTING.md](.github/CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the [MIT License](LICENSE).

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
> I've been looking forward to this being conceptualized and formalized as there wasn't a prior established field. Prompt engineering receives quite the stigma and doesn't quite cover what most researchers and I do.

- [Andrej Karpathy](https://x.com/karpathy/status/1937902205765607626) for coining "context engineering" and inspiring this repo 
- All contributors and the open source community