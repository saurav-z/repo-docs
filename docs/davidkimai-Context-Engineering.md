# Context Engineering: Master the Art & Science of LLM Context Design

**Unlock the full potential of Large Language Models (LLMs) by mastering context engineering – the art of crafting the perfect information payload for optimal LLM performance.** Learn how to go beyond basic prompt engineering and design, orchestrate, and optimize the "everything else" that LLMs see, enabling superior results.

[View the original repository on GitHub](https://github.com/davidkimai/Context-Engineering)

## Key Features

*   **Comprehensive & Cutting-Edge:** Stay ahead of the curve with a handbook built on first principles and the latest research from ICML, IBM, NeurIPS, OHBM, and more.
*   **Practical & Actionable:**  Move beyond theory with hands-on guides, working examples, and reusable templates.
*   **Progressive Learning Path:**  Follow a structured curriculum that builds from fundamentals to advanced techniques.
*   **Visual & Intuitive:**  Grasp complex concepts with clear diagrams, and code examples, inspired by leading thinkers like 3Blue1Brown.
*   **Community-Driven:**  Join a vibrant community on [Discord](https://discord.gg/JeFENHNNNQ) and contribute your knowledge.

## What is Context Engineering?

Context engineering is the practice of designing the complete information payload given to an LLM at inference time, encompassing all structured informational components, going beyond simple prompts.

```
                    Prompt Engineering  │  Context Engineering
                       ↓                │            ↓                      
               "What you say"           │  "Everything else the model sees"
             (Single instruction)       │    (Examples, memory, retrieval,
                                        │     tools, state, control flow)
```

## Why Context Engineering Matters

By mastering context engineering, you can dramatically improve LLM performance, reduce costs, and unlock advanced capabilities. The real power comes from engineering the **entire context window** that surrounds those prompts.

## Core Concepts Covered

*   **Token Budgeting:** Optimize your context window for cost and speed.
*   **Few-Shot Learning:**  Use examples to guide LLM behavior.
*   **Memory Systems:**  Enable stateful and coherent interactions.
*   **Retrieval Augmentation (RAG):** Ground responses in facts and reduce hallucinations.
*   **Control Flow:** Break down complex tasks for better results.
*   **Context Pruning:** Eliminate irrelevant information.
*   **Metrics & Evaluation:**  Measure and optimize context effectiveness.
*   **Cognitive Tools & Prompt Programming:**  Build custom tools.
*   **Neural Field Theory:** Model context as a dynamic neural field.
*   **Symbolic Mechanisms:** Implement symbolic architectures.
*   **Quantum Semantics:** Leverage superpositional techniques.

## Learning Path & Quick Start

1.  **[00_foundations/01_atoms_prompting.md](00_foundations/01_atoms_prompting.md)** (5 min): Understand why prompts alone often underperform.
2.  **[10_guides_zero_to_hero/01_min_prompt.py](10_guides_zero_to_hero/01_min_prompt.py)**: Experiment with a minimal working example.
3.  **[20_templates/minimal_context.yaml](20_templates/minimal_context.yaml)**: Copy/paste a template into your own project.
4.  **[30_examples/00_toy_chatbot/](30_examples/00_toy_chatbot/)**: See a complete implementation with context management.

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────┐
│ 00_foundations/ │     │ 10_guides_zero_  │     │ 20_templates/  │
│                 │────▶│ to_one/          │────▶│                │
│ Theory & core   │     │ Hands-on         │     │ Copy-paste     │
│ concepts        │     │ walkthroughs     │     │ snippets       │
└─────────────────┘     └──────────────────┘     └────────────────┘
         │                                                │
         │                                                │
         ▼                                                ▼
┌─────────────────┐                             ┌────────────────┐
│ 40_reference/   │◀───────────────────────────▶│ 30_examples/   │
│                 │                             │                │
│ Deep dives &    │                             │ Real projects, │
│ eval cookbook   │                             │ progressively  │
└─────────────────┘                             │ complex        │
         ▲                                      └────────────────┘
         │                                                ▲
         │                                                │
         └────────────────────┐               ┌───────────┘
                              ▼               ▼
                         ┌─────────────────────┐
                         │ 50_contrib/         │
                         │                     │
                         │ Community           │
                         │ contributions       │
                         └─────────────────────┘
```

## Research-Backed Insights

This repository is grounded in the latest research. Explore the key areas:

*   **[MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents](https://www.arxiv.org/pdf/2506.15841):**  Learn how agents blend memory and reasoning for efficiency.
*   **[Eliciting Reasoning in Language Models with Cognitive Tools](https://www.arxiv.org/pdf/2506.12115):** Explore how cognitive tools enhance reasoning capabilities.
*   **[Emergent Symbolic Mechanisms Support Abstract Reasoning in Large Language Models](https://openreview.net/forum?id=y1SnRPDWx4):** Understand how LLMs develop internal symbolic structures.

## Resources & Links

*   [Comprehensive Course Under Construction](https://github.com/davidkimai/Context-Engineering/tree/main/00_COURSE)
*   [Context Engineering Survey-Review of 1400 Research Papers](https://arxiv.org/pdf/2507.13334)
*   [Awesome Context Engineering Repo](https://github.com/Meirtz/Awesome-Context-Engineering)
*   [Discord](https://discord.gg/JeFENHNNNQ)
*   [`Agent Commands`](https://github.com/davidkimai/Context-Engineering/tree/main/.claude/commands)

## Contributing

We welcome contributions! See our [CONTRIBUTING.md](.github/CONTRIBUTING.md) for guidelines.

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