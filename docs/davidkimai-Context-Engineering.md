# Context Engineering: Master the Art and Science of LLM Context Design

**Go beyond prompt engineering and unlock the full potential of Large Language Models (LLMs) with context engineering, the key to crafting effective and intelligent AI systems.** Learn to guide thought in LLMs. Explore the original repository: [davidkimai/Context-Engineering](https://github.com/davidkimai/Context-Engineering).

## Key Features

*   **Comprehensive Course:** A structured, in-depth course covering the foundations, system implementation, integration, and frontier applications of context engineering.
*   **AgenticOS Support:** Integration with leading LLM platforms and tools, including Claude Code, OpenCode, Amp, Kiro, Codex, and Gemini CLI.
*   **Practical Examples:** Real-world implementations and hands-on tutorials to guide you from basic concepts to advanced techniques.
*   **Modular Approach:** A first-principles approach that takes you from the atomic level (single prompts) to advanced methodologies involving agents, memory, and neural field theory.
*   **Research-Driven:** Operationalizing cutting-edge research with a focus on the latest advancements in context engineering from leading institutions.

## What is Context Engineering?

> "Context engineering is the delicate art and science of filling the context window with just the right information for the next step." — [Andrej Karpathy](https://x.com/karpathy/status/1937902205765607626)

Context Engineering is a discipline beyond prompt engineering, focusing on how to provide the optimal context to Large Language Models (LLMs). Context includes everything the model "sees" at inference time - examples, memory, retrieval, tools, state, and control flow. This provides an alternative to the existing approaches for LLM training, which are often time-consuming, expensive, and lack the ability to improve in real-time. This repository aims to teach the first principles of context design, orchestration, and optimization.

## Core Concepts

*   **Prompt Engineering vs. Context Engineering:** Understand the difference between what you say (single prompts) and everything else the model sees (examples, memory, tools, state, control flow).
*   **Mathematical Foundations:** Learn the mathematical underpinnings of context engineering with a focus on C = A(c₁, c₂, ..., cₙ).
*   **Progressive Learning Path:** A step-by-step guide that takes you from foundations to advanced topics such as Meta-Recursion and Quantum Semantics.

## Learning Path

The learning path progresses through these levels:

1.  **Atoms (Single Instructions)**: Learn the basics and understand prompting.
2.  **Molecules (Few-Shot)**: Utilize examples and few-shot prompting.
3.  **Cells (Persistent Memory)**: Implement conversational chatbots and conversational memory.
4.  **Organs (Multi-Agent Systems)**: Design multi-step control flows and use orchestration
5.  **Neural Systems (Cognitive Tools)**: Implement tools like Chain-of-Thought reasoning.
6.  **Neural & Semantic Field Theory:** Explore continuous meaning and attractor resonance.

## Repository Structure

The repository is structured to help you learn and apply context engineering techniques:

*   **`00_foundations/`:** First-principles theory on all the concepts.
*   **`10_guides_zero_to_hero/`:** Hands-on tutorials that allow you to experiment in a Jupyter Notebook style.
*   **`20_templates/`:** Reusable code snippets and configurations.
*   **`30_examples/`:** Practical implementations of real-world projects.
*   **`40_reference/`:** Deep dives and evaluation methods.
*   **`50_contrib/`:** Contribution guidelines and how to add to the project.
*   **`60_protocols/`:** Protocol shells and frameworks.
*   **`70_agents/`:** Agent demonstrations
*   **`80_field_integration/`:** Complete projects using field theory.
*   **`cognitive-tools/`:**  Advanced cognitive framework including understanding, reasoning, and verification.

## Key Takeaways

| Concept                   | What It Is                                      | Why It Matters                                     |
| :------------------------ | :---------------------------------------------- | :------------------------------------------------- |
| **Token Budgeting**       | Optimizing every token in your context          | More tokens = more $$ and slower responses         |
| **Few-Shot Learning**     | Teaching by showing examples                   | Often works better than explanation alone          |
| **Memory Systems**        | Persisting information across turns           | Enables stateful, coherent interactions             |
| **Retrieval Augmentation** | Finding & injecting relevant documents         | Grounds responses in facts, reduces hallucination |
| **Control Flow**          | Breaking complex tasks into steps              | Solve harder problems with simpler prompts          |
| **Context Pruning**       | Removing irrelevant information               | Keep only what's necessary for performance         |
| **Metrics & Evaluation**  | Measuring context effectiveness                | Iterative optimization of token use vs. quality    |
| **Cognitive Tools & Prompt Programming** | Build custom tools and templates    | Prompt programming enables new layers for context engineering |
| **Neural Field Theory**   | Context as a dynamic neural field             | Modeling context as a dynamic neural field allows for iterative context updating |
| **Symbolic Mechanisms** | Symbolic architectures enable higher order reasoning | Smarter systems = less work |
| **Quantum Semantics** |  Meaning as observer-dependent  | Design context systems leveraging superpositional techniques |

## Inspired by Experts

*   **Inspired by [Andrej Karpathy](https://x.com/karpathy/status/1937902205765607626)** – start with the fundamentals of context and move on to a higher order.
*   **3Blue1Brown Inspired style** – Every concept visualized.

## Research Focus

This project is based on the most recent research from:

*   **[MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents - Singapore-MIT June 2025](https://www.arxiv.org/pdf/2506.15841)**
*   **[Eliciting Reasoning in Language Models with Cognitive Tools - IBM Zurich June 2025](https://www.arxiv.org/pdf/2506.12115)**
*   **[Emergent Symbolic Mechanisms Support Abstract Reasoning in Large Language Models - ICML Princeton June 18, 2025](https://openreview.net/forum?id=y1SnRPDWx4)**
*   **[A Systematic Analysis of Over 1400 Research Papers on Context Engineering](https://arxiv.org/pdf/2507.13334)**

## Contributing

We welcome contributions! Check out [.github/CONTRIBUTING.md](.github/CONTRIBUTING.md) for guidelines.

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

- [Andrej Karpathy](https://x.com/karpathy/status/1937902205765607626) for coining "context engineering" and inspiring this repo
- All contributors and the open source community