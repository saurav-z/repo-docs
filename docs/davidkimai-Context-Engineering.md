# Context Engineering: Master the Art of Context Design for LLMs

**Context Engineering is the advanced discipline of designing the entire context window around your prompts to unlock the full potential of Large Language Models (LLMs).** Dive into the cutting edge of LLM research and learn how to engineer the context window for optimal performance. Explore the full potential of context engineering on [GitHub](https://github.com/davidkimai/Context-Engineering)!

## Key Features:

*   **First-Principles Approach:** Learn context engineering from the ground up with a progressive, biological metaphor.
*   **Practical Handbook:**  Move beyond basic prompt engineering with a comprehensive guide to context design, orchestration, and optimization.
*   **Hands-on Tutorials:**  Get started quickly with step-by-step guides and interactive examples.
*   **Reusable Templates:** Utilize pre-built components for common context engineering tasks.
*   **Deep-Dive Documentation:**  Explore advanced topics such as token budgeting, memory systems, and cognitive tools.
*   **Cutting-Edge Research:** Stay ahead of the curve with the latest research from ICML, IBM, and more.
*   **Community Contributions:** Collaborate and contribute to the evolving field of context engineering.

## Why Context Engineering?

Prompt engineering only scratches the surface.  This repository provides a first-principles approach to context engineering, offering a structured guide to mastering "everything else the model sees" (examples, memory, tools, and control flow).

### From Prompt Engineering to Context Engineering

| Prompt Engineering                      | Context Engineering                             |
| :------------------------------------- | :---------------------------------------------- |
| "What you say" (Single instruction) | "Everything else the model sees" (Examples, memory, retrieval, tools, state, control flow) |

## Core Concepts and Structure

This repository is structured around a progressive learning path, built using a biological metaphor:

```
atoms → molecules → cells → organs → neural systems → neural & semantic field theory 
  │        │         │         │             │                         │        
single    few-     memory +   multi-   cognitive tools +     context = fields +
prompt    shot     agents     agents   operating systems     persistence & resonance
```

### Levels of Understanding

Explore a structured learning path with four key levels:

*   **Level 1: Basic Context Engineering:**  Focuses on the fundamentals like single prompts, examples, and few-shot learning.
*   **Level 2: Field Theory:**  Delves into memory, state management, and multi-agent systems.
*   **Level 3: Protocol Systems:** Introduces reasoning frameworks, cognitive tools, and control flow.
*   **Level 4: Meta-Recursion:**  Explores meta-recursive frameworks for recursive improvement and self-reflection.

## Learning Path Overview

The repository offers a guided learning path:

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

## What You'll Learn

| Concept                    | What It Is                                              | Why It Matters                                                      |
| :------------------------- | :------------------------------------------------------- | :------------------------------------------------------------------ |
| **Token Budgeting**      | Optimizing token usage in your context               |  Reduce costs and improve response times.                           |
| **Few-Shot Learning**      | Teaching by demonstration and examples                  | Often more effective than explanations.                            |
| **Memory Systems**         | Persisting information across interactions           | Enables stateful, coherent conversations.                       |
| **Retrieval Augmentation** | Finding & injecting relevant documents              | Grounds responses in facts, reduces hallucinations.                 |
| **Control Flow**           | Breaking tasks into manageable steps                   | Enables solving complex problems through orchestration.      |
| **Context Pruning**        | Removing irrelevant information                      |  Enhance model performance and reduce noise.                       |
| **Metrics & Evaluation**   | Measuring context effectiveness                       | Enable iterative optimization of performance.       |
| **Cognitive Tools**        | Building custom tools and templates                    |  Adds layers to the process of context engineering.                         |
| **Neural Field Theory** | Context as a Neural Field  | Modeling context as a dynamic neural field allows for iterative context updating |
| **Symbolic Mechanisms** | Symbolic architectures enable higher order reasoning | Smarter systems = less work |
| **Quantum Semantics** |  Meaning as observer-dependent  | Design context systems leveraging superpositional techniques |


## Research Insights

*   **[MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents - Singapore-MIT](https://www.arxiv.org/pdf/2506.15841):** Focuses on merging memory and reasoning for long-horizon agents, emphasizing memory consolidation and efficiency.
*   **[Eliciting Reasoning in Language Models with Cognitive Tools - IBM Zurich](https://www.arxiv.org/pdf/2506.12115):** Introduces "cognitive tools" as prompt templates to break down tasks, enhance reasoning, and improve model transparency.
*   **[Emergent Symbolic Mechanisms Support Abstract Reasoning in Large Language Models - ICML Princeton](https://openreview.net/forum?id=y1SnRPDWx4):** Shows LLMs develop their own symbolic "logic circuits" for abstraction and reasoning, bridging the gap between symbolic AI and neural networks.

## Contributing

We welcome contributions! Please review our [CONTRIBUTING.md](.github/CONTRIBUTING.md) for guidelines.

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

Special thanks to [Andrej Karpathy](https://x.com/karpathy/status/1937902205765607626) for the inspiration and to all contributors and the open-source community.