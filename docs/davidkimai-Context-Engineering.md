# Context Engineering: Master the Art of Information for LLMs

**Unlock the full potential of Large Language Models (LLMs) by mastering context engineering, the strategic design and optimization of information to guide LLMs towards desired outcomes. Explore the cutting edge of LLM interaction, moving beyond simple prompts to create truly intelligent systems.** [Visit the Original Repository](https://github.com/davidkimai/Context-Engineering)

## Key Features

*   **First-Principles Approach:** Learn context engineering from the ground up, starting with fundamental concepts and building towards advanced techniques.
*   **Hands-on Learning:** Practical examples, runnable code snippets, and clear visualizations to solidify your understanding.
*   **Comprehensive Course:**  A structured learning path covering foundations, system implementation, integration, and cutting-edge frontier research.
*   **Real-World Applications:**  Explore practical implementations, including a toy chatbot and example templates, to see context engineering in action.
*   **Up-to-Date Research:** Stay ahead of the curve with insights from the latest research papers and industry trends.

## What is Context Engineering?

> **"Context engineering is the delicate art and science of filling the context window with just the right information for the next step." — [Andrej Karpathy](https://x.com/karpathy/status/1937902205765607626)**

Context Engineering goes beyond simple prompt engineering. It encompasses the entire information payload given to an LLM, including prompts, examples, memory, retrieval mechanisms, tools, state management, and control flow. It's the art and science of shaping the LLM's environment to achieve specific goals.

## Learning Path

This repository offers a structured learning path:

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

## Core Concepts

| Concept | What It Is | Why It Matters |
|---|---|---|
| **Token Budget** | Optimizing every token in your context | More tokens = more $$ and slower responses |
| **Few-Shot Learning** | Teaching by showing examples | Often works better than explanation alone |
| **Memory Systems** | Persisting information across turns | Enables stateful, coherent interactions |
| **Retrieval Augmentation** | Finding & injecting relevant documents | Grounds responses in facts, reduces hallucination |
| **Control Flow** | Breaking complex tasks into steps | Solve harder problems with simpler prompts |
| **Context Pruning** | Removing irrelevant information | Keep only what's necessary for performance |
| **Metrics & Evaluation** | Measuring context effectiveness | Iterative optimization of token use vs. quality |
| **Cognitive Tools & Prompt Programming** | Learm to build custom tools and templates | Prompt programming enables new layers for context engineering |
| **Neural Field Theory** | Context as a Neural Field | Modeling context as a dynamic neural field allows for iterative context updating |
| **Symbolic Mechanisms** | Symbolic architectures enable higher order reasoning | Smarter systems = less work |
| **Quantum Semantics** |  Meaning as observer-dependent  | Design context systems leveraging superpositional techniques |

## Research Highlights

This repository incorporates findings from cutting-edge research papers, providing insights into the latest advancements in context engineering:

*   **MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents - Singapore-MIT June 2025:** Trains AI agents to keep only what matters—merging memory and reasoning at every step.
*   **Eliciting Reasoning in Language Models with Cognitive Tools - IBM Zurich June 2025:**  Breaks down complex tasks into modular "cognitive tools," like inner mental shortcuts, for greater accuracy and flexibility.
*   **Emergent Symbolic Mechanisms Support Abstract Reasoning in Large Language Models - ICML Princeton June 18, 2025:**  Large language models develop their own inner symbolic “logic circuits”—enabling them to reason with abstract variables, not just surface word patterns.

### Examples from IBM Zurich - Cognitive Tools

![Cognitive Tools](https://github.com/user-attachments/assets/cd06c3f5-5a0b-4ee7-bbba-2f9f243f70ae)
These structured prompt templates break down a complex question into modular blocks.

### Examples from IBM Zurich - Prompts and Prompt Programs

![Prompt Programs](https://github.com/user-attachments/assets/f7ce8605-6fa3-494f-94cd-94e6b23032b6)
Modular, specialized tools increase accuracy and facilitate complex reasoning within the LLM.

### Examples from ICML Princeton - Emergent Symbols

![Emergent Symbols](https://github.com/user-attachments/assets/76c6e6cb-b65d-4af7-95a5-6d52aee7efc0)

Symbolic induction heads, symbolic abstraction heads, and retrieval heads.

![Emergent Symbols - 2](https://github.com/user-attachments/assets/2428544e-332a-4e32-9070-9f9d8716d491)

## Quick Start

1.  **Explore Foundations:** Start with [`00_foundations/01_atoms_prompting.md`](00_foundations/01_atoms_prompting.md) (5 min) to understand the fundamentals.
2.  **Run a Minimal Example:** Experiment with [`10_guides_zero_to_hero/01_min_prompt.py`](10_guides_zero_to_hero/01_min_prompt.py).
3.  **Use Templates:** Copy/paste templates from [`20_templates/minimal_context.yaml`](20_templates/minimal_context.yaml).
4.  **Build and Test:** Study a complete implementation at [`30_examples/00_toy_chatbot/`](30_examples/00_toy_chatbot/).

## Quick Links

*   [Comprehensive Course Under Construction](https://github.com/davidkimai/Context-Engineering/tree/main/00_COURSE)
*   [`Agent Commands`](https://github.com/davidkimai/Context-Engineering/tree/main/.claude/commands)
*   [Context Engineering Survey-Review of 1400 Research Papers](https://arxiv.org/pdf/2507.13334)
*   [Awesome Context Engineering Repo](https://github.com/Meirtz/Awesome-Context-Engineering)
*   [![Discord](https://img.shields.io/badge/Discord-join%20chat-7289DA.svg?logo=discord")](https://discord.gg/JeFENHNNNQ)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=davidkimai/Context-Engineering&type=Date)](https://www.star-history.com/#davidkimai/Context-Engineering&Date)

## Contributing

We welcome contributions!  See [CONTRIBUTING.md](.github/CONTRIBUTING.md) for guidelines.

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

*   [Andrej Karpathy](https://x.com/karpathy/status/1937902205765607626) for coining "context engineering."
*   All contributors and the open-source community.