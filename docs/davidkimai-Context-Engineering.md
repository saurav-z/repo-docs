# Context Engineering: Unlock the Power of the Context Window 

**Context Engineering** is the art and science of designing the perfect information environment for your Large Language Models, transforming prompts into powerful applications. Discover how to move beyond prompt engineering and master context design, orchestration, and optimization.  ([Original Repository](https://github.com/davidkimai/Context-Engineering))

## Key Features

*   **Comprehensive Course in Progress:** Access a detailed guide, incorporating the latest research from ICML, IBM, NeurIPS, and more, with a planned completion date of July 2025.
*   **AgenticOS Support:** Explore a wide range of models and tools, including Claude Code, OpenCode, Amp, Kiro, Codex, and Gemini CLI.
*   **First-Principles Approach:** A progressive, biological-inspired framework for understanding context engineering, moving from atoms to neural & semantic field theory.
*   **Hands-on Tutorials:** Get started quickly with practical guides and Jupyter Notebook examples.
*   **Extensive Code & Templates:** Access reusable code components and templates to accelerate your projects.
*   **Deep Dive Documentation:** Explore comprehensive documentation, including token optimization, retrieval strategies, and evaluation checklists.
*   **Community Contributions:** Join the growing community and contribute to the project.
*   **Cutting-edge Research**: Leverage recent findings like  "cognitive tools" and "emergent symbolic mechanisms" and "memory + reasoning"
*   **Focus on Emergence and Dynamical Systems Theory**: Learn to model context as a dynamic field that can dynamically change based on inputs.

## Why Context Engineering?

While prompt engineering focused on "what you say," context engineering is about "everything else the model sees"—examples, memory, retrieval, tools, state, and control flow. This repo provides a first-principles approach to context engineering, providing the building blocks you need to create powerful language model applications.

## Core Principles

*   **Abstraction:** Understand and leverage the cost of generalization.
*   **Iteration:** Add to the models capabilities in an iterative manner.
*   **Measurements and Metrics:** Understand how to test the new features to measure improvements.
*   **Visualization:** Use code to visualize concepts for better understanding.

## Learning Path

The learning path is a combination of:

1.  **Theory:** Deep dive into the fundamentals
2.  **Walkthroughs:** Hands-on practice.
3.  **Examples:** Real world applications.

## What You'll Learn

| Concept | What It Is | Why It Matters |
|---------|------------|----------------|
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

## Research Focus

This repository emphasizes the latest advancements in the field, including:

*   **[MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents - Singapore-MIT June 2025](https://www.arxiv.org/pdf/2506.15841)**

    *   MEM1 trains AI agents to keep only what matters—merging memory and reasoning at every step—so they never get overwhelmed, no matter how long the task.
    *   By blending memory and thinking into a single flow, MEM1 learns to remember only the essentials—making agents faster, sharper, and able to handle much longer conversations.
*   **[Eliciting Reasoning in Language Models with Cognitive Tools - IBM Zurich June 2025](https://www.arxiv.org/pdf/2506.12115)**

    *   This research shows that breaking complex tasks into modular “cognitive tools” lets AI solve problems more thoughtfully—mirroring how expert humans reason step by step.
    *   By compartmentalizing reasoning steps into modular blocks, these tools prevent confusion, reduce error, and make the model’s thought process transparent and auditable—even on hard math problems.
*   **[Emergent Symbolic Mechanisms Support Abstract Reasoning in Large Language Models - ICML Princeton June 18, 2025](https://openreview.net/forum?id=y1SnRPDWx4)**

    *   LLMs show a three-stage process: first abstracting symbols from input, then reasoning over these variables, and finally mapping the abstract answer back to real-world tokens.
    *   By running targeted experiments and interventions, the authors show these symbolic processes are both necessary and sufficient for abstract reasoning, across multiple models and tasks.

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
> I've been looking forward to this being conceptualized and formalized as there wasn't a prior established field. Prompt engineering receives quite the stigma and doesn't quite cover what most researchers and I do.

- [Andrej Karpathy](https://x.com/karpathy/status/1937902205765607626) for coining "context engineering" and inspiring this repo 
- All contributors and the open source community