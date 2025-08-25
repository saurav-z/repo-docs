# Context Engineering: Master the Art and Science of LLM Context

**Go beyond prompt engineering and learn to design, orchestrate, and optimize the complete context window for powerful AI applications.** Explore cutting-edge research, practical examples, and a structured learning path to master the emerging field of context engineering.

[View the original repository on GitHub](https://github.com/davidkimai/Context-Engineering)

## What is Context Engineering?

> “Context engineering is the delicate art and science of filling the context window with just the right information for the next step.” — Andrej Karpathy

Context engineering is the practice of constructing the "everything else" that the model sees to improve performance and achieve specific outcomes. This goes beyond simple prompting and delves into the strategic use of:
*   **Examples:** Few-shot learning
*   **Memory:** Persistent information
*   **Retrieval:** Relevant document injection
*   **Tools:** Custom tool building
*   **State & Control Flow:** Multi-step processes and specialized components

## Key Features

*   **First-Principles Approach:** Learn the foundational concepts and build from there.
*   **Practical Examples:** Run code and see how to use it.
*   **Visualizations:** Understand complex ideas with diagrams and analogies.
*   **Cutting-Edge Research:** Access the latest research papers on the cutting edge of context design, orchestration, and optimization.
*   **Progressive Learning Path:** The repository is built to follow a biological metaphor: atoms → molecules → cells → organs → neural systems → neural & semantic field theory 

## Learning Path

The repository is structured to guide you from basic concepts to advanced techniques.

*   **00\_foundations/:** Introduction to core concepts and theory.
*   **10\_guides\_zero\_to\_one/:** Hands-on walkthroughs and practical guides.
*   **20\_templates/:** Copy-paste snippets and pre-built templates.
*   **30\_examples/:** Complete, real-world project implementations.
*   **40\_reference/:** Deep dives, evaluation techniques, and a cookbook.
*   **50\_contrib/:** Community contributions.

## What You'll Learn

| Concept             | What It Is                                          | Why It Matters                                                                                                          |
| ------------------- | --------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **Token Budget**    | Optimizing every token in your context              | More tokens = more $$ and slower responses                                                                            |
| **Few-Shot Learning** | Teaching by providing examples                     | Often works better than explanation alone                                                                               |
| **Memory Systems**  | Persisting information across turns                  | Enables stateful, coherent interactions                                                                               |
| **Retrieval Augmentation** | Finding & injecting relevant documents              | Grounds responses in facts, reduces hallucination                                                                            |
| **Control Flow**    | Breaking complex tasks into steps                  | Solve harder problems with simpler prompts                                                                               |
| **Context Pruning**   | Removing irrelevant information                    | Keep only what's necessary for performance                                                                              |
| **Metrics & Evaluation**  | Measuring context effectiveness                 | Iterative optimization of token use vs. quality                                                                         |
| **Cognitive Tools & Prompt Programming**   | Learm to build custom tools and templates                 | Prompt programming enables new layers for context engineering                                                                         |
| **Neural Field Theory**    | Context as a Neural Field                   | Modeling context as a dynamic neural field allows for iterative context updating                                                                         |
| **Symbolic Mechanisms**    | Symbolic architectures enable higher order reasoning                  | Smarter systems = less work                                                                        |
| **Quantum Semantics**   |  Meaning as observer-dependent  | Design context systems leveraging superpositional techniques                                                                          |

## Quick Start

1.  **Understand the basics:** Read [`00_foundations/01_atoms_prompting.md`] (5 min).
2.  **Experiment:** Run [`10_guides_zero_to_hero/01_min_prompt.py`] (Jupyter Notebook style).
3.  **Build:** Explore and use [`20_templates/minimal_context.yaml`].
4.  **Implement:** Study [`30_examples/00_toy_chatbot/`] to see a complete implementation.

## Research Insights

Explore recent research to understand emerging methods, the value of cognitive tools, and how LLMs can learn their own inner symbolic logic circuits.

### Memory + Reasoning

*   **[MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents - Singapore-MIT June 2025](https://www.arxiv.org/pdf/2506.15841)** 
    *   Trains AI agents to merge memory and reasoning at every step.

### Cognitive Tools

*   **[Eliciting Reasoning in Language Models with Cognitive Tools - IBM Zurich June 2025](https://www.arxiv.org/pdf/2506.12115)** 
    *   Breaking complex tasks into modular “cognitive tools” enables AI models to reason like expert humans.

### Emergent Symbols

*   **[Emergent Symbolic Mechanisms Support Abstract Reasoning in Large Language Models - ICML Princeton June 18, 2025](https://openreview.net/forum?id=y1SnRPDWx4)**
    *   LLMs are shown to develop their own inner symbolic “logic circuits”—enabling them to reason with abstract variables.

## Contributing

Contribute to this growing resource! See the [CONTRIBUTING.md](.github/CONTRIBUTING.md) file for guidelines.

## License

MIT License ([LICENSE](LICENSE))

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

Special thanks to:

*   [Andrej Karpathy](https://x.com/karpathy/status/1937902205765607626) for coining "context engineering" and inspiring this repo.
*   All contributors and the open-source community.