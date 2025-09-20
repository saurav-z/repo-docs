# Context Engineering: Master the Art of Information for Next-Level AI

**Unlock the true potential of large language models by mastering context engineering – the strategic design and orchestration of information within the context window. [Explore the original repository](https://github.com/davidkimai/Context-Engineering).**

> *"Context engineering is the delicate art and science of filling the context window with just the right information for the next step." – [Andrej Karpathy](https://x.com/karpathy/status/1937902205765607626)*

## Key Features

*   **Comprehensive Guide:** A first-principles approach to context design, moving beyond prompt engineering.
*   **Visual Learning:** Concepts are explained with clear visuals, diagrams, and code examples.
*   **Progressive Learning Path:** Structured learning through foundations, guides, examples, and reference materials.
*   **Community-Driven:** Contributions from the open-source community.
*   **Cutting-Edge Research:** Incorporates the latest findings from leading AI research, including:
    *   [MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents - Singapore-MIT June 2025](https://www.arxiv.org/pdf/2506.15841)
    *   [Eliciting Reasoning in Language Models with Cognitive Tools - IBM Zurich June 2025](https://www.arxiv.org/pdf/2506.12115)
    *   [Emergent Symbolic Mechanisms Support Abstract Reasoning in Large Language Models - ICML Princeton June 18, 2025](https://openreview.net/forum?id=y1SnRPDWx4)

## Why Context Engineering Matters

Prompt engineering alone is not enough. Context engineering is the discipline of designing, orchestrating, and optimizing all the information a model *sees* at inference time — examples, memory, tools, and control flow. This repository teaches you how to build context like a biological system:

```
atoms → molecules → cells → organs → neural systems → neural & semantic field theory 
  │        │         │         │             │                         │        
single    few-     memory +   multi-   cognitive tools +     context = fields +
prompt    shot     agents     agents   operating systems     persistence & resonance
```

## Quick Start

1.  **Start Here:** [`00_foundations/01_atoms_prompting.md`](00_foundations/01_atoms_prompting.md) (5 min) - Understand the fundamentals.
2.  **Get Hands-on:**  [`10_guides_zero_to_hero/01_min_prompt.py`](10_guides_zero_to_hero/01_min_prompt.py) - Experiment with a minimal example.
3.  **Explore Templates:** [`20_templates/minimal_context.yaml`](20_templates/minimal_context.yaml) - Customize existing templates.
4.  **See it in Action:** [`30_examples/00_toy_chatbot/`](30_examples/00_toy_chatbot/) - Examine a complete implementation.

## Learning Path

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

| Concept                      | What It Is                                  | Why It Matters                                        |
| :--------------------------- | :------------------------------------------ | :---------------------------------------------------- |
| **Token Budget**             | Optimizing context token usage              | Reduces cost, increases speed                        |
| **Few-Shot Learning**        | Learning through examples                   | Boosts performance without retraining              |
| **Memory Systems**           | Persisting information across turns       | Enables stateful and coherent interactions           |
| **Retrieval Augmentation**   | Injecting relevant documents                | Grounds responses in facts, combats hallucinations |
| **Control Flow**             | Breaking down complex tasks                | Enables solving more challenging problems            |
| **Context Pruning**          | Removing irrelevant information             | Improves performance                                 |
| **Metrics & Evaluation**     | Measuring context effectiveness             | Enables data-driven optimization                      |
| **Cognitive Tools & Prompt Programming**  | Building custom tools and templates       | New Layers for context engineering                    |
| **Neural Field Theory**        | Context as a Neural Field | Dynamic context updating                            |
| **Symbolic Mechanisms**      | Symbolic architectures for higher order reasoning | Smarter systems                                               |
| **Quantum Semantics** |  Meaning as observer-dependent  | Design context systems leveraging superpositional techniques |

## Research-Backed Insights

This repository incorporates key findings from recent AI research, including:

*   **Memory + Reasoning:** MEM1 (Singapore-MIT, June 2025) explores agents that integrate memory and reasoning for long-horizon tasks.
*   **Cognitive Tools:**  Research from IBM Zurich (June 2025) demonstrates how using modular cognitive tools within LLMs boosts reasoning capabilities.
*   **Emergent Symbols:**  ICML Princeton (June 18, 2025) reveals how LLMs develop internal symbolic logic circuits.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=davidkimai/Context-Engineering&type=Date)](https://www.star-history.com/#davidkimai/Context-Engineering&Date)

## Contributing

Join the community! Read the [CONTRIBUTING.md](.github/CONTRIBUTING.md) for contribution guidelines.

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

Special thanks to [Andrej Karpathy](https://x.com/karpathy/status/1937902205765607626) for the inspiration.