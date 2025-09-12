# Context Engineering: Unlock the Full Potential of LLMs

**Tired of basic prompts? Dive into Context Engineering, the art and science of crafting the perfect information payload to guide Large Language Models (LLMs) to peak performance.** Explore the cutting edge of LLM interaction, and learn how to optimize the entire context window to create powerful, intelligent systems.  [Explore the original repository here](https://github.com/davidkimai/Context-Engineering).

## Key Features

*   **Comprehensive Approach:** A first-principles guide to move beyond prompt engineering and master context design, orchestration, and optimization.
*   **Biological Metaphor:** Understand the progression from atoms (single prompts) to complex neural systems, employing a conceptual structure.
*   **Hands-on Learning:**  Includes practical examples, runnable code, and visual aids to solidify understanding.
*   **Cutting-Edge Research:** Incorporates the latest findings from leading AI research (ICML, IBM, NeurIPS, etc.) to keep you at the forefront.
*   **Community Focused:** Join the vibrant community to contribute, collaborate, and learn together.

## What is Context Engineering?

Context Engineering is a rapidly evolving discipline that moves beyond prompt engineering, focusing on how to structure and enrich the entire context window of an LLM to improve performance.

> "Context is not just the single prompt users send to an LLM. Context is the complete information payload provided to a LLM at inference time, encompassing all structured informational components that the model needs to plausibly accomplish a given task." – [Definition from A Systematic Analysis of Over 1400 Research Papers](https://arxiv.org/pdf/2507.13334)

## Core Concepts

The repository provides a roadmap to mastering the following:

*   **Token Budget:**  Optimize token usage to reduce cost and improve speed.
*   **Few-Shot Learning:** Use examples to boost LLM understanding and performance.
*   **Memory Systems:** Implement stateful interactions for coherent conversations and complex tasks.
*   **Retrieval Augmentation:**  Provide context by fetching relevant documents.
*   **Control Flow:** Organize complex tasks into manageable steps.
*   **Context Pruning:**  Refine context by removing irrelevant information.
*   **Metrics & Evaluation:**  Measure the impact of context design choices.
*   **Cognitive Tools & Prompt Programming:** Learn to build custom tools and templates.
*   **Neural Field Theory:** Use the context as a dynamic neural field to facilitate iterative context updating.
*   **Symbolic Mechanisms:** Build smarter systems using symbolic architectures.
*   **Quantum Semantics:** Design context systems leveraging superpositional techniques.

## Learning Path

The learning path is structured to guide you through the core concepts:

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

## Research-Driven Insights

This repository highlights the latest research, including:

*   [MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents - Singapore-MIT June 2025](https://www.arxiv.org/pdf/2506.15841)
*   [Eliciting Reasoning in Language Models with Cognitive Tools - IBM Zurich June 2025](https://www.arxiv.org/pdf/2506.12115)
*   [Emergent Symbolic Mechanisms Support Abstract Reasoning in Large Language Models - ICML Princeton June 18, 2025](https://openreview.net/forum?id=y1SnRPDWx4)

## How to Get Started

1.  **Foundations:** Start by reading `00_foundations/01_atoms_prompting.md` (5 min).
2.  **Hands-on:** Run `10_guides_zero_to_hero/01_min_prompt.py` (Jupyter Notebook).
3.  **Templates:** Explore `20_templates/minimal_context.yaml`.
4.  **Examples:** Review `30_examples/00_toy_chatbot/` for a complete implementation.

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](.github/CONTRIBUTING.md) for details.

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

*   [Andrej Karpathy](https://x.com/karpathy/status/1937902205765607626) for inspiring the name and concept.
*   The open source community.