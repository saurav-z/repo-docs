# Context Engineering: Unlock the Power of LLMs with Optimized Context

**Tired of basic prompt engineering? Dive into Context Engineering, the cutting-edge approach to designing, orchestrating, and optimizing the *entire* context window for Large Language Models, leading to dramatically improved performance and results.**  Explore the original repository here: [Context Engineering](https://github.com/davidkimai/Context-Engineering).

## Key Features & Benefits:

*   **Comprehensive Approach:** Move beyond simple prompts to master the full context window, including examples, memory, retrieval, tools, state, and control flow.
*   **First-Principles Handbook:** Learn a progressive, first-principles approach to context engineering.
*   **Actionable Guidance:**  Find practical code examples, hands-on tutorials, and reusable templates for immediate implementation.
*   **Research-Driven:** Access cutting-edge research from leading institutions like IBM, Princeton, and MIT.
*   **Community Focused:**  Join a collaborative community and contribute to the advancement of this emerging field.

## What is Context Engineering?

Context Engineering is the art and science of providing the optimal information payload to a Large Language Model (LLM) at inference time. This includes everything the model "sees": prompts, examples, memory, retrieval, tools, state, and control flow – far beyond just the single prompt.

**The core difference:**

*   **Prompt Engineering:** Focuses on "What you say" (a single instruction).
*   **Context Engineering:** Focuses on "Everything else the model sees" (examples, memory, tools, state, and control flow).

## Why This Matters

Prompt engineering has been the focus, but mastering the entire context window around the prompts is the key to unlocking an LLM's true potential. This repository offers a first-principles approach, built with a biological metaphor.

## Core Concepts & Learning Path

This project provides a comprehensive learning path to guide you, from basic concepts to advanced techniques.

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

### Quick Start

1.  **Start here:** `00_foundations/01_atoms_prompting.md` (5 min) - Understand why prompts alone often underperform.
2.  **Experiment:** Run `10_guides_zero_to_one/01_min_prompt.py` (Jupyter Notebook style).
3.  **Template it:** Explore `20_templates/minimal_context.yaml`.
4.  **See it:** Study `30_examples/00_toy_chatbot/` - a complete implementation with context management.

### Learning Path Overview

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

## Cutting-Edge Research & Resources

This repository is built on the latest research. Here are some key papers and resources:

*   **[Comprehensive Course Under Construction](https://github.com/davidkimai/Context-Engineering/tree/main/00_COURSE)**
*   **[Context Engineering Survey-Review of 1400 Research Papers](https://arxiv.org/pdf/2507.13334)**
*   **[Awesome Context Engineering Repo](https://github.com/Meirtz/Awesome-Context-Engineering)**
*   **[MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents - Singapore-MIT June 2025](https://www.arxiv.org/pdf/2506.15841)**
*   **[Eliciting Reasoning in Language Models with Cognitive Tools - IBM Zurich June 2025](https://www.arxiv.org/pdf/2506.12115)**
*   **[Emergent Symbolic Mechanisms Support Abstract Reasoning in Large Language Models - ICML Princeton June 18, 2025](https://openreview.net/forum?id=y1SnRPDWx4)**

## Contributing

We welcome contributions! Review the guidelines in [CONTRIBUTING.md](.github/CONTRIBUTING.md).

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

*   Inspired by Andrej Karpathy's coining of "context engineering" and [his work](https://x.com/karpathy/status/1937902205765607626)
*   Thanks to all contributors and the open-source community.