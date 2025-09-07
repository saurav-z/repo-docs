# Context Engineering: Beyond Prompt Engineering

**Master the art and science of crafting the perfect context for Large Language Models (LLMs) to unlock their full potential.  Explore the repository: [davidkimai/Context-Engineering](https://github.com/davidkimai/Context-Engineering)**

<div align="center">
  
## Table of Contents

*   [Key Features](#key-features)
*   [What is Context Engineering?](#definition-of-context-engineering)
*   [Why This Repository Exists](#why-this-repository-exists)
*   [Learning Path](#learning-path)
*   [What You'll Learn](#what-youll-learn)
*   [Research Evidence](#research-evidence)
*   [Quick Start](#quick-start)
*   [Star History](#star-history)
*   [Contributing](#contributing)
*   [License](#license)
*   [Citation](#citation)
*   [Acknowledgements](#acknowledgements)
</div>

## Key Features

*   **First-Principles Approach:**  Learn context engineering from the ground up, building a strong foundation.
*   **Practical Examples:**  Runnable code examples and templates to accelerate your learning and implementation.
*   **Visualizations:**  Understand complex concepts with clear diagrams and analogies.
*   **Community Driven:**  Contribute and learn from a growing community of context engineering enthusiasts.
*   **Cutting-Edge Research:**  Stay informed with links to the latest research papers and breakthroughs.

## Definition of Context Engineering

> Context is not just the single prompt users send to an LLM. Context is the complete information payload provided to a LLM at inference time, encompassing all structured informational components that the model needs to plausibly accomplish a given task.

## Why This Repository Exists

Prompt engineering set the stage, but **Context Engineering focuses on shaping the entire context window to extract the most from LLMs.** This repository goes beyond simple prompts, providing a comprehensive guide to context design, orchestration, and optimization using biological metaphors (atoms to neural systems)

## Learning Path

This resource provides a structured approach for context engineering:

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

Master key concepts for optimizing LLM interactions:

*   **Token Budget:** Optimize token usage for cost and speed.
*   **Few-Shot Learning:** Guide models with examples.
*   **Memory Systems:** Enable stateful interactions.
*   **Retrieval Augmentation:** Ground responses with facts.
*   **Control Flow:** Structure complex tasks into steps.
*   **Context Pruning:** Remove irrelevant information.
*   **Metrics & Evaluation:**  Measure context effectiveness.
*   **Cognitive Tools & Prompt Programming:** Build custom tools and templates.
*   **Neural Field Theory:** Model context as a dynamic neural field.
*   **Symbolic Mechanisms:** Enable higher order reasoning.
*   **Quantum Semantics:**  Design context systems leveraging superpositional techniques.

## Research Evidence

This repository is influenced by cutting edge research. Below are highlights.

### **[MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents - Singapore-MIT June 2025](https://www.arxiv.org/pdf/2506.15841)**

*   Trains AI agents to merge memory and reasoning.
*   Compresses interactions into a compact "internal state".
*   Blends memory and thinking for efficiency.
*   Structured, auditable actions and information.
*   Prunes old clutter to retain the most relevant insights.
*   Recursive, protocol-driven memory outperforms traditional methods.

### **[Eliciting Reasoning in Language Models with Cognitive Tools - IBM Zurich June 2025](https://www.arxiv.org/pdf/2506.12115)**

*   Breaks complex tasks into modular "cognitive tools"
*   Model calls specialized prompt templates (cognitive tools).
*   Tool calls plan reasoning and actions.
*   Compartmentalization prevents confusion.
*   Modular approach boosts real-world problem-solving.

### **[Emergent Symbolic Mechanisms Support Abstract Reasoning in Large Language Models - ICML Princeton June 18, 2025](https://openreview.net/forum?id=y1SnRPDWx4)**

*   LLMs develop internal symbolic "logic circuits".
*   Three-stage process: abstraction, reasoning, retrieval.
*   LLMs create internal representations.
*   Attention heads act as "symbol extractors".
*   Symbolic processes support abstract reasoning.

## Quick Start

1.  **Read `00_foundations/01_atoms_prompting.md` (5 min)**: Understand basic prompting.
2.  **Run `10_guides_zero_to_hero/01_min_prompt.py`**: Experiment with a minimal working example.
3.  **Explore `20_templates/minimal_context.yaml`**: Copy/paste a template.
4.  **Study `30_examples/00_toy_chatbot/`**: Explore a complete implementation.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=davidkimai/Context-Engineering&type=Date)](https://www.star-history.com/#davidkimai/Context-Engineering&Date)

## Contributing

See [CONTRIBUTING.md](.github/CONTRIBUTING.md) for guidelines.

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

*   [Andrej Karpathy](https://x.com/karpathy/status/1937902205765607626) for coining "context engineering".
*   All contributors and the open-source community.