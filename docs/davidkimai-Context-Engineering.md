# Context Engineering: Unlock the Power of LLMs Beyond Prompting

**Go beyond prompt engineering and master the art and science of shaping the entire context window to achieve unparalleled results with Large Language Models (LLMs).**  [Explore the original repository](https://github.com/davidkimai/Context-Engineering)

## Key Features

*   **Comprehensive Handbook:** A first-principles guide for moving beyond prompt engineering.
*   **Practical Examples:** Code-driven approach with runnable examples and templates.
*   **Cutting-Edge Research:** Grounded in the latest findings from leading AI research.
*   **Biological Metaphor:** Context engineering presented with intuitive concepts: atoms -> molecules -> cells -> organs -> neural systems.
*   **Community-Driven:** Contributions welcome to foster collaboration and innovation.

## Introduction

> *"Context engineering is the delicate art and science of filling the context window with just the right information for the next step."* — [Andrej Karpathy](https://x.com/karpathy/status/1937902205765607626)

This repository provides a comprehensive approach to context engineering, going beyond prompt engineering by focusing on the entire context window: the examples, memory, retrieval, tools, state, and control flow provided to the LLM.

## What is Context Engineering?

Context engineering is the strategic design, orchestration, and optimization of all informational components provided to an LLM during inference.

```
                    Prompt Engineering  │  Context Engineering
                       ↓                │            ↓
               "What you say"           │  "Everything else the model sees"
             (Single instruction)       │    (Examples, memory, retrieval,
                                        │     tools, state, control flow)
```

## Learning Path

Follow a structured path to master context engineering:

*   **Foundations:** Key concepts and theories.
*   **Hands-on Guides:** Step-by-step walkthroughs.
*   **Templates:** Reusable snippets for projects.
*   **Examples:** Complete implementations of real-world systems.
*   **Deep Dives:** Advanced explorations of key areas.
*   **Community Contributions:** Share knowledge and collaborate.

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

## Why This Repository Exists

Prompt engineering received all the attention, but we can now get excited for what comes next. Once you've mastered prompts, the real power comes from engineering the **entire context window** that surrounds those prompts. Guiding thought, if you will.

This repository provides a progressive, first-principles approach to context engineering, built around a biological metaphor:

```
atoms → molecules → cells → organs → neural systems → neural & semantic field theory
  │        │         │         │             │                         │
single    few-     memory +   multi-   cognitive tools +     context = fields +
prompt    shot     agents     agents   operating systems     persistence & resonance
```
> "Abstraction is the cost of generalization"— [**Grant Sanderson (3Blue1Brown)**](https://www.3blue1brown.com/)

## Key Concepts to Master

| Concept                    | What It Is                                       | Why It Matters                                          |
| -------------------------- | ------------------------------------------------ | ------------------------------------------------------- |
| **Token Budget**           | Optimizing every token in your context           | More tokens = more $$ and slower responses             |
| **Few-Shot Learning**      | Teaching by showing examples                      | Often works better than explanation alone                |
| **Memory Systems**         | Persisting information across turns              | Enables stateful, coherent interactions                 |
| **Retrieval Augmentation** | Finding & injecting relevant documents           | Grounds responses in facts, reduces hallucination      |
| **Control Flow**           | Breaking complex tasks into steps                | Solve harder problems with simpler prompts             |
| **Context Pruning**        | Removing irrelevant information                  | Keep only what's necessary for performance            |
| **Metrics & Evaluation**   | Measuring context effectiveness                  | Iterative optimization of token use vs. quality       |
| **Cognitive Tools & Prompt Programming** | Build custom tools and templates           | Prompt programming enables new layers for context engineering |
| **Neural Field Theory**    | Context as a Neural Field                      | Modeling context as a dynamic neural field              |
| **Symbolic Mechanisms**      | Symbolic architectures enable higher order reasoning | Smarter systems = less work |
| **Quantum Semantics**      | Meaning as observer-dependent | Design context systems leveraging superpositional techniques |

## Quick Start

1.  **Learn:**  `00_foundations/01_atoms_prompting.md` (5 min) - Understand why prompts alone underperform
2.  **Experiment:** Run `10_guides_zero_to_hero/01_min_prompt.py` (Jupyter Notebook style)
3.  **Implement:** Explore `20_templates/minimal_context.yaml` - Copy/paste a template
4.  **Explore:**  `30_examples/00_toy_chatbot/`  - Complete implementation

## Research Spotlight: The Future of Context Engineering

### 1. **MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents**
*   MEM1 agents keep only what matters—merging memory and reasoning.
*   Compress each interaction into a compact "internal state" and blend memory and thinking.
*   By blending memory and thinking into a single flow, MEM1 learns to remember only the essentials—making agents faster, sharper, and able to handle much longer conversations.

### 2. **Eliciting Reasoning in Language Models with Cognitive Tools - IBM Zurich June 2025**
*   Breaking complex tasks into modular “cognitive tools” lets AI solve problems more thoughtfully.
*   The model calls specialized prompt templates, aka cognitive tools like “understand question,” “recall related,” “examine answer,” and “backtracking”—each handling a distinct mental operation.
*   Cognitive tools work like inner mental shortcuts.

### 3. **Emergent Symbolic Mechanisms Support Abstract Reasoning in Large Language Models - ICML Princeton June 18, 2025**
*   LLMs develop their own inner symbolic “logic circuits”—enabling them to reason with abstract variables, not just surface word patterns.
*   LLMs show a three-stage process: first abstracting symbols from input, then reasoning over these variables, and finally mapping the abstract answer back to real-world tokens.
*   These emergent mechanisms mean LLMs don’t just memorize—they actually create internal, flexible representations that let them generalize to new problems and analogies.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=davidkimai/Context-Engineering&type=Date)](https://www.star-history.com/#davidkimai/Context-Engineering&Date)

## Contributing

We welcome contributions! Review the [CONTRIBUTING.md](.github/CONTRIBUTING.md) for guidelines.

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

*   Inspired by [Andrej Karpathy](https://x.com/karpathy/status/1937902205765607626).
*   Thanks to all contributors and the open-source community.