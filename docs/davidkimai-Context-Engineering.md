# Context Engineering: Master the Art and Science of LLM Context

**Unlock the full potential of Large Language Models by mastering context engineering, moving beyond prompts to orchestrate and optimize the *entire* information payload the model sees.** Explore the original repository: [davidkimai/Context-Engineering](https://github.com/davidkimai/Context-Engineering)

> *"Context engineering is the delicate art and science of filling the context window with just the right information for the next step."* — Andrej Karpathy

<img width="1600" height="400" alt="Context Engineering Image" src="https://github.com/user-attachments/assets/f41f9664-b707-4291-98c8-5bab3054a572" />

## Key Features

*   **Comprehensive Course Under Construction:** A first-principles approach to context design, orchestration, and optimization.
*   **Actionable Examples:**  Learn through runnable code, examples, and practical applications.
*   **Research-Driven:** Explore cutting-edge research on Memory, Reasoning, and Emergent Symbols.
*   **Visual Learning:**  Concepts are explained through diagrams, code, and clear explanations.
*   **Community Focus:**  Join the [Discord](https://discord.gg/JeFENHNNNQ) to connect, collaborate, and contribute.

## What is Context Engineering?

Context Engineering isn't just about the prompts. It's about the complete information payload the model receives at inference time, including examples, memory, retrieval, tools, state, and control flow.

```
                    Prompt Engineering  │  Context Engineering
                       ↓                │            ↓                      
               "What you say"           │  "Everything else the model sees"
             (Single instruction)       │    (Examples, memory, retrieval,
                                        │     tools, state, control flow)
```

## Explore the Future: Research Highlights

Dive into the latest research papers and publications from leading AI institutions:

*   [Context Engineering Survey-Review of 1400 Research Papers](https://arxiv.org/pdf/2507.13334)
*   [MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents - Singapore-MIT June 2025](https://www.arxiv.org/pdf/2506.15841)
*   [Eliciting Reasoning in Language Models with Cognitive Tools - IBM Zurich June 2025](https://www.arxiv.org/pdf/2506.12115)
*   [Emergent Symbolic Mechanisms Support Abstract Reasoning in Large Language Models - ICML Princeton June 18, 2025](https://openreview.net/forum?id=y1SnRPDWx4)

## Learning Path: From Foundations to Frontier

This repository offers a structured learning path:

*   **00_foundations/:** Core concepts and theory.
*   **10_guides_zero_to_one/:** Hands-on walkthroughs.
*   **20_templates/:** Copy-paste snippets.
*   **30_examples/:** Real-world projects.
*   **40_reference/:** Deep dives and evaluation.
*   **50_contrib/:** Community contributions.

## Quick Start

1.  **Understand:**  Read `00_foundations/01_atoms_prompting.md` (5 min) - why prompts alone often underperform.
2.  **Experiment:** Run `10_guides_zero_to_hero/01_min_prompt.py` (Jupyter Notebook).
3.  **Explore:** Examine `20_templates/minimal_context.yaml`.
4.  **Implement:** Study `30_examples/00_toy_chatbot/` for a complete implementation.

## Core Concepts

| Concept                     | What It Is                                    | Why It Matters                                              |
|-----------------------------|-----------------------------------------------|-------------------------------------------------------------|
| **Token Budget**              | Optimizing context length                    |  Reduce costs, improve speed                                 |
| **Few-Shot Learning**         | Learning from examples                        | Often works better than explicit explanations              |
| **Memory Systems**            | Persisting information                        |  Enables stateful and coherent interactions                |
| **Retrieval Augmentation**  | Injecting relevant documents                  |  Grounds responses in facts, reduces hallucination         |
| **Control Flow**              | Breaking tasks into steps                      |  Solve harder problems with simpler prompts                |
| **Context Pruning**           | Removing irrelevant information               |  Maintain performance, reduce unnecessary context          |
| **Metrics & Evaluation**      | Measuring context effectiveness               |  Iterative optimization of token use vs. quality            |
| **Cognitive Tools & Prompt Programming** | Building custom tools and templates      | Enable new layers for context engineering                  |
| **Neural Field Theory**     | Context as a Neural Field                     |  Model context as a dynamic neural field for iterative updating |
| **Symbolic Mechanisms**      | Symbolic architectures for higher order reasoning| Smarter systems = less work                                 |
| **Quantum Semantics**          | Meaning as observer-dependent                 | Design context systems using superpositional techniques    |

## Research: Key Insights

### 1. MEM1: Learning to Synergize Memory and Reasoning

> MEM1 trains AI agents to keep only what matters—merging memory and reasoning at every step—so they never get overwhelmed, no matter how long the task.

### 2. Cognitive Tools for Eliciting Reasoning

>  These cognitive tools (structured prompt templates as tool calls) break down the problem by identifying the main concepts at hand, extracting relevant information in the question, and highlighting meaningful properties, theorems, and techniques that might be helpful in solving the problem.

### 3. Emergent Symbolic Mechanisms in LLMs

>  These emergent mechanisms mean LLMs don’t just memorize—they actually create internal, flexible representations that let them generalize to new problems and analogies.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=davidkimai/Context-Engineering&type=Date)](https://www.star-history.com/#davidkimai/Context-Engineering&Date)

## Contributing

We welcome contributions! Check out [CONTRIBUTING.md](.github/CONTRIBUTING.md) for guidelines.

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