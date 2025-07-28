<div align="center">
  
# Context Engineering: Beyond Prompt Engineering

</div>

<!-- Add a captivating image or GIF here related to context engineering -->
<img width="1600" height="400" alt="image" src="https://github.com/user-attachments/assets/f41f9664-b707-4291-98c8-5bab3054a572" />

**Unlock the full potential of Large Language Models (LLMs) by mastering Context Engineering, the art and science of optimizing the information provided to LLMs for superior results.  Learn how to orchestrate memory, retrieval, and tools to move beyond prompt engineering and design effective LLM interactions!**

> *“Context engineering is the delicate art and science of filling the context window with just the right information for the next step.” — [Andrej Karpathy](https://x.com/karpathy/status/1937902205765607626)*

**Explore the original repository: [Context-Engineering](https://github.com/davidkimai/Context-Engineering)**

## Key Features

*   **Comprehensive Course:** A step-by-step guide to mastering context design, orchestration, and optimization.
*   **Hands-on Tutorials:** Practical examples and Jupyter Notebooks to get you started quickly.
*   **First-Principles Approach:** Understand the underlying principles and build a strong foundation.
*   **Code & Visuals:** Learn through runnable code examples and clear visual representations.
*   **Research-Backed:** Leverage the latest research in context engineering.
*   **Community Driven:** Join the discussion and contribute to the open-source project via [Discord](https://discord.gg/pCM6kdxB) and [Ask DeepWiki](https://deepwiki.com/davidkimai/Context-Engineering).

## What is Context Engineering?

Context Engineering goes beyond prompt engineering by focusing on the **entire information payload** provided to an LLM at inference time.  This encompasses:

*   Examples, memory, and retrieval systems.
*   Tools, state, and control flow mechanisms.

This repository provides a progressive, first-principles approach to context engineering, built around a biological metaphor:

```
atoms → molecules → cells → organs → neural systems → neural & semantic field theory 
  │        │         │         │             │                         │        
single    few-     memory +   multi-   cognitive tools +     context = fields +
prompt    shot     agents     agents   operating systems     persistence & resonance
```

```
                    Prompt Engineering  │  Context Engineering
                       ↓                │            ↓                      
               "What you say"           │  "Everything else the model sees"
             (Single instruction)       │    (Examples, memory, retrieval,
                                        │     tools, state, control flow)
```

## Learning Path

A structured, progressive approach to learning context engineering:

*   **00\_Foundations:** Theory & Core Concepts
*   **10\_Guides\_zero\_to\_one:** Hands-on Walkthroughs
*   **20\_Templates:** Copy-paste Snippets
*   **30\_Examples:** Real Projects, Progressively Complex
*   **40\_Reference:** Deep Dives & Evaluation Cookbook
*   **50\_Contrib:** Community Contributions

## Core Concepts to Master

*   **Token Budget:** Optimize every token in your context for cost efficiency and speed.
*   **Few-Shot Learning:** Teach models by providing examples instead of relying solely on instructions.
*   **Memory Systems:** Implement persistent information storage for stateful interactions.
*   **Retrieval Augmentation (RAG):** Enhance responses by incorporating relevant external knowledge.
*   **Control Flow:** Structure complex tasks into manageable steps for improved performance.
*   **Context Pruning:** Eliminate irrelevant information to improve efficiency and accuracy.
*   **Metrics & Evaluation:** Measure and iterate on your context engineering strategies.
*   **Cognitive Tools & Prompt Programming:** Develop custom tools and templates to unlock new capabilities.
*   **Neural Field Theory:** Model context as a dynamic neural field for iterative updates.
*   **Symbolic Mechanisms:** Leverage symbolic architectures for higher-order reasoning capabilities.
*   **Quantum Semantics:** Experiment with superpositional techniques to design innovative context systems.

## Research Insights

This project is heavily influenced by recent research advancements. Here are a few key findings:

### [MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents - Singapore-MIT June 2025](https://www.arxiv.org/pdf/2506.15841)

> “Our results demonstrate the promise of reasoning-driven memory consolidation as a scalable alternative to existing solutions for training long-horizon interactive agents, where both efficiency and performance are optimized." — [Singapore-MIT](https://arxiv.org/pdf/2506.15841)

*   Trains AI agents to merge memory and reasoning, compressing each interaction into a compact internal state, ensuring agents handle long tasks with efficiency and performance.

### [Eliciting Reasoning in Language Models with Cognitive Tools - IBM Zurich June 2025](https://www.arxiv.org/pdf/2506.12115)

### Prompts and Prompt Programs as Reasoning Tool Calls
> “Cognitive tools” encapsulate reasoning operations within the LLM itself — [IBM Zurich](https://www.arxiv.org/pdf/2506.12115)

*   Breaks down complex tasks into modular "cognitive tools" (structured prompt templates), mirroring how expert humans reason step-by-step and solve math problems.

### [Emergent Symbolic Mechanisms Support Abstract Reasoning in Large Language Models - ICML Princeton June 18, 2025](https://openreview.net/forum?id=y1SnRPDWx4)

> **TL;DR: A three-stage architecture is identified that supports abstract reasoning in LLMs via a set of emergent symbol-processing mechanisms.**

*   LLMs develop their own inner symbolic "logic circuits", creating internal representations that let them generalize to new problems and analogies.

## Quick Start

1.  **Read `00_foundations/01_atoms_prompting.md`** (5 min)
2.  **Run `10_guides_zero_to_one/01_min_prompt.py (Jupyter Notebook style)`**
3.  **Explore `20_templates/minimal_context.yaml`**
4.  **Study `30_examples/00_toy_chatbot/`**

## Contributing

We welcome contributions! See [CONTRIBUTING.md](.github/CONTRIBUTING.md) for guidelines.

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

*   [Andrej Karpathy](https://x.com/karpathy/status/1937902205765607626) for coining "context engineering" and inspiring this repo
*   All contributors and the open-source community