# Context Engineering: Master the Art of Guiding LLMs Beyond Prompts

Unlock the full potential of Large Language Models (LLMs) by mastering **Context Engineering**, the cutting-edge discipline of designing, orchestrating, and optimizing the entire context window. Explore the [original repository](https://github.com/davidkimai/Context-Engineering) for a deep dive into the latest research and practical applications.

## Key Features

*   **Comprehensive Handbook:** A first-principles guide to context design, going beyond prompt engineering.
*   **Modular Learning Path:** Structured approach, from foundational concepts to advanced techniques.
*   **Practical Examples:** Hands-on tutorials, reusable templates, and real-world implementations.
*   **Research-Driven:** Operationalizing cutting-edge research from ICML, NeurIPS, and more.
*   **Community-Driven:**  Contribute and collaborate with the open-source community.

## What is Context Engineering?

Context Engineering encompasses *everything* the LLM sees at inference time: not just the prompt, but also examples, memory, retrieval tools, state, and control flow. This repository provides a progressive approach, built on a biological metaphor:

```
atoms → molecules → cells → organs → neural systems → neural & semantic field theory 
  │        │         │         │             │                         │        
single    few-     memory +   multi-   cognitive tools +     context = fields +
prompt    shot     agents     agents   operating systems     persistence & resonance
```

## Core Concepts & Benefits

| Concept                | What It Is                                              | Why It Matters                                                           |
| ---------------------- | ------------------------------------------------------- | ------------------------------------------------------------------------ |
| **Token Budget**       | Optimizing every token in your context                  | Cost-effective LLM interactions, faster responses.                        |
| **Few-Shot Learning**    | Teaching by showing examples                             | Often more effective than extensive explanations.                         |
| **Memory Systems**       | Persisting information across turns                     | Enables stateful, coherent, and context-aware interactions.              |
| **Retrieval Augmentation** | Injecting relevant documents                            | Grounds responses in facts, reduces hallucinations, improves accuracy.    |
| **Control Flow**         | Breaking complex tasks into steps                        | Enables solving of difficult problems through task decomposition.        |
| **Context Pruning**      | Removing irrelevant information                         | Improves performance by focusing on essential data.                       |
| **Metrics & Evaluation** | Measuring context effectiveness                         | Iterative optimization: maximizing quality per token.                    |
| **Cognitive Tools & Prompt Programming**| Build custom tools and templates | Enables new layers for context engineering|
| **Neural Field Theory**  | Context as a Neural Field                             | Iterative context updating using Dynamic Neural Fields |
| **Symbolic Mechanisms**  | Symbolic architectures for higher-order reasoning        | Enable reasoning with abstract variables, reduce need for data         |
| **Quantum Semantics**  | Meaning as observer-dependent  | Design context systems leveraging superpositional techniques |

## Recent Research Highlights

*   **MEM1 (Singapore-MIT):**  Efficient long-horizon agents, optimizing memory and reasoning.  [Read more...](https://www.arxiv.org/pdf/2506.15841)
*   **Cognitive Tools (IBM Zurich):**  Eliciting reasoning in LLMs with structured templates (prompt programs). [Read more...](https://www.arxiv.org/pdf/2506.12115)
*   **Emergent Symbols (ICML Princeton):**  LLMs develop inner symbolic "logic circuits" enabling reasoning. [Read more...](https://openreview.net/forum?id=y1SnRPDWx4)

## Structure & Resources

*   **00\_foundations/**: Foundational theory and core concepts.
*   **10\_guides\_zero\_to\_hero/**: Hands-on tutorials and walkthroughs.
*   **20\_templates/**: Reusable code snippets and templates.
*   **30\_examples/**: Real-world project implementations.
*   **40\_reference/**: Deep dives and evaluation resources.
*   **50\_contrib/**: Community contributions.
*   **60\_protocols/**: Protocol shells and frameworks.
*   **70\_agents/**: Agent demonstrations.
*   **80\_field\_integration/**: Complete field projects.
*   **cognitive-tools/:** Advanced cognitive framework

## Quick Start

1.  **Start with:** `00_foundations/01_atoms_prompting.md` (5 min read).
2.  **Experiment:** Run `10_guides_zero_to_one/01_min_prompt.ipynb`.
3.  **Explore:** Use `20_templates/minimal_context.yaml`.
4.  **Implement:** Review `30_examples/00_toy_chatbot/`.

## Contributing

We welcome contributions! Review the [CONTRIBUTING.md](.github/CONTRIBUTING.md) guidelines.

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