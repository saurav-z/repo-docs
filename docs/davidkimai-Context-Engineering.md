# Context Engineering: Master the Art and Science of LLM Context Design

**Unlock the full potential of Large Language Models by moving beyond prompt engineering and mastering context design, orchestration, and optimization.**  Explore this comprehensive repository [here](https://github.com/davidkimai/Context-Engineering) and revolutionize how you interact with and leverage LLMs.

## Key Features

*   **First-Principles Approach:** Learn context engineering from the ground up, with a biological metaphor:  `atoms → molecules → cells → organs → neural systems → neural & semantic field theory`.
*   **Comprehensive Course:**  Progressive learning path with foundations, system implementation, integration, and frontier concepts, all with accompanying hands-on tutorials and practical examples.
*   **Research-Driven:** Operationalize the latest research papers from ICML, IBM, NeurIPS, and more.  Access key research papers and insights, including:
    *   **[Context Engineering Survey-Review of 1400 Research Papers](https://arxiv.org/pdf/2507.13334)** (July 2025)
    *   **[MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents](https://www.arxiv.org/pdf/2506.15841)** (Singapore-MIT, June 2025)
    *   **[Eliciting Reasoning in Language Models with Cognitive Tools](https://www.arxiv.org/pdf/2506.12115)** (IBM Zurich, June 2025)
    *   **[Emergent Symbolic Mechanisms Support Abstract Reasoning in Large Language Models](https://openreview.net/forum?id=y1SnRPDWx4)** (ICML Princeton, June 2025)
*   **Hands-on Guides and Templates:**  Get started quickly with a quick start guide, example notebooks, and reusable code templates.
*   **Cognitive Tools & Prompt Programming:**  Learn to build custom tools and templates.
*   **Community-Driven:**  Contribute to the project and learn from a vibrant community through discussions, contributions, and more via the [Discord server](https://discord.gg/pCM6kdxB).
*   **Extensive Documentation & Resources:**  Access deep dives, pattern libraries, evaluation checklists, and more for advanced exploration.

## What is Context Engineering?

>   "Context engineering is the delicate art and science of filling the context window with just the right information for the next step." — Andrej Karpathy

Unlike prompt engineering, which focuses on the initial input, context engineering encompasses the *entire* information payload provided to a LLM. This includes:

*   Prompts
*   Examples
*   Memory
*   Retrieval
*   Tools
*   State
*   Control flow

This allows developers to go beyond basic interactions and build complex, intelligent applications.

## Learning Path

Follow a structured learning path progressing through foundational concepts, hands-on guides, templates, and real-world examples.

1.  **Foundations:**  Understand core concepts and the biological metaphor.
2.  **Hands-on Guides:** Experiment with minimal examples and context expansion techniques.
3.  **Templates:**  Utilize reusable code snippets to jumpstart projects.
4.  **Examples:**  Explore practical implementations, building progressively more complex applications.

## What You'll Learn

| Concept                    | What It Is                                        | Why It Matters                                                                           |
| -------------------------- | ------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| **Token Budget**           | Optimizing token usage                            | Reduces costs and improves response times.                                                 |
| **Few-Shot Learning**      | Teaching through examples                         | Effective for guiding model behavior.                                                      |
| **Memory Systems**         | Persisting information across turns                | Enables stateful, coherent conversations and interactions.                                  |
| **Retrieval Augmentation** | Injecting relevant documents                      | Improves accuracy and reduces hallucinations by grounding responses in facts.               |
| **Control Flow**           | Structuring complex tasks into steps                | Enables solving more complex problems by breaking them down into manageable components.    |
| **Context Pruning**        | Removing irrelevant information                   | Maximizes efficiency and focus.                                                             |
| **Metrics & Evaluation**   | Measuring context effectiveness                    | Enables iterative optimization.                                                            |
| **Cognitive Tools**        | Building and using custom tools and templates      | Unlocks new layers of flexibility and control.                                             |
| **Neural Field Theory**    | Modeling context as a dynamic neural field         |  Enables iterative context updating.                                              |
| **Symbolic Mechanisms**    | Symbolic architectures enable higher order reasoning | Smarter systems = less work.                                                                |
| **Quantum Semantics**    |  Meaning as observer-dependent  | Design context systems leveraging superpositional techniques |

## Research Insights

The repository is aligned with the latest research, covering key concepts like:

*   **Memory + Reasoning:** Explore advanced techniques like MEM1 to efficiently manage long-horizon agent tasks.
*   **Cognitive Tools:** Learn how to leverage structured prompt templates as tool calls.
*   **Emergent Symbols:** Understand how LLMs develop internal symbolic logic circuits.

## Quick Start

1.  **Understand:**  Read  `00_foundations/01_atoms_prompting.md` (5 min).
2.  **Experiment:**  Run  `10_guides_zero_to_one/01_min_prompt.ipynb`.
3.  **Use Templates:** Explore `20_templates/minimal_context.yaml`.
4.  **Explore:** Study `30_examples/00_toy_chatbot/`.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=davidkimai/Context-Engineering&type=Date)](https://www.star-history.com/#davidkimai/Context-Engineering&Date)

## Contributing

We welcome contributions! See [CONTRIBUTING.md](.github/CONTRIBUTING.md) for details.

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

*   [Andrej Karpathy](https://x.com/karpathy/status/1937902205765607626) for coining "context engineering" and inspiring this repo
*   All contributors and the open source community