# Context Engineering: Beyond Prompt Engineering - Mastering the Art and Science of LLM Context ([Original Repo](https://github.com/davidkimai/Context-Engineering))

Unleash the full potential of Large Language Models by mastering context engineering: moving beyond prompts to design, orchestrate, and optimize the entire information payload.

**Context engineering** is a rapidly evolving field focused on optimizing the information provided to a Large Language Model (LLM) at inference time to achieve superior performance. This repository provides a comprehensive, first-principles approach, helping you build more effective and efficient AI systems.

**Key Features:**

*   **Comprehensive Course:** A structured learning path covering foundational concepts, system implementation, integration strategies, and cutting-edge research. (See [Comprehensive Course Under Construction](https://github.com/davidkimai/Context-Engineering/tree/main/00_COURSE))
*   **Hands-on Guides & Examples:** Practical, runnable code examples and walkthroughs to accelerate your learning (e.g. [`00_foundations/01_atoms_prompting.md`](00_foundations/01_atoms_prompting.md) and [`10_guides_zero_to_hero/01_min_prompt.py`](10_guides_zero_to_hero/01_min_prompt.py)).
*   **Cutting-Edge Research:** Explore the latest advancements in the field, including research papers on memory, reasoning, cognitive tools, and emergent symbolic mechanisms ([Context Engineering Survey-Review of 1400 Research Papers](https://arxiv.org/pdf/2507.13334)).
*   **Visualizations & Diagrams:** Concepts are explained with clear visuals and diagrams to improve understanding, inspired by the style of 3Blue1Brown and Andrej Karpathy.
*   **Community Driven:** A collaborative environment that encourages contributions and knowledge sharing (See [CONTRIBUTING.md](.github/CONTRIBUTING.md)).
*   **Agent Commands** Integration of various LLMs ([`Agent Commands`](https://github.com/davidkimai/Context-Engineering/tree/main/.claude/commands) : Support for [Claude Code](https://www.anthropic.com/claude-code) | [OpenCode](https://opencode.ai/) | [Amp](https://sourcegraph.com/amp) | [Kiro](https://kiro.dev/) | [Codex](https://openai.com/codex/) | [Gemini CLI](https://github.com/google-gemini/gemini-cli))

## What is Context Engineering?

Context Engineering moves beyond simple prompt design, focusing on the entire information payload an LLM receives. The information that surrounds prompts is more often than not overlooked. Context is the complete information provided to an LLM at inference time to enable it to accomplish a given task.

```
                    Prompt Engineering  │  Context Engineering
                       ↓                │            ↓                      
               "What you say"           │  "Everything else the model sees"
             (Single instruction)       │    (Examples, memory, retrieval,
                                        │     tools, state, control flow)
```

## Learning Path

This repository provides a structured learning path to guide you from the fundamentals to advanced concepts:

1.  **Foundations:** Theory and core concepts. (00_foundations/)
2.  **Hands-on Guides:** Hands-on walkthroughs and tutorials. (10_guides_zero_to_one/)
3.  **Templates:** Copy-paste code snippets. (20_templates/)
4.  **Examples:** Real-world projects. (30_examples/)
5.  **Reference:** Deep dives and evaluation. (40\_reference/)
6.  **Community Contributions:**  Community contributions (50\_contrib/)

## Key Concepts You'll Master

| Concept                   | What It Is                                            | Why It Matters                                                                              |
| ------------------------- | ----------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| **Token Budget**          | Optimizing token usage                                 | Cost savings and faster responses.                                                          |
| **Few-Shot Learning**     | Teaching through examples                             | Improves performance and reduces the need for extensive explanations.                        |
| **Memory Systems**        | Persisting information across turns                    | Enables stateful, coherent interactions.                                                     |
| **Retrieval Augmentation** | Finding & injecting relevant documents                 | Grounds responses in facts, reduces hallucinations.                                         |
| **Control Flow**          | Breaking complex tasks into steps                      | Enables solving harder problems with simpler prompts.                                        |
| **Context Pruning**       | Removing irrelevant information                        | Improves performance by keeping only essential information.                                  |
| **Metrics & Evaluation**  | Measuring context effectiveness                        | Iterative optimization of token use vs. quality.                                            |
| **Cognitive Tools**        | Learning to build custom tools and templates.              | Facilitates more organized and advanced LLM interaction. |
| **Neural Field Theory**        | Designing context systems leveraging superpositional techniques.              | Allows for iterative context updating. |
| **Quantum Semantics**        | Designing context systems with observer-dependent meaning.              | Makes LLMs more flexible and reliable. |

## Research Highlights
This repository highlights the most recent research papers and discoveries for better understanding.

## Contributing

We welcome contributions! See [`CONTRIBUTING.md`](.github/CONTRIBUTING.md) for guidelines.

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
- [Andrej Karpathy](https://x.com/karpathy/status/1937902205765607626) for coining "context engineering" and inspiring this repo 
- All contributors and the open source community