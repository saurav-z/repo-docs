# Context Engineering: Master the Art of Crafting LLM Context

**Unlock the full potential of Large Language Models (LLMs) by moving beyond prompt engineering and mastering the art of *context engineering*.**  This repository provides a first-principles, hands-on guide to designing, orchestrating, and optimizing the information LLMs receive, leading to more accurate, efficient, and insightful results.  ([Original Repository](https://github.com/davidkimai/Context-Engineering))

## Key Features

*   **First-Principles Approach:**  Learn the foundational concepts and build a deep understanding of context design.
*   **Hands-on Guides:**  Step-by-step tutorials and runnable code examples to get you started quickly.
*   **Progressive Learning Path:**  Follow a structured learning path, from basic prompt understanding to advanced techniques.
*   **Real-World Examples:**  Explore complete implementations of context management in various applications, including a toy chatbot.
*   **Community-Driven:**  Contribute and collaborate with the community to share knowledge and build innovative solutions.

## What is Context Engineering?

Context engineering focuses on the *entire information payload* provided to an LLM at inference time—not just the prompt. It encompasses all structured informational components that the model needs to plausibly accomplish a given task. This includes:

*   Examples
*   Memory
*   Retrieval
*   Tools
*   State
*   Control Flow

## Key Concepts You'll Master

| Concept                | What It Is                                                                 | Why It Matters                                                                |
| ---------------------- | -------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| **Token Budget**         | Optimizing the number of tokens in your context.                       | Reduces cost and speeds up response times.                                   |
| **Few-Shot Learning**    | Providing examples to guide the model.                                    | Improves accuracy and guides the models understanding of the desired output |
| **Memory Systems**       | Storing and retrieving information across interactions.                  | Enables stateful conversations and builds knowledge over time.                |
| **Retrieval Augmentation** | Injecting relevant external documents.                                     | Grounds responses in facts and reduces hallucinations.                       |
| **Control Flow**         | Structuring complex tasks into manageable steps.                         | Simplifies prompts and enables complex problem solving.                      |
| **Context Pruning**      | Removing irrelevant information.                                           | Improves performance and focuses the model's attention.                      |
| **Metrics & Evaluation** | Measuring the effectiveness of your context engineering techniques.     | Iteratively optimizes token use and quality.                                  |
| **Cognitive Tools**  | Build custom tools and templates | Prompt programming enables new layers for context engineering |
| **Neural Field Theory** | Modeling context as a dynamic neural field | Allows for iterative context updating |
| **Symbolic Mechanisms** | Symbolic architectures enable higher order reasoning | Smarter systems = less work |
| **Quantum Semantics** |  Meaning as observer-dependent  | Design context systems leveraging superpositional techniques |

## Learning Path

Follow the learning path designed to progress your understanding from basic to advanced concepts:

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

## Quick Start

1.  **Start with the Basics:** Read [`00_foundations/01_atoms_prompting.md`]
2.  **Experiment with Code:** Run [`10_guides_zero_to_hero/01_min_prompt.py`] (Jupyter Notebook style)
3.  **Use Templates:** Explore [`20_templates/minimal_context.yaml`]
4.  **Explore a Full Implementation:** Study [`30_examples/00_toy_chatbot/`]

## Inspiration & Style

This repository is inspired by the style of [3Blue1Brown](https://www.3blue1brown.com/) and Andrej Karpathy, emphasizing first principles, iterative learning, measurable results, and clear visualizations.

## Recent Research Highlights

**Leverage the latest research:**

*   **MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents:** Trains agents to keep only what matters, merging memory and reasoning at every step for efficiency and performance ([MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents - Singapore-MIT June 2025](https://www.arxiv.org/pdf/2506.15841)).
*   **Eliciting Reasoning in Language Models with Cognitive Tools:** By using modular "cognitive tools" to plan reasoning, which improves the accuracy and transparency of solutions ([Eliciting Reasoning in Language Models with Cognitive Tools - IBM Zurich June 2025](https://www.arxiv.org/pdf/2506.12115)).
*   **Emergent Symbolic Mechanisms Support Abstract Reasoning in Large Language Models:** LLMs develop their own symbolic logic circuits to generalize and perform abstract reasoning over problems ([Emergent Symbolic Mechanisms Support Abstract Reasoning in Large Language Models - ICML Princeton June 18, 2025](https://openreview.net/forum?id=y1SnRPDWx4)).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=davidkimai/Context-Engineering&type=Date)](https://www.star-history.com/#davidkimai/Context-Engineering&Date)

## Contribute

We welcome contributions!  See the [CONTRIBUTING.md](.github/CONTRIBUTING.md) guidelines.

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