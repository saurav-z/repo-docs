# Context Engineering: Master the Art of Maximizing LLM Performance

**Unlock the true potential of Large Language Models by mastering context engineering, the next frontier beyond prompt engineering.** This repository provides a comprehensive, first-principles guide to designing, orchestrating, and optimizing the information LLMs use at inference time. ([Original Repository](https://github.com/davidkimai/Context-Engineering))

<div align="center">
    <img width="1600" height="400" alt="image" src="https://github.com/user-attachments/assets/f41f9664-b707-4291-98c8-5bab3054a572" />
</div>

> "Context engineering is the delicate art and science of filling the context window with just the right information for the next step." — [Andrej Karpathy](https://x.com/karpathy/status/1937902205765607626)

## Key Features:

*   **First-Principles Approach:**  A progressive learning path built on fundamental concepts, moving from basic prompting to advanced context design.
*   **Visual Learning:**  Concepts are explained with clear visuals, including ASCII diagrams, symbolic representations, and real-world parallels.
*   **Code-Driven:** Practical examples and runnable code cells are provided to solidify understanding.
*   **Progressive Learning Path:** A structured approach, from foundational principles to advanced techniques like memory systems, retrieval augmentation, and cognitive tools.
*   **Research-Backed:**  Incorporates the latest research on context engineering, including papers from ICML, NeurIPS, and more.
*   **Practical Tools & Templates:** Includes helpful templates, code, and practical advice you can put to work immediately.

## Why Context Engineering?

Prompt engineering is just the beginning. The real power lies in optimizing the *entire context window*: the examples, memory, retrieval, tools, state, and control flow that the model sees.  This is where you can truly customize performance.

```
                    Prompt Engineering  │  Context Engineering
                       ↓                │            ↓                      
               "What you say"           │  "Everything else the model sees"
             (Single instruction)       │    (Examples, memory, retrieval,
                                        │     tools, state, control flow)
```

## Core Concepts:

*   **Token Budget Optimization:**  Efficiently manage the context window's token count for better performance and cost savings.
*   **Few-Shot Learning Mastery:**  Leverage the power of examples to guide the model's behavior.
*   **Memory Systems Integration:**  Implement systems for persisting information across interactions, creating stateful applications.
*   **Retrieval-Augmented Generation (RAG):**  Enrich responses with relevant external data.
*   **Context Pruning Strategies:**  Remove irrelevant data to focus the model's attention.
*   **Evaluation and Measurement:**  Understand metrics for assessing and improving context engineering strategies.
*   **Cognitive Tool & Prompt Programming:** Learn to build custom tools, and prompt programs to get more from your context engineering.

## Learning Path:

The repository offers a structured learning path:

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

## Quick Start:

1.  **Start with the Fundamentals:**  Review the basics at  [`00_foundations/01_atoms_prompting.md`](00_foundations/01_atoms_prompting.md) (5 min) to understand the limitations of prompt engineering alone.
2.  **Run a Minimal Example:** Experiment with a basic working example found at  [`10_guides_zero_to_hero/01_min_prompt.py`](10_guides_zero_to_hero/01_min_prompt.py) (Jupyter Notebook style).
3.  **Explore Templates:** Use provided templates by studying  [`20_templates/minimal_context.yaml`](20_templates/minimal_context.yaml).
4.  **Review Real Projects:** See complete implementations with context management by exploring the examples at  [`30_examples/00_toy_chatbot/`](30_examples/00_toy_chatbot/).

## Research Evidence:

This project draws from the latest research:

*   **MEM1 (Learning to Synergize Memory and Reasoning)**
*   **Cognitive Tools (Eliciting Reasoning in Language Models)**
*   **Emergent Symbolic Mechanisms**

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

Special thanks to [Andrej Karpathy](https://x.com/karpathy/status/1937902205765607626) for coining the term "context engineering" and inspiring this repository.