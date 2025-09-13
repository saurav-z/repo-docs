# Context Engineering: Unlock the Power of LLMs by Mastering the Context Window

**Context Engineering is the practice of orchestrating the complete information payload delivered to a Large Language Model (LLM) at inference time to achieve optimal results.** This repository provides a first-principles approach to context design, optimization, and advanced techniques. Explore practical examples and insights to elevate your work beyond prompt engineering and unlock the true potential of LLMs.

[![Star History Chart](https://api.star-history.com/svg?repos=davidkimai/Context-Engineering&type=Date)](https://www.star-history.com/#davidkimai/Context-Engineering&Date)

[View the original repository on GitHub](https://github.com/davidkimai/Context-Engineering)

## Key Features

*   **Comprehensive Learning Path:** A structured approach covering foundations, system implementation, integration, and advanced concepts.
*   **First-Principles Approach:** Learn from the ground up, focusing on fundamental principles and biological metaphors for understanding.
*   **Practical Examples:**  Hands-on guides, code samples, and real-world project implementations to solidify your understanding.
*   **Visualizations:** Benefit from diagrams and visual aids to simplify complex concepts and promote intuitive learning.
*   **Community Driven:** Join the community and contribute to the continuous development of context engineering techniques.
*   **Deep Dive into Research:** Comprehensive survey and summaries of cutting-edge research papers in context engineering, covering the latest advancements and findings.

## Core Concepts Covered

*   **Token Budget Optimization:** Maximize efficiency and reduce costs by carefully managing token usage.
*   **Few-Shot Learning:** Leverage the power of examples to guide and instruct LLMs effectively.
*   **Memory Systems:** Implement stateful interactions and maintain context across multiple turns.
*   **Retrieval Augmentation:** Enhance accuracy by incorporating relevant external knowledge.
*   **Control Flow:** Break down complex tasks into manageable steps using modular design.
*   **Context Pruning:** Improve performance by removing irrelevant information from the context window.
*   **Metrics & Evaluation:**  Learn to measure and optimize context engineering strategies.
*   **Cognitive Tools & Prompt Programming:** Build custom tools and templates.
*   **Neural Field Theory:** Model context as a dynamic neural field.
*   **Symbolic Mechanisms:** Explore architectures enabling higher-order reasoning.
*   **Quantum Semantics:** Design context systems leveraging superpositional techniques

## Getting Started Quickly

1.  **Explore the Foundations:** Begin by reading `00_foundations/01_atoms_prompting.md` (5 minutes).
2.  **Run a Minimal Example:** Experiment with a working example using `10_guides_zero_to_hero/01_min_prompt.py`.
3.  **Explore Templates:** Customize by copying and pasting from `20_templates/minimal_context.yaml`.
4.  **Dive into an Implementation:** Review a full implementation in `30_examples/00_toy_chatbot/`.

## Learning Path Structure

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

## Research Highlights & Key Insights

This repository synthesizes and operationalizes cutting-edge research findings in context engineering, drawing insights from leading institutions such as ICML, IBM, NeurIPS, and more.

### Memory + Reasoning: MEM1

*   MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents
*   By merging memory and reasoning, agents become more efficient, and can handle longer tasks.
*   Instead of piling up endless context, MEM1 compresses each interaction into a compact “internal state.”

### Cognitive Tools

*   Eliciting Reasoning in Language Models with Cognitive Tools (IBM Zurich)
*   "Cognitive tools" break down problems using modular prompt templates.
*   Instead of relying on a single, big prompt, the model calls specialized prompt templates.

### Emergent Symbols

*   Emergent Symbolic Mechanisms Support Abstract Reasoning in Large Language Models (ICML Princeton)
*   LLMs develop inner symbolic logic circuits.
*   Large language models create internal, flexible representations that let them generalize to new problems and analogies.

## Contributing

We welcome contributions!  Review [`CONTRIBUTING.md`](.github/CONTRIBUTING.md) for details.

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