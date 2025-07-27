# Context Engineering: Beyond Prompt Engineering

**Unlock the full potential of LLMs by mastering context design, orchestration, and optimization: This repository provides a first-principles, biological-inspired approach to context engineering, moving beyond prompt engineering to create truly intelligent systems. Explore the cutting-edge research, practical applications, and hands-on guides for building advanced LLM applications. ([Original Repo](https://github.com/davidkimai/Context-Engineering))**

> *"Context engineering is the delicate art and science of filling the context window with just the right information for the next step."* – [Andrej Karpathy](https://x.com/karpathy/status/1937902205765607626)

## Key Features

*   **Comprehensive Course:** A structured learning path covering foundational concepts to advanced techniques.
*   **Practical Guides:** Step-by-step tutorials and hands-on examples to get you started quickly.
*   **Real-World Applications:** Explore practical implementations and build your own LLM-powered applications.
*   **Research-Driven:** Leverage the latest research in context engineering, including MEM1, IBM Zurich, and more.
*   **Modular Approach:** Learn to build cognitive tools and reasoning engines.
*   **Community-Driven:** Contribute, collaborate, and share your knowledge with the community.

## Why Context Engineering?

While prompt engineering focuses on "what you say," context engineering focuses on "everything else the model sees" – examples, memory, retrieval, tools, state, and control flow. This repository provides a first-principles, biological-inspired approach to context design, orchestration, and optimization.

```
                    Prompt Engineering  │  Context Engineering
                       ↓                │            ↓                      
               "What you say"           │  "Everything else the model sees"
             (Single instruction)       │    (Examples, memory, retrieval,
                                        │     tools, state, control flow)
```

## Core Concepts

*   **Token Budgeting:** Optimize context for cost-effectiveness and performance.
*   **Few-Shot Learning:** Leverage examples to guide model behavior.
*   **Memory Systems:** Implement persistent and stateful interactions.
*   **Retrieval Augmentation:** Enhance accuracy and reduce hallucination.
*   **Control Flow:** Orchestrate complex tasks with step-by-step processes.
*   **Context Pruning:** Maintain focus by removing unnecessary information.
*   **Evaluation & Metrics:** Measure and refine the effectiveness of your context.
*   **Cognitive Tools & Prompt Programming:** Build custom tools and architectures
*   **Neural Field Theory:** Model context as a dynamic neural field for iterative updating.
*   **Symbolic Mechanisms:** Higher-order reasoning with symbolic architectures.
*   **Quantum Semantics:** Meaning as observer-dependent for more advanced systems.

## Learning Path

This repository offers a structured learning path, from foundational theory to practical application.

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

## Research Highlights

Stay ahead with key research findings:

### MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents
> "Our results demonstrate the promise of reasoning-driven memory consolidation as a scalable alternative to existing solutions for training long-horizon interactive agents, where both efficiency and performance are optimized." - Singapore-MIT

### Eliciting Reasoning in Language Models with Cognitive Tools
> “Cognitive tools” encapsulate reasoning operations within the LLM itself - IBM Zurich

### Emergent Symbolic Mechanisms Support Abstract Reasoning in Large Language Models
> TL;DR: A three-stage architecture is identified that supports abstract reasoning in LLMs via a set of emergent symbol-processing mechanisms. - ICML Princeton

## Get Started

1.  **Explore:** Dive into the foundations: `00_foundations/01_atoms_prompting.md` (5 min)
2.  **Experiment:** Run a minimal example: `10_guides_zero_to_one/01_min_prompt.py`
3.  **Use Templates:** Copy templates: `20_templates/minimal_context.yaml`
4.  **Implement:** Study real-world examples: `30_examples/00_toy_chatbot/`

## Contributing

Contributions are welcome! Please review our [CONTRIBUTING.md](.github/CONTRIBUTING.md) guidelines.

## License

This project is licensed under the [MIT License](LICENSE).

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

*   [Andrej Karpathy](https://x.com/karpathy/status/1937902205765607626) for inspiring the term "context engineering."
*   All contributors and the open-source community.