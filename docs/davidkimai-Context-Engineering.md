# Context Engineering: Beyond Prompt Engineering

**Unleash the power of LLMs by mastering the art and science of Context Engineering, moving beyond prompts to design, orchestrate, and optimize the complete information payload for superior performance.**

[View the original repository on GitHub](https://github.com/davidkimai/Context-Engineering)

## Key Features

*   **Comprehensive Course:** A first-principles handbook for moving beyond prompt engineering to context design, orchestration, and optimization.
*   **Practical Guides:** Step-by-step tutorials and hands-on examples to accelerate your learning.
*   **Advanced Techniques:** Explore Memory Systems, Retrieval Augmentation, Cognitive Tools, and more.
*   **Research-Driven:** Grounded in cutting-edge research, including MEM1, IBM Zurich, and ICML Princeton.
*   **Modular Architecture:**  A biological metaphor: atoms → molecules → cells → organs → neural systems → neural & semantic field theory
*   **Community-Driven:**  Contribute and collaborate with the open-source community.

## What is Context Engineering?

Context Engineering is the art and science of crafting the complete information payload provided to a Large Language Model (LLM) at inference time. It goes far beyond prompt engineering.

**Prompt Engineering vs. Context Engineering**

```
                    Prompt Engineering  │  Context Engineering
                       ↓                │            ↓                      
               "What you say"           │  "Everything else the model sees"
             (Single instruction)       │    (Examples, memory, retrieval,
                                        │     tools, state, control flow)
```

This includes examples, memory, retrieval, tools, state, and control flow.  It's about providing the LLM with everything it needs to successfully complete a task.

## Core Concepts

*   **Token Budgeting:** Optimize token usage for efficiency and cost savings.
*   **Few-Shot Learning:** Leverage examples to guide LLM behavior.
*   **Memory Systems:** Enable stateful interactions and coherence.
*   **Retrieval Augmentation:** Ground responses in facts and reduce hallucinations.
*   **Control Flow:** Orchestrate complex tasks with multi-step processes.
*   **Context Pruning:** Eliminate irrelevant information for optimal performance.
*   **Metrics & Evaluation:** Measure the effectiveness of your context engineering strategies.
*   **Cognitive Tools & Prompt Programming:** Build custom tools and templates.
*   **Neural Field Theory:** Model context as a dynamic neural field for iterative updates.
*   **Symbolic Mechanisms:** Integrate symbolic architectures for advanced reasoning.
*   **Quantum Semantics:** Design context systems leveraging superpositional techniques.

## Learning Path

The repository is organized to guide you from the fundamentals to advanced techniques.

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

## Example Implementations

Explore practical applications of context engineering:

*   Toy Chatbot
*   Data Annotation
*   Multi-Agent Orchestration
*   VSCode Helper
*   RAG Implementation

## Stay Connected

*   **Discord:** Join the chat!  [Discord](https://discord.gg/pCM6kdxB)
*   **DeepWiki:** Explore additional content.  [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/davidkimai/Context-Engineering)

## Supporting Research

This project is built on a foundation of groundbreaking research:

*   **MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents - Singapore-MIT June 2025**  (https://www.arxiv.org/pdf/2506.15841)
*   **Eliciting Reasoning in Language Models with Cognitive Tools - IBM Zurich June 2025** (https://www.arxiv.org/pdf/2506.12115)
*   **Emergent Symbolic Mechanisms Support Abstract Reasoning in Large Language Models - ICML Princeton June 18, 2025** (https://openreview.net/forum?id=y1SnRPDWx4)

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](.github/CONTRIBUTING.md) for guidelines.

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

*   [Andrej Karpathy](https://x.com/karpathy/status/1937902205765607626) for coining "context engineering."
*   All contributors and the open source community.