# Context Engineering: Master the Art of Information Orchestration for LLMs

**Unlock the full potential of Large Language Models (LLMs) by mastering context engineering—the art and science of shaping the *entire* context window.** Explore this [original repo](https://github.com/davidkimai/Context-Engineering) to move beyond basic prompt engineering and unlock a new frontier of LLM capabilities.

## Key Features

*   **Comprehensive, First-Principles Approach:** Go beyond prompt engineering and design, orchestrate, and optimize your LLM interactions for superior results.
*   **Practical, Hands-On Learning:**  Learn through a biological metaphor—atoms to molecules to neural systems—with runnable code examples and visual aids.
*   **Cutting-Edge Research & Applications:** Explore the latest advancements in LLM context design with insights from leading research papers and practical applications.
*   **Progressive Learning Path:** Start with fundamentals and advance through key concepts like memory systems, retrieval augmentation, control flow, and evaluation.
*   **Community-Driven & Open Source:** Join the community, contribute, and help shape the future of context engineering.

## What is Context Engineering?

> "Context engineering is the delicate art and science of filling the context window with just the right information for the next step." — [Andrej Karpathy](https://x.com/karpathy/status/1937902205765607626)

Context Engineering focuses on optimizing all the information provided to an LLM at inference time. It encompasses much more than the prompt itself. Rather, it includes examples, memory, retrieval, tools, state, and control flow.

**The Difference:**

```
                    Prompt Engineering  │  Context Engineering
                       ↓                │            ↓                      
               "What you say"           │  "Everything else the model sees"
             (Single instruction)       │    (Examples, memory, retrieval,
                                        │     tools, state, control flow)
```

## Core Concepts

*   **Token Budget:** Optimize your context for efficiency and cost-effectiveness.
*   **Few-Shot Learning:** Teach models with examples for improved performance.
*   **Memory Systems:** Enable stateful and coherent interactions.
*   **Retrieval Augmentation:** Ground responses in facts to reduce hallucinations.
*   **Control Flow:** Build complex task pipelines with simple prompts.
*   **Context Pruning:** Refine information to enhance performance.
*   **Evaluation & Metrics:**  Measure and refine LLM performance.
*   **Cognitive Tools:** Learn how to build custom tools and templates.
*   **Neural Field Theory:** Learn iterative context updating using a dynamic neural field.
*   **Symbolic Mechanisms:** Leverage symbolic architectures for reasoning.
*   **Quantum Semantics:** Explore the power of superpositional techniques in LLM design.

## Dive Into Key Research

This repository provides insights and implementations based on the latest research in the field, including:

*   **MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents - Singapore-MIT June 2025**
*   **Eliciting Reasoning in Language Models with Cognitive Tools - IBM Zurich June 2025**
*   **Emergent Symbolic Mechanisms Support Abstract Reasoning in Large Language Models - ICML Princeton June 18, 2025**

##  Get Started

1.  **Explore Foundations:** Read the [`00_foundations/01_atoms_prompting.md`] document to start your learning journey.
2.  **Run a Simple Example:** Execute the [`10_guides_zero_to_hero/01_min_prompt.py`] notebook.
3.  **Experiment with Templates:** Customize a template by using the [`20_templates/minimal_context.yaml`] file.
4.  **Study a Complete Example:** See a comprehensive implementation by exploring the [`30_examples/00_toy_chatbot/`] directory.

##  Learning Path

*   **Foundations:** Understand the core concepts and theory.
*   **Guides:** Hands-on walkthroughs and practical application.
*   **Templates:** Copy-paste snippets for quick implementations.
*   **Examples:** Real projects and practical applications.
*   **Reference:** Deep dives and evaluation cookbooks.
*   **Contributions:** Join the community and share your progress.

##  Contribute & Connect

We welcome contributions! See our [CONTRIBUTING.md](.github/CONTRIBUTING.md) file for instructions. Join our [Discord](https://discord.gg/JeFENHNNNQ) for discussions and support.

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

## License

[MIT License](LICENSE)

## Acknowledgements
> I've been looking forward to this being conceptualized and formalized as there wasn't a prior established field. Prompt engineering receives quite the stigma and doesn't quite cover what most researchers and I do.

- [Andrej Karpathy](https://x.com/karpathy/status/1937902205765607626) for coining "context engineering" and inspiring this repo 
- All contributors and the open source community