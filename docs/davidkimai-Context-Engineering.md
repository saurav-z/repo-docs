# Context Engineering: Go Beyond Prompts and Build Powerful LLM Applications

**Master the art of building effective Large Language Model (LLM) applications by mastering context engineering, a new discipline that focuses on optimizing the information provided to LLMs to enhance performance.** [Explore the original repository](https://github.com/davidkimai/Context-Engineering) for a comprehensive guide.

**Key Features:**

*   **Comprehensive Handbook:** A first-principles guide to context design, orchestration, and optimization.
*   **Practical Learning Path:** Step-by-step approach from foundational concepts to advanced techniques.
*   **Research-Driven:** Operationalizes the latest research in context engineering.
*   **Visualizations & Code Examples:** Clear explanations with code and diagrams to aid understanding.
*   **Community-Driven:** Open to contributions and collaboration.

## What is Context Engineering?

Context engineering is the practice of curating the *complete* information payload delivered to a Large Language Model (LLM) during inference to improve outcomes. Instead of relying solely on prompts, this involves providing the model with all the necessary data, examples, memory, tools, and structure it needs to succeed.

> “Context is not just the single prompt users send to an LLM. Context is the complete information payload provided to a LLM at inference time, encompassing all structured informational components that the model needs to plausibly accomplish a given task.”
> – [A Systematic Analysis of Over 1400 Research Papers](https://arxiv.org/pdf/2507.13334)

## Core Concepts

*   **Token Budget:** Optimizing token usage for cost and speed.
*   **Few-Shot Learning:** Providing examples to guide model behavior.
*   **Memory Systems:** Enabling stateful interactions.
*   **Retrieval Augmentation (RAG):** Grounding responses with relevant documents.
*   **Control Flow:** Structuring complex tasks with steps.
*   **Context Pruning:** Removing irrelevant information.
*   **Metrics & Evaluation:** Measuring the impact of your context design.
*   **Cognitive Tools & Prompt Programming:** Custom tool and template creation.
*   **Neural Field Theory:** Treating context as a dynamic neural field.
*   **Symbolic Mechanisms:** Harnessing symbolic architectures for better reasoning.
*   **Quantum Semantics:** Leveraging superpositional techniques for context.

## Learning Path

A progressive approach designed to take you from prompt engineering to advanced context design.

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

1.  **Fundamentals:** [00\_foundations/01\_atoms\_prompting.md](00\_foundations/01\_atoms\_prompting.md) (5 min)
2.  **Experiment:** Run a simple example: [`10_guides_zero_to_hero/01_min_prompt.py`](10_guides_zero_to_hero/01_min_prompt.py) (Jupyter Notebook)
3.  **Templates:** Explore: [`20_templates/minimal_context.yaml`](20_templates/minimal_context.yaml)
4.  **Implementation:** See a complete example: [`30_examples/00_toy_chatbot/`](30_examples/00_toy_chatbot/)

## Why Context Engineering Matters

While prompt engineering is a valuable skill, the future lies in optimizing the *entire* context window. Context engineering gives you the tools to design smarter applications.

## Research & Resources

This repository draws inspiration from the latest research:

*   [**Context Engineering Survey-Review of 1400 Research Papers**](https://arxiv.org/pdf/2507.13334)
*   [**MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents - Singapore-MIT June 2025**](https://www.arxiv.org/pdf/2506.15841)
*   [**Eliciting Reasoning in Language Models with Cognitive Tools - IBM Zurich June 2025**](https://www.arxiv.org/pdf/2506.12115)
*   [**Emergent Symbolic Mechanisms Support Abstract Reasoning in Large Language Models - ICML Princeton June 18, 2025**](https://openreview.net/forum?id=y1SnRPDWx4)

**Additional Resources:**

*   [Ask DeepWiki](https://deepwiki.com/davidkimai/Context-Engineering)
*   [DeepGraph](https://www.deepgraph.co/davidkimai/Context-Engineering)
*   [Chat with NotebookLM + Podcast Deep Dive](https://notebooklm.google.com/notebook/0c6e4dc6-9c30-4f53-8e1a-05cc9ff3bc7e)
*   [Discord Community](https://discord.gg/JeFENHNNNQ)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=davidkimai/Context-Engineering&type=Date)](https://www.star-history.com/#davidkimai/Context-Engineering&Date)

## Contribute

Join us in building the future of LLM applications.  See the [CONTRIBUTING.md](.github/CONTRIBUTING.md) for guidelines.

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

*   [Andrej Karpathy](https://x.com/karpathy/status/1937902205765607626) for the "context engineering" concept.
*   The open-source community.
*   All contributors.