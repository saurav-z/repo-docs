# Context Engineering: Master the Art and Science of LLM Context

**Go beyond prompt engineering and unlock the full potential of Large Language Models (LLMs) by mastering context engineering.** Learn how to design, orchestrate, and optimize the information LLMs receive to achieve superior results. Check out the original repo at [https://github.com/davidkimai/Context-Engineering](https://github.com/davidkimai/Context-Engineering)

<img width="1600" height="400" alt="Context Engineering Overview" src="https://github.com/user-attachments/assets/f41f9664-b707-4291-98c8-5bab3054a572" />

> *"Context engineering is the delicate art and science of filling the context window with just the right information for the next step." — [Andrej Karpathy](https://x.com/karpathy/status/1937902205765607626)*

## Key Features

*   **Comprehensive Course:**  A step-by-step handbook covering foundational concepts to advanced techniques.
*   **First-Principles Approach:**  Learn from the ground up, building a strong understanding of context engineering principles.
*   **Practical Examples:**  Hands-on examples and code snippets to guide your learning and experimentation.
*   **Cutting-Edge Research:**  Incorporates the latest research on context optimization, memory systems, cognitive tools, and more.
*   **Visualizations and Diagrams:**  Intuitive diagrams and illustrations to help you grasp complex concepts.

## What is Context Engineering?

Context Engineering is the process of shaping the complete information payload provided to an LLM at inference time. It goes beyond simple prompts, encompassing all the structured components the model needs to plausibly accomplish a task.

> "Providing 'cognitive tools' to GPT-4.1 increases its pass@1 performance on AIME2024 from 26.7% to 43.3%, bringing it very close to the performance of o1-preview." – [IBM Zurich](https://www.arxiv.org/pdf/2506.12115)

```
                    Prompt Engineering  │  Context Engineering
                       ↓                │            ↓                      
               "What you say"           │  "Everything else the model sees"
             (Single instruction)       │    (Examples, memory, retrieval,
                                        │     tools, state, control flow)
```

## Learning Path

The repository offers a structured learning path:

1.  **Foundations:** Understanding core concepts.
2.  **Hands-on Guides:** Walkthroughs to accelerate your progress.
3.  **Templates:** Customizable code snippets to implement in your projects.
4.  **Examples:** Real-world implementations for reference.
5.  **Reference:** Deep dives and an evaluation cookbook.

## Core Concepts Covered

*   **Token Budget:**  Optimize token usage for cost and speed.
*   **Few-Shot Learning:**  Teach through examples.
*   **Memory Systems:**  Enable stateful interactions.
*   **Retrieval Augmentation:**  Ground responses in fact using the tools.
*   **Control Flow:**  Break down tasks into manageable steps.
*   **Context Pruning:**  Remove irrelevant information.
*   **Metrics & Evaluation:**  Measure and refine performance.
*   **Cognitive Tools & Prompt Programming:**  Build custom tools and templates.
*   **Neural Field Theory:** Use LLMs as a dynamic neural field.
*   **Symbolic Mechanisms:** Build smarter systems by combining and optimizing symbolic architectures
*   **Quantum Semantics:** Design context systems leveraging superpositional techniques

## Quick Start

1.  **Read:** [`00_foundations/01_atoms_prompting.md`](00_foundations/01_atoms_prompting.md) (5 min) to understand prompts.
2.  **Run:** [`10_guides_zero_to_hero/01_min_prompt.py`](10_guides_zero_to_hero/01_min_prompt.py) (Jupyter Notebook) to experiment.
3.  **Explore:**  [`20_templates/minimal_context.yaml`](20_templates/minimal_context.yaml) to copy and paste templates.
4.  **Study:** [`30_examples/00_toy_chatbot/`](30_examples/00_toy_chatbot/) to understand implementations.

## Research Highlights

This repository incorporates the latest findings from leading research papers, including:

### Memory + Reasoning

### **[MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents - Singapore-MIT June 2025](https://www.arxiv.org/pdf/2506.15841)**

### **[Eliciting Reasoning in Language Models with Cognitive Tools - IBM Zurich June 2025](https://www.arxiv.org/pdf/2506.12115)**

### [Emergent Symbolic Mechanisms Support Abstract Reasoning in Large Language Models - ICML Princeton June 18, 2025](https://openreview.net/forum?id=y1SnRPDWx4)

## Ecosystem

*   [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/davidkimai/Context-Engineering)
*   [DeepGraph](https://www.deepgraph.co/davidkimai/Context-Engineering)
*   [Chat with NotebookLM + Podcast Deep Dive](https://notebooklm.google.com/notebook/0c6e4dc6-9c30-4f53-8e1a-05cc9ff3bc7e)
*   [![Discord](https://img.shields.io/badge/Discord-join%20chat-7289DA.svg?logo=discord")](https://discord.gg/JeFENHNNNQ)

## Contribute

We welcome contributions! See the [CONTRIBUTING.md](.github/CONTRIBUTING.md) file for guidelines.

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

*   [Andrej Karpathy](https://x.com/karpathy/status/1937902205765607626)
*   All contributors and the open source community