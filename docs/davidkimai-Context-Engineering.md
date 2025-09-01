# Context Engineering: Master the Art of Information in LLMs

**Go beyond prompt engineering and learn how to design, orchestrate, and optimize the complete context window of your Large Language Models (LLMs).** ([Original Repository](https://github.com/davidkimai/Context-Engineering))

> *"Context engineering is the delicate art and science of filling the context window with just the right information for the next step." — [Andrej Karpathy](https://x.com/karpathy/status/1937902205765607626)*

This repository provides a practical, first-principles approach to context engineering, moving beyond simple prompt design to encompass the entire information ecosystem of your LLMs.

## Key Features

*   **Comprehensive Course:** A structured learning path from foundational concepts to advanced techniques.
*   **Hands-on Examples:**  Real-world projects and code examples to accelerate your learning.
*   **Visualizations:** Understand complex concepts through clear diagrams and illustrations.
*   **Research-Driven:**  Explore cutting-edge research on memory, reasoning, and cognitive tools in LLMs.
*   **Community Focused:**  Join the [Discord](https://discord.gg/JeFENHNNNQ) and contribute to a growing knowledge base.

## What is Context Engineering?

Context engineering is more than just crafting prompts; it's the art of providing a complete informational payload to an LLM at inference time. This includes:

*   **Prompts:** The initial instruction.
*   **Examples:** Demonstrations of desired behavior (few-shot learning).
*   **Memory:**  Information persisted across interactions.
*   **Retrieval:** Dynamically pulling in relevant data from external sources.
*   **Tools & Agents:**  Integrating external tools and/or multi-agent systems.
*   **State & Control Flow:**  Managing conversational context and task steps.

## Core Concepts You Will Master

*   **Token Budgeting:** Optimize token usage for efficiency and cost savings.
*   **Few-Shot Learning:**  Improve performance by providing illustrative examples.
*   **Memory Systems:**  Build stateful, coherent conversations and tasks.
*   **Retrieval Augmentation:**  Ground responses in facts and reduce hallucinations.
*   **Control Flow:**  Break down complex tasks into manageable steps.
*   **Context Pruning:**  Remove irrelevant information to boost performance.
*   **Metrics & Evaluation:**  Measure the effectiveness of context designs.
*   **Cognitive Tools & Prompt Programming:**  Create specialized tools for your LLMs.
*   **Neural Field Theory:** Model context as a dynamic neural field to support iterative updating.
*   **Symbolic Mechanisms:** Leverage symbolic architectures for enhanced reasoning and generalization.
*   **Quantum Semantics:**  Design systems using superpositional techniques for more sophisticated designs.

## Learning Path

*   **Foundations:** Understand core concepts and the limitations of basic prompting.
*   **Hands-on Guides:** Walkthroughs and tutorials to help you implement context engineering techniques.
*   **Templates:** Ready-to-use code snippets that can be copied and adapted for your projects.
*   **Examples:** Explore complete implementations, from simple chatbots to complex applications.
*   **Reference:** Detailed explanations, deep dives, and evaluation methodologies.
*   **Community Contributions:** Share your own implementations and collaborate with other members.

## Research Highlights & Evidence

This repository is informed by the latest research in the field. Some key research areas covered include:

*   **[MEM1: Learning to Synergize Memory and Reasoning](https://www.arxiv.org/pdf/2506.15841):** Explore how to train AI agents to merge memory and reasoning, optimizing both efficiency and performance for long-horizon tasks.
*   **[Eliciting Reasoning in Language Models with Cognitive Tools](https://www.arxiv.org/pdf/2506.12115):** Discover how to use cognitive tools and prompt programs to break down complex tasks into modular steps that mirror human reasoning processes.
*   **[Emergent Symbolic Mechanisms Support Abstract Reasoning](https://openreview.net/forum?id=y1SnRPDWx4):** Learn how LLMs develop their own inner symbolic "logic circuits" to enable reasoning with abstract variables and real-world generalization.

## Get Started Now

1.  **Dive into the Fundamentals:** Begin with [`00_foundations/01_atoms_prompting.md`](00_foundations/01_atoms_prompting.md) (5 min read).
2.  **Experiment with Code:** Run the minimal example in [`10_guides_zero_to_hero/01_min_prompt.py`](10_guides_zero_to_hero/01_min_prompt.py).
3.  **Explore Templates:**  Check out [`20_templates/minimal_context.yaml`](20_templates/minimal_context.yaml).
4.  **Study Examples:** Explore [`30_examples/00_toy_chatbot/`](30_examples/00_toy_chatbot/) for a practical use case.

## Additional Resources

*   [Awesome Context Engineering Repo](https://github.com/Meirtz/Awesome-Context-Engineering)
*   [Context Engineering Survey-Review of 1400 Research Papers](https://arxiv.org/pdf/2507.13334)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=davidkimai/Context-Engineering&type=Date)](https://www.star-history.com/#davidkimai/Context-Engineering&Date)

## Contribute

Help build the future of LLM development!  See the [CONTRIBUTING.md](.github/CONTRIBUTING.md) for guidelines.

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

*   Inspired by [Andrej Karpathy](https://x.com/karpathy/status/1937902205765607626) for coining "context engineering" and the open source community
*   Thanks to all contributors and the open source community