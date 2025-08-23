# Context Engineering: Mastering LLM Performance Beyond Prompts

**Unlock the full potential of Large Language Models by mastering context engineering – the art and science of curating the perfect information environment for optimal LLM performance.** ([Original Repository](https://github.com/davidkimai/Context-Engineering))

Context engineering goes beyond prompt engineering, focusing on the **entire information payload** provided to an LLM. This includes examples, memory, retrieval tools, state, and control flow, enabling more complex and intelligent applications.

## Key Features

*   **First-Principles Approach:** Learn context engineering from the ground up, building from atomic instructions to complex neural systems and field theory.
*   **Hands-On Tutorials:**  Interactive guides and Jupyter notebooks to experiment with different context engineering techniques.
*   **Practical Examples:** Explore real-world implementations, including chatbot, data annotation, multi-agent orchestration, and more.
*   **Reusable Templates:** Leverage pre-built components, including context structures, control loops, and evaluation metrics, to accelerate development.
*   **Community Driven:** Join a community of practitioners and researchers, and contribute to this open-source resource.

## What You Will Learn

| Concept             | What It Is                                   | Why It Matters                                                        |
| ------------------- | -------------------------------------------- | --------------------------------------------------------------------- |
| **Token Budget**     | Optimizing every token in your context         | More tokens = higher cost & slower responses                           |
| **Few-Shot Learning** | Teaching by showing examples                  | Often more effective than just providing explanations                  |
| **Memory Systems**   | Persisting information across turns           | Enables stateful, coherent conversations                               |
| **Retrieval Augmentation** | Finding & injecting relevant documents      | Grounds responses in facts, reduces hallucination                      |
| **Control Flow**     | Breaking complex tasks into steps            | Solve harder problems with simpler prompts                             |
| **Context Pruning**  | Removing irrelevant information               | Keeps only what is needed to maintain peak performance                |
| **Metrics & Evaluation**| Measuring context effectiveness            | Enables iterative optimization                                        |
| **Cognitive Tools & Prompt Programming** | Build Custom Tools & Templates          | Create new layers for context engineering                                    |
| **Neural Field Theory** | Modeling context as a dynamic neural field | Iterative context updating                                |
| **Symbolic Mechanisms** | Symbolic architectures enabling higher order reasoning  | Smarter systems = less work |
| **Quantum Semantics** |  Meaning as observer-dependent  | Design context systems leveraging superpositional techniques |

## Research-Driven Insights

This repository is continuously updated with the latest research and insights. Key findings include:

*   **MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents**: By compressing each interaction into a compact "internal state," MEM1 agents avoid the problems of endless context expansion.
*   **Eliciting Reasoning in Language Models with Cognitive Tools**: Utilize modular prompt templates to break down complex tasks, enhancing accuracy and transparency.
*   **Emergent Symbolic Mechanisms Support Abstract Reasoning in Large Language Models**: LLMs develop internal "logic circuits," enabling abstract reasoning, generalization, and innovative solutions.

## Getting Started

1.  **Foundations:** Start with  `00_foundations/01_atoms_prompting.md` for understanding prompts.
2.  **Experiment:** Run the hands-on guide `10_guides_zero_to_one/01_min_prompt.ipynb` (Jupyter Notebook style).
3.  **Explore:** Use a template like `20_templates/minimal_context.yaml`.
4.  **Implement:** Study the implementation `30_examples/00_toy_chatbot/`.
5. **Join the Discord:**  [![Discord](https://img.shields.io/badge/Discord-join%20chat-7289DA.svg?logo=discord")](https://discord.gg/JeFENHNNNQ)


## Roadmap
**Comprehensive Course Under Construction:**  See the `00_COURSE` directory for in-progress materials.

```
                    Prompt Engineering  │  Context Engineering
                       ↓                │            ↓                      
               "What you say"           │  "Everything else the model sees"
             (Single instruction)       │    (Examples, memory, retrieval,
                                        │     tools, state, control flow)
```

*   **[Comprehensive Course Under Construction](https://github.com/davidkimai/Context-Engineering/tree/main/00_COURSE)**
*   **[Context Engineering Survey-Review of 1400 Research Papers](https://arxiv.org/pdf/2507.13334)**
*   **[Awesome Context Engineering Repo](https://github.com/Meirtz/Awesome-Context-Engineering)**

## Dive Deeper

*   **[DeepWiki](https://deepwiki.com/davidkimai/Context-Engineering)**
*   **[DeepGraph](https://www.deepgraph.co/davidkimai/Context-Engineering)**
*   **[Chat with NotebookLM + Podcast Deep Dive](https://notebooklm.google.com/notebook/0c6e4dc6-9c30-4f53-8e1a-05cc9ff3bc7e)**
*   **[On Emergence, Attractors, and Dynamical Systems Theory](https://content.csbs.utah.edu/~butner/systems/DynamicalSystemsIntro.html)**
*   **[Columbia DST](http://wordpress.ei.columbia.edu/ac4/about/our-approach/dynamical-systems-theory/)**

## Contribute

Help us build the future of context engineering!  Check out our [CONTRIBUTING.md](.github/CONTRIBUTING.md) for guidelines.

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

*   [Andrej Karpathy](https://x.com/karpathy/status/1937902205765607626) for inspiring this repo and coining the term "context engineering".
*   All contributors and the open-source community.