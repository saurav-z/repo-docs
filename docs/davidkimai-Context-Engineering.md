# Context Engineering: Unlock the Power of LLMs with Strategic Context Design

**Go beyond prompt engineering and master the art and science of context engineering—the key to unlocking the true potential of large language models (LLMs).** Discover how to design, orchestrate, and optimize the complete information payload provided to an LLM at inference time, using the original repo: [davidkimai/Context-Engineering](https://github.com/davidkimai/Context-Engineering).

## Key Features

*   **Comprehensive Course:** Access a first-principles handbook, progressing from foundational concepts to advanced techniques.
*   **Cutting-Edge Research:** Learn to operationalize the latest research, including findings from ICML, IBM, NeurIPS, and more.
*   **Practical Examples:** Dive into hands-on tutorials, reusable templates, and real-world implementations to build and test your skills.
*   **Modular Design:** Build cognitive tools and prompt programs using flexible and reusable structures.
*   **Neural Field Theory:** Learn to model context as a dynamic neural field that allows for iterative context updating.
*   **Symbolic Mechanisms:** Explore how symbolic architectures enable higher-order reasoning for smarter, more efficient systems.
*   **Quantum Semantics:** Design context systems leveraging superpositional techniques.

## What is Context Engineering?

Context engineering is a multidisciplinary field that focuses on designing and optimizing the entire context window for LLMs, enabling them to perform complex tasks and deliver superior results. Rather than focusing solely on the prompt, this approach encompasses:

*   Examples, memory, retrieval, tools, state, and control flow.

**Key Components of Context Engineering:**

```
                    Prompt Engineering  │  Context Engineering
                       ↓                │            ↓                      
               "What you say"           │  "Everything else the model sees"
             (Single instruction)       │    (Examples, memory, retrieval,
                                        │     tools, state, control flow)
```

## Core Concepts & Learning Path

The repository follows a structured approach, enabling you to master the intricacies of context engineering.

**Progressive Learning Path:**

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

**What You'll Learn:**

| Concept | What It Is | Why It Matters |
|---------|------------|----------------|
| **Token Budget** | Optimizing every token in your context | More tokens = more $$ and slower responses |
| **Few-Shot Learning** | Teaching by showing examples | Often works better than explanation alone |
| **Memory Systems** | Persisting information across turns | Enables stateful, coherent interactions |
| **Retrieval Augmentation** | Finding & injecting relevant documents | Grounds responses in facts, reduces hallucination |
| **Control Flow** | Breaking complex tasks into steps | Solve harder problems with simpler prompts |
| **Context Pruning** | Removing irrelevant information | Keep only what's necessary for performance |
| **Metrics & Evaluation** | Measuring context effectiveness | Iterative optimization of token use vs. quality |
| **Cognitive Tools & Prompt Programming** | Learm to build custom tools and templates | Prompt programming enables new layers for context engineering |
| **Neural Field Theory** | Context as a Neural Field | Modeling context as a dynamic neural field allows for iterative context updating |
| **Symbolic Mechanisms** | Symbolic architectures enable higher order reasoning | Smarter systems = less work |
| **Quantum Semantics** |  Meaning as observer-dependent  | Design context systems leveraging superpositional techniques |

## Research Highlights & Supporting Evidence

Explore key research papers and real-world examples that demonstrate the power of context engineering.

### **[MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents - Singapore-MIT June 2025](https://www.arxiv.org/pdf/2506.15841)**

>   “Our results demonstrate the promise of reasoning-driven memory consolidation as a scalable alternative to existing solutions for training long-horizon interactive agents, where both efficiency and performance are optimized." — [Singapore-MIT](https://arxiv.org/pdf/2506.15841)

### **[Eliciting Reasoning in Language Models with Cognitive Tools - IBM Zurich June 2025](https://www.arxiv.org/pdf/2506.12115)**

>   “Cognitive tools” encapsulate reasoning operations within the LLM itself — [IBM Zurich](https://www.arxiv.org/pdf/2506.12115)

### **[Emergent Symbolic Mechanisms Support Abstract Reasoning in Large Language Models - ICML Princeton June 18, 2025](https://openreview.net/forum?id=y1SnRPDWx4)**

>   TL;DR: A three-stage architecture is identified that supports abstract reasoning in LLMs via a set of emergent symbol-processing mechanisms.

## Key Resources & Tools

*   **Agent Commands:** Support for various platforms (Claude Code, OpenCode, Amp, Kiro, Codex, Gemini CLI).
*   **Comprehensive Course Under Construction:** Deep dive into context engineering principles and practices.
*   **DeepGraph & DeepWiki:** Explore interactive visualizations and resources.
*   **Chat with NotebookLM + Podcast Deep Dive:** Learn context engineering with NotebookLM
*   **Discord:** Join the community for discussions and support.

## Getting Started

1.  **Read the foundational materials:** Begin with `00_foundations/01_atoms_prompting.md` (5 min).
2.  **Experiment:** Run a minimal working example using  `10_guides_zero_to_one/01_min_prompt.py`.
3.  **Explore templates:** Copy and paste code samples from `20_templates/minimal_context.yaml`.
4.  **Study real-world projects:** Experiment with the complete implementations with context management in `30_examples/00_toy_chatbot/`.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=davidkimai/Context-Engineering&type=Date)](https://www.star-history.com/#davidkimai/Context-Engineering&Date)

## Contribute

We welcome contributions! Check out our [CONTRIBUTING.md](.github/CONTRIBUTING.md) for guidelines.

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

-   [Andrej Karpathy](https://x.com/karpathy/status/1937902205765607626) for coining "context engineering" and inspiring this repo
-   All contributors and the open source community