# Context Engineering: Master the Art and Science of LLM Context

**Unleash the full potential of Large Language Models (LLMs) by mastering context engineering – moving beyond prompts to design, orchestrate, and optimize the entire information payload.**  [Explore the original repository](https://github.com/davidkimai/Context-Engineering).

> *"Context engineering is the delicate art and science of filling the context window with just the right information for the next step."* — [Andrej Karpathy](https://x.com/karpathy/status/1937902205765607626)

<div align="center">
  
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/davidkimai/Context-Engineering)
</div>

## Key Features

*   **Comprehensive Learning Path:** From foundational principles to advanced techniques.
*   **First-Principles Approach:** Built on a biological metaphor, progressing from atoms (single prompts) to neural & semantic field theory (context as fields).
*   **Hands-on Examples:** Practical code and templates for immediate application.
*   **Visualizations:** Conceptual clarity with diagrams, charts, and real-world analogies.
*   **Research-Driven:** Incorporates the latest findings from leading AI research (ICML, IBM, NeurIPS, etc.).
*   **Community-Focused:** A space to share knowledge, learn, and contribute.

## Core Concepts

| Concept                  | What It Is                                     | Why It Matters                                                        |
| ------------------------ | ---------------------------------------------- | --------------------------------------------------------------------- |
| **Token Budget**         | Optimizing every token in your context          | More tokens = more $$ and slower responses                            |
| **Few-Shot Learning**    | Teaching by showing examples                    | Often works better than explanation alone                              |
| **Memory Systems**       | Persisting information across turns            | Enables stateful, coherent interactions                                |
| **Retrieval Augmentation** | Finding & injecting relevant documents           | Grounds responses in facts, reduces hallucination                     |
| **Control Flow**           | Breaking complex tasks into steps                 | Solve harder problems with simpler prompts                             |
| **Context Pruning**        | Removing irrelevant information                 | Keep only what's necessary for performance                            |
| **Metrics & Evaluation**  | Measuring context effectiveness                 | Iterative optimization of token use vs. quality                       |
| **Cognitive Tools & Prompt Programming**  | Learm to build custom tools and templates  | Prompt programming enables new layers for context engineering                  |
| **Neural Field Theory**| Context as a Neural Field | Modeling context as a dynamic neural field allows for iterative context updating |
| **Symbolic Mechanisms** | Symbolic architectures enable higher order reasoning | Smarter systems = less work |
| **Quantum Semantics** |  Meaning as observer-dependent  | Design context systems leveraging superpositional techniques |

## Learning Path & Quick Start

1.  **Foundations:** Start with [00_foundations/01_atoms_prompting.md](00_foundations/01_atoms_prompting.md) (5 min) to understand why prompts alone fall short.
2.  **Experiment:** Run [10_guides_zero_to_hero/01_min_prompt.py](10_guides_zero_to_hero/01_min_prompt.py) (Jupyter Notebook style) for a minimal working example.
3.  **Template:** Explore [20_templates/minimal_context.yaml](20_templates/minimal_context.yaml) and adapt it to your projects.
4.  **Examples:** Study [30_examples/00_toy_chatbot/](30_examples/00_toy_chatbot/) for a complete implementation.

## Roadmap

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

Explore how cutting-edge research is shaping the future of context engineering:

### **Memory + Reasoning**

### **[MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents - Singapore-MIT June 2025](https://www.arxiv.org/pdf/2506.15841)**

> “Our results demonstrate the promise of reasoning-driven memory consolidation as a scalable alternative to existing solutions for training long-horizon interactive agents, where both efficiency and performance are optimized." — [Singapore-MIT](https://arxiv.org/pdf/2506.15841)

*   **MEM1 trains AI agents to keep only what matters—merging memory and reasoning at every step—so they never get overwhelmed, no matter how long the task.**
*   **Instead of piling up endless context, MEM1 compresses each interaction into a compact “internal state,” just like a smart note that gets updated, not recopied.**
*   **By blending memory and thinking into a single flow, MEM1 learns to remember only the essentials—making agents faster, sharper, and able to handle much longer conversations.**
*   **Everything the agent does is tagged and structured, so each action, question, or fact is clear and easy to audit—no more mystery meat memory.**
*   **With every cycle, old clutter is pruned and only the latest, most relevant insights are carried forward, mirroring how expert problem-solvers distill their notes.**
*   **MEM1 proves that recursive, protocol-driven memory—where you always refine and integrate—outperforms traditional “just add more context” approaches in both speed and accuracy.**
### Cognitive Tools

### **[Eliciting Reasoning in Language Models with Cognitive Tools - IBM Zurich June 2025](https://www.arxiv.org/pdf/2506.12115)**

### Prompts and Prompt Programs as Reasoning Tool Calls
> “Cognitive tools” encapsulate reasoning operations within the LLM itself — [IBM Zurich](https://www.arxiv.org/pdf/2506.12115)

*   **This research shows that breaking complex tasks into modular “cognitive tools” lets AI solve problems more thoughtfully—mirroring how expert humans reason step by step.**
*   **Instead of relying on a single, big prompt, the model calls specialized prompt templates, aka cognitive tools like “understand question,” “recall related,” “examine answer,” and “backtracking”—each handling a distinct mental operation.**
*   **Cognitive tools work like inner mental shortcuts: the AI picks the right program at each stage and runs it to plan its reasoning and downstream actions before conducting the task for greater accuracy and flexibility.**
*   **By compartmentalizing reasoning steps into modular blocks, these tools prevent confusion, reduce error, and make the model’s thought process transparent and auditable—even on hard math problems.**
*   **This modular approach upgrades both open and closed models—boosting real-world math problem-solving and approaching the performance of advanced RL-trained “reasoning” models, without extra training.**
*   **The results suggest that the seeds of powerful reasoning are already inside large language models—cognitive tools simply unlock and orchestrate these abilities, offering a transparent, efficient, and interpretable alternative to black-box tuning.**

### Emergent Symbols

### **[Emergent Symbolic Mechanisms Support Abstract Reasoning in Large Language Models - ICML Princeton June 18, 2025](https://openreview.net/forum?id=y1SnRPDWx4)**

*   **This paper proves that large language models develop their own inner symbolic “logic circuits”—enabling them to reason with abstract variables, not just surface word patterns.**
*   **LLMs show a three-stage process: first abstracting symbols from input, then reasoning over these variables, and finally mapping the abstract answer back to real-world tokens.**
*   **These emergent mechanisms mean LLMs don’t just memorize—they actually create internal, flexible representations that let them generalize to new problems and analogies.**
*   **Attention heads in early layers act like “symbol extractors,” intermediate heads perform symbolic reasoning, and late heads retrieve the concrete answer—mirroring human-like abstraction and retrieval.**
*   **By running targeted experiments and interventions, the authors show these symbolic processes are both necessary and sufficient for abstract reasoning, across multiple models and tasks.**
*   **The results bridge the historic gap between symbolic AI and neural nets—showing that, at scale, neural networks can invent and use symbolic machinery, supporting real generalization and reasoning.**

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=davidkimai/Context-Engineering&type=Date)](https://www.star-history.com/#davidkimai/Context-Engineering&Date)

## Contribute

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
> I've been looking forward to this being conceptualized and formalized as there wasn't a prior established field. Prompt engineering receives quite the stigma and doesn't quite cover what most researchers and I do.

-   [Andrej Karpathy](https://x.com/karpathy/status/1937902205765607626) for coining "context engineering" and inspiring this repo
-   All contributors and the open source community