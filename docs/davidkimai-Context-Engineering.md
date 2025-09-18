# Context Engineering: Master the Art and Science of Information Orchestration for LLMs

**Go beyond prompt engineering and unlock the true potential of large language models by mastering the art and science of context engineering: the strategic design, orchestration, and optimization of information within the context window.**

[Explore the original repository](https://github.com/davidkimai/Context-Engineering)

<img width="1600" height="400" alt="image" src="https://github.com/user-attachments/assets/f41f9664-b707-4291-98c8-5bab3054a572" />

## Key Features

*   **Comprehensive Handbook:** A first-principles guide moving beyond prompt engineering to context design and optimization.
*   **Practical Learning Path:**  From foundational concepts to advanced techniques, with hands-on examples and code-driven explorations.
*   **Research-Driven Insights:**  Operationalizing the latest research from leading institutions like ICML, IBM, and NeurIPS.
*   **Visual and Conceptual Clarity:**  Leveraging analogies, diagrams, and code to make complex concepts accessible to all.
*   **Community-Driven Development:** Open for contributions and focused on the future of LLM understanding.

## What is Context Engineering?

Context engineering is the strategic design and management of the entire information payload provided to a Language Model at inference time. Context is more than just the prompt; it encompasses examples, memory, retrieval, tools, state, and control flow.

> "Context engineering is the delicate art and science of filling the context window with just the right information for the next step." — [Andrej Karpathy](https://x.com/karpathy/status/1937902205765607626)

## Core Concepts

```
                    Prompt Engineering  │  Context Engineering
                       ↓                │            ↓                      
               "What you say"           │  "Everything else the model sees"
             (Single instruction)       │    (Examples, memory, retrieval,
                                        │     tools, state, control flow)
```

### The Context Engineering Mastery Course - Under Construction

```
╭─────────────────────────────────────────────────────────────╮
│              CONTEXT ENGINEERING MASTERY COURSE             │
│                    From Zero to Frontier                    │
╰─────────────────────────────────────────────────────────────╯
                          ▲
                          │
                 Mathematical Foundations
                  C = A(c₁, c₂, ..., cₙ)
                          │
                          ▼
┌─────────────┬──────────────┬──────────────┬─────────────────┐
│ FOUNDATIONS │ SYSTEM IMPL  │ INTEGRATION  │ FRONTIER        │
│ (Weeks 1-4) │ (Weeks 5-8)  │ (Weeks 9-10) │ (Weeks 11-12)   │
└─────┬───────┴──────┬───────┴──────┬───────┴─────────┬───────┘
      │              │              │                 │
      ▼              ▼              ▼                 ▼
┌─────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Math Models │ │ RAG Systems  │ │ Multi-Agent  │ │ Meta-Recurs  │
│ Components  │ │ Memory Arch  │ │ Orchestrat   │ │ Quantum Sem  │
│ Processing  │ │ Tool Integr  │ │ Field Theory │ │ Self-Improv  │
│ Management  │ │ Agent Systems│ │ Evaluation   │ │ Collaboration│
└─────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
```

## Why This Matters

> "Meaning is not an intrinsic, static property of a semantic expression, but rather an emergent phenomenon" — [Agostino et al. — July 2025, Indiana University](https://arxiv.org/pdf/2506.10077)

Context engineering empowers you to go beyond the basics of prompt engineering by controlling the entire information ecosystem that feeds the LLM. This repository provides a progressive, first-principles approach, drawing on biological metaphors:

```
atoms → molecules → cells → organs → neural systems → neural & semantic field theory 
  │        │         │         │             │                         │        
single    few-     memory +   multi-   cognitive tools +     context = fields +
prompt    shot     agents     agents   operating systems     persistence & resonance
```

## Learning Path & Resources

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

### Quick Start

1.  **Read** [`00_foundations/01_atoms_prompting.md`](00_foundations/01_atoms_prompting.md) (5 min) - Understand why prompts alone often underperform
2.  **Run** [`10_guides_zero_to_hero/01_min_prompt.py`](10_guides_zero_to_hero/01_min_prompt.py) (Jupyter Notebook style) - Experiment with a minimal working example
3.  **Explore** [`20_templates/minimal_context.yaml`](20_templates/minimal_context.yaml) - Copy/paste a template into your own project
4.  **Study** [`30_examples/00_toy_chatbot/`](30_examples/00_toy_chatbot/) - See a complete implementation with context management

## What You'll Learn

| Concept                 | What It Is                                      | Why It Matters                                                   |
| ----------------------- | ----------------------------------------------- | ---------------------------------------------------------------- |
| **Token Budget**        | Optimizing every token in your context          | More tokens = more $$ and slower responses                         |
| **Few-Shot Learning**   | Teaching by showing examples                    | Often works better than explanation alone                          |
| **Memory Systems**      | Persisting information across turns             | Enables stateful, coherent interactions                          |
| **Retrieval Augmentation** | Finding & injecting relevant documents        | Grounds responses in facts, reduces hallucination                  |
| **Control Flow**        | Breaking complex tasks into steps             | Solve harder problems with simpler prompts                         |
| **Context Pruning**     | Removing irrelevant information                | Keep only what's necessary for performance                      |
| **Metrics & Evaluation** | Measuring context effectiveness               | Iterative optimization of token use vs. quality                  |
| **Cognitive Tools & Prompt Programming** | Building custom tools and templates      | Prompt programming enables new layers for context engineering   |
| **Neural Field Theory**  | Context as a Neural Field                       | Modeling context as a dynamic neural field allows for iterative context updating |
| **Symbolic Mechanisms** | Symbolic architectures enable higher order reasoning | Smarter systems = less work                                  |
| **Quantum Semantics** | Meaning as observer-dependent | Design context systems leveraging superpositional techniques |

## Research Evidence & Key Papers
>  [Context Engineering Survey-Review of 1400 Research Papers](https://arxiv.org/pdf/2507.13334)

### Memory + Reasoning
*   **[MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents - Singapore-MIT June 2025](https://www.arxiv.org/pdf/2506.15841)**

### Cognitive Tools
*   **[Eliciting Reasoning in Language Models with Cognitive Tools - IBM Zurich June 2025](https://www.arxiv.org/pdf/2506.12115)**

### Emergent Symbols
*   **[Emergent Symbolic Mechanisms Support Abstract Reasoning in Large Language Models - ICML Princeton June 18, 2025](https://openreview.net/forum?id=y1SnRPDWx4)**

## Tools and Integrations
*  Support for [Claude Code](https://www.anthropic.com/claude-code) | [OpenCode](https://opencode.ai/) | [Amp](https://sourcegraph.com/amp) | [Kiro](https://kiro.dev/) | [Codex](https://openai.com/codex/) | [Gemini CLI](https://github.com/google-gemini/gemini-cli)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=davidkimai/Context-Engineering&type=Date)](https://www.star-history.com/#davidkimai/Context-Engineering&Date)

## Contributing

We welcome contributions! See [CONTRIBUTING.md](.github/CONTRIBUTING.md) for guidelines.

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

*   [Andrej Karpathy](https://x.com/karpathy/status/1937902205765607626) for coining "context engineering" and inspiring this repo
*   All contributors and the open source community