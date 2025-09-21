# Context Engineering: Beyond Prompt Engineering 

**Unlock the full potential of Large Language Models by mastering context design, orchestration, and optimization. Dive into the emerging field of Context Engineering and transform how you interact with AI.** ([Original Repo](https://github.com/davidkimai/Context-Engineering))

<img width="1600" height="400" alt="image" src="https://github.com/user-attachments/assets/f41f9664-b707-4291-98c8-5bab3054a572" />

> *"Context engineering is the delicate art and science of filling the context window with just the right information for the next step." — [Andrej Karpathy](https://x.com/karpathy/status/1937902205765607626)*

## Key Features:

*   **First-Principles Approach:** Learn context engineering through a structured, progressive learning path inspired by biological systems.
*   **Practical Guides & Examples:** Get hands-on with code and see complete implementations to build your skills.
*   **Comprehensive Course:** A course is under construction, offering a deep dive from foundational concepts to advanced techniques.
*   **Cutting-Edge Research:** Stay up-to-date with the latest advancements, including research from ICML, IBM, NeurIPS, and more.
*   **Agent Commands Integration:** Support for popular LLM platforms like Claude Code, OpenCode, and Gemini CLI, and more.
*   **Visual Learning:** Concepts are explained using clear diagrams and analogies for easier understanding.

## Core Concepts Covered:

*   **Token Budget:** Optimize context to reduce costs and improve performance.
*   **Few-Shot Learning:** Improve model performance by demonstrating examples.
*   **Memory Systems:** Enable stateful, coherent interactions.
*   **Retrieval Augmentation:** Ground responses in facts by injecting relevant information.
*   **Control Flow:** Structure complex tasks into manageable steps.
*   **Context Pruning:** Keep only essential information in context.
*   **Metrics & Evaluation:** Measure and optimize context effectiveness.
*   **Cognitive Tools:** Build custom tools and templates.
*   **Neural Field Theory:** Model context as a dynamic neural field.
*   **Symbolic Mechanisms:** Enable advanced reasoning.
*   **Quantum Semantics:** Explore meaning in context.

## Why Context Engineering Matters:

Context Engineering expands beyond prompt engineering by focusing on the complete information payload delivered to LLMs. It is a paradigm shift in the way we interact with and leverage the power of Large Language Models.

```
                    Prompt Engineering  │  Context Engineering
                       ↓                │            ↓                      
               "What you say"           │  "Everything else the model sees"
             (Single instruction)       │    (Examples, memory, retrieval,
                                        │     tools, state, control flow)
```

## Learning Path:

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

## Research Highlights:

*   **MEM1 - Singapore-MIT:** Agent memory and reasoning.
*   **IBM Zurich:** Cognitive tools for eliciting reasoning.
*   **ICML Princeton:** Emergent symbolic mechanisms in LLMs.

See examples below

## Research Evidence 
## Memory + Reasoning

### **[MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents - Singapore-MIT June 2025](https://www.arxiv.org/pdf/2506.15841)**

> “Our results demonstrate the promise of reasoning-driven memory consolidation as a scalable alternative to existing solutions for training long-horizon interactive agents, where both efficiency and performance are optimized." — [Singapore-MIT](https://arxiv.org/pdf/2506.15841)

![image](https://github.com/user-attachments/assets/16e3f241-5f44-4ed5-9622-f0b4acbb67b0)

1. **MEM1 trains AI agents to keep only what matters—merging memory and reasoning at every step—so they never get overwhelmed, no matter how long the task.**

2. **Instead of piling up endless context, MEM1 compresses each interaction into a compact “internal state,” just like a smart note that gets updated, not recopied.**

3. **By blending memory and thinking into a single flow, MEM1 learns to remember only the essentials—making agents faster, sharper, and able to handle much longer conversations.**

4. **Everything the agent does is tagged and structured, so each action, question, or fact is clear and easy to audit—no more mystery meat memory.**

5. **With every cycle, old clutter is pruned and only the latest, most relevant insights are carried forward, mirroring how expert problem-solvers distill their notes.**

6. **MEM1 proves that recursive, protocol-driven memory—where you always refine and integrate—outperforms traditional “just add more context” approaches in both speed and accuracy.**
## Cognitive Tools

### **[Eliciting Reasoning in Language Models with Cognitive Tools - IBM Zurich June 2025](https://www.arxiv.org/pdf/2506.12115)**

### Prompts and Prompt Programs as Reasoning Tool Calls
> “Cognitive tools” encapsulate reasoning operations within the LLM itself — [IBM Zurich](https://www.arxiv.org/pdf/2506.12115)



![image](https://github.com/user-attachments/assets/cd06c3f5-5a0b-4ee7-bbba-2f9f243f70ae)

> **These cognitive tools (structured prompt templates as tool calls) break down the problem by identifying the main concepts at hand, extracting relevant information in the question, and highlighting meaningful properties, theorems, and techniques that
might be helpful in solving the problem.**

![image](https://github.com/user-attachments/assets/f7ce8605-6fa3-494f-94cd-94e6b23032b6)


> **These templates scaffold reasoning layers similar to cognitive mental shortcuts, commonly studied as "heuristics".**

1. **This research shows that breaking complex tasks into modular “cognitive tools” lets AI solve problems more thoughtfully—mirroring how expert humans reason step by step.**

2. **Instead of relying on a single, big prompt, the model calls specialized prompt templates, aka cognitive tools like “understand question,” “recall related,” “examine answer,” and “backtracking”—each handling a distinct mental operation.**

3. **Cognitive tools work like inner mental shortcuts: the AI picks the right program at each stage and runs it to plan its reasoning and downstream actions before conducting the task for greater accuracy and flexibility.**

4. **By compartmentalizing reasoning steps into modular blocks, these tools prevent confusion, reduce error, and make the model’s thought process transparent and auditable—even on hard math problems.**

5. **This modular approach upgrades both open and closed models—boosting real-world math problem-solving and approaching the performance of advanced RL-trained “reasoning” models, without extra training.**

6. **The results suggest that the seeds of powerful reasoning are already inside large language models—cognitive tools simply unlock and orchestrate these abilities, offering a transparent, efficient, and interpretable alternative to black-box tuning.**
## Emergent Symbols

## **[Emergent Symbolic Mechanisms Support Abstract Reasoning in Large Language Models - ICML Princeton June 18, 2025](https://openreview.net/forum?id=y1SnRPDWx4)**


![image](https://github.com/user-attachments/assets/76c6e6cb-b65d-4af7-95a5-6d52aee7efc0)

> **TL;DR: A three-stage architecture is identified that supports abstract reasoning in LLMs via a set of emergent symbol-processing mechanisms.**
>
>


**These include symbolic induction heads, symbolic abstraction heads, and retrieval heads.**

**1. In early layers, symbol abstraction heads convert input tokens to abstract variables based on the relations between those tokens.**

**2. In intermediate layers, symbolic induction heads perform sequence induction over these abstract variables.**

**3. Finally, in later layers, retrieval heads predict the next token by retrieving the value associated with the predicted abstract variable.**

**These results point toward a resolution of the longstanding debate between symbolic and neural network approaches, suggesting that emergent reasoning in neural networks depends on the emergence of symbolic mechanisms.** — [**ICML Princeton**](https://openreview.net/forum?id=y1SnRPDWx4) 


![image](https://github.com/user-attachments/assets/2428544e-332a-4e32-9070-9f9d8716d491)


>
> **Why Useful?**
>
>
> **This supports why Markdown, Json, and similar structured, symbolic formats are more easily LLM parsable**
>
> **Concept: Collaborate with agents to apply delimiters, syntax, symbols, symbolic words, metaphors and structure to improve reasoning/context/memory/persistence during inference**

1. **This paper proves that large language models develop their own inner symbolic “logic circuits”—enabling them to reason with abstract variables, not just surface word patterns.**

2. **LLMs show a three-stage process: first abstracting symbols from input, then reasoning over these variables, and finally mapping the abstract answer back to real-world tokens.**

3. **These emergent mechanisms mean LLMs don’t just memorize—they actually create internal, flexible representations that let them generalize to new problems and analogies.**

4. **Attention heads in early layers act like “symbol extractors,” intermediate heads perform symbolic reasoning, and late heads retrieve the concrete answer—mirroring human-like abstraction and retrieval.**

5. **By running targeted experiments and interventions, the authors show these symbolic processes are both necessary and sufficient for abstract reasoning, across multiple models and tasks.**

6. **The results bridge the historic gap between symbolic AI and neural nets—showing that, at scale, neural networks can invent and use symbolic machinery, supporting real generalization and reasoning.**

## Quick Start:

1.  **Read `00_foundations/01_atoms_prompting.md`** (5 min)
2.  **Run `10_guides_zero_to_hero/01_min_prompt.py`** (Jupyter Notebook style)
3.  **Explore `20_templates/minimal_context.yaml`**
4.  **Study `30_examples/00_toy_chatbot/`**

## Contribute:

We welcome contributions! Review the [CONTRIBUTING.md](.github/CONTRIBUTING.md) file for details.

## License:

[MIT License](LICENSE)

## Citation:

```bibtex
@misc{context-engineering,
  author = {Context Engineering Contributors},
  title = {Context Engineering: Beyond Prompt Engineering},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/davidkimai/context-engineering}
}
```

## Acknowledgements:

*   [Andrej Karpathy](https://x.com/karpathy/status/1937902205765607626) for inspiring this work.
*   All contributors and the open-source community.