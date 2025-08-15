# Context Engineering: Master the Art and Science of LLM Context

**Unlock the full potential of Large Language Models (LLMs) by mastering context engineering – the key to building smarter, more effective AI applications.** Dive into a comprehensive guide to context design, orchestration, and optimization, moving beyond basic prompt engineering to create advanced AI solutions. ([Original Repository](https://github.com/davidkimai/Context-Engineering))

## Key Features:

*   **First-Principles Approach:** Learn the foundational concepts and techniques to build and refine LLM context.
*   **Hands-On Tutorials:** Step-by-step guides and code examples to help you build real-world applications.
*   **Research-Backed Insights:** Explore cutting-edge research on memory, reasoning, and cognitive tools.
*   **Progressive Learning Path:** Master concepts from the basics to advanced techniques.
*   **Code-Driven Examples:** See every concept brought to life with practical, runnable code.

## What is Context Engineering?

Context engineering is a discipline to design the information payload to a LLM at inference time, encompassing all structured informational components. The components consist of prompts, examples, memory, retrieval, tools, state, and control flow.

> *"Context is not just the single prompt users send to an LLM. Context is the complete information payload provided to a LLM at inference time, encompassing all structured informational components that the model needs to plausibly accomplish a given task."* - [**Definition of Context Engineering from A Systematic Analysis of Over 1400 Research Papers](https://arxiv.org/pdf/2507.13334)

```
                    Prompt Engineering  │  Context Engineering
                       ↓                │            ↓                      
               "What you say"           │  "Everything else the model sees"
             (Single instruction)       │    (Examples, memory, retrieval,
                                        │     tools, state, control flow)
```

## Core Concepts:

*   **Token Budgeting:** Optimize context for cost and speed.
*   **Few-Shot Learning:** Teach LLMs by providing clear examples.
*   **Memory Systems:** Enable stateful and coherent interactions.
*   **Retrieval Augmentation (RAG):** Integrate external knowledge sources.
*   **Control Flow:** Structure complex tasks into manageable steps.
*   **Context Pruning:** Remove irrelevant information for optimal performance.
*   **Cognitive Tools & Prompt Programming:** Build custom functions and patterns.
*   **Neural Field Theory:** Modeling context as a dynamic neural field allows for iterative context updating.
*   **Symbolic Mechanisms:** Harness symbolic architectures to enhance reasoning.
*   **Quantum Semantics:** Design systems leveraging superpositional techniques.

## Why Context Engineering Matters

Context engineering is the next frontier for achieving impactful results. By mastering it, you can:

*   Improve accuracy and reduce hallucinations
*   Enable more complex and nuanced interactions
*   Create applications that are more efficient and cost-effective
*   Build cutting-edge AI solutions.

## Learning Path and Structure:

This repository offers a structured learning path, moving from the basics to advanced concepts:

**Level 1: Atoms to Organs - (Basic Context Engineering)**
   *   Single instructions
   *   Few-shot learning
   *   Conversational Chatbots
   *   Multi-agent Systems

**Level 2: Neural Systems and Fields - (Field Theory)**
    *   Reasoning frameworks
    *   Verification tools
    *   Continuous meaning
    *   Attractors & resonance

**Level 3: Protocol System**
    *   Structured templates
    *   Field operations
    *   Emergence protocols
    *   Self-organizing systems

**Level 4: Meta-Recursion**
    *   Self-reflection
    *   Recursive improvement
    *   Interpretable evolution
    *   Self-improving intelligence

## Resources:

*   **Comprehensive Course Under Construction:** [Here is the course content.](https://github.com/davidkimai/Context-Engineering/tree/main/00_COURSE)
*   **Agent Commands:** Support for tools. [`Agent Commands`](https://github.com/davidkimai/Context-Engineering/tree/main/.claude/commands)

### Research Evidence:
*   [MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents - Singapore-MIT June 2025](https://www.arxiv.org/pdf/2506.15841)
*   [Eliciting Reasoning in Language Models with Cognitive Tools - IBM Zurich June 2025](https://www.arxiv.org/pdf/2506.12115)
*   [Emergent Symbolic Mechanisms Support Abstract Reasoning in Large Language Models - ICML Princeton June 18, 2025](https://openreview.net/forum?id=y1SnRPDWx4)

## Contributing:

We welcome contributions! See [.github/CONTRIBUTING.md](.github/CONTRIBUTING.md) for details.

## Star History:

[![Star History Chart](https://api.star-history.com/svg?repos=davidkimai/Context-Engineering&type=Date)](https://www.star-history.com/#davidkimai/Context-Engineering&Date)

## Acknowledgements:

-   [Andrej Karpathy](https://x.com/karpathy/status/1937902205765607626) for coining "context engineering" and inspiring this repo.
-   All contributors and the open-source community.