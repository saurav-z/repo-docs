# WilmerAI: Expert Prompt Routing and Workflow Orchestration for LLMs

**Unlock the power of advanced AI with WilmerAI, a node-based workflow engine designed to intelligently route and orchestrate LLMs for enhanced context and performance.** ([View on GitHub](https://github.com/SomeOddCodeGuy/WilmerAI))

## Key Features

*   **Advanced Contextual Routing:** Analyze entire conversation histories, not just the latest message, for accurate prompt categorization and routing using:
    *   **Prompt Routing:** Directs user requests to the most suitable specialized workflow.
    *   **In-Workflow Routing:** Dynamic conditional logic allows for adaptive processes within each workflow.

*   **Node-Based Workflow Engine:** Build complex AI processes with JSON-defined workflows, consisting of individual nodes that each perform a specific task, allowing for advanced chained-thought processes.

*   **Multi-LLM & Multi-Tool Orchestration:** Seamlessly integrate different LLMs and external tools within a single workflow, optimizing for specialized tasks and enhanced results.

*   **Modular & Reusable Workflows:** Create reusable workflows for common tasks, simplifying complex agent design.

*   **Stateful Conversation Memory:** Maintain context and improve routing with a chronological summary, rolling summary, and vector database for Retrieval-Augmented Generation (RAG).

*   **Adaptable API Gateway:** Connect your existing front-end applications and tools without modification using OpenAI- and Ollama-compatible API endpoints.

*   **Flexible Backend Connectors:** Easily connect to various LLM backends, including OpenAI, Ollama, and KoboldCpp, using a simple configuration system.

*   **MCP Server Tool Integration:** Supports MCP server tool calling using MCPO, enabling tool use mid-workflow. (Thanks to [iSevenDays](https://github.com/iSevenDays)!) More info in [/Public/modules/README_MCP_TOOLS.md](Public/modules/README_MCP_TOOLS.md).

## Why Use WilmerAI?

WilmerAI empowers you to move beyond simple AI interactions. It allows you to build a sophisticated backend using workflows, making it possible to design complex solutions without needing to modify the existing frontend tool. It's flexible enough to allow for both the routing of prompts to LLMs, and even the use of completely bypassing the routing, so the user can hit a particular workflow directly.

## Getting Started

*   **Installation:** Follow the installation instructions in the original README to get started.

*   **Configuration:** WilmerAI is configured via JSON files. Find essential settings and configuration examples in the "Public" folder, including pre-made user templates to begin experimenting.

*   **User Documentation:** [/Docs/_User_Documentation/README.md](Docs/_User_Documentation/README.md)

*   **Developer Documentation:** [/Docs/Developer_Docs/README.md](Docs/Developer_Docs/README.md)

## Connect with WilmerAI

*   **Email:** WilmerAI.Project@gmail.com

---