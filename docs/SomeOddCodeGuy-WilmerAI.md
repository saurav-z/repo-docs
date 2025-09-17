# WilmerAI: Advanced Semantic Prompt Routing and Task Orchestration

**Unlock the power of intelligent context with WilmerAI, a node-based workflow engine designed to revolutionize how language models interact with your applications.** ([Original Repo](https://github.com/SomeOddCodeGuy/WilmerAI))

## Key Features

*   **Advanced Contextual Routing:** Analyze entire conversation histories, not just the last message, to understand user intent and select the most appropriate workflow. This is powered by:
    *   **Prompt Routing:** Directs user requests to specialized workflows.
    *   **In-Workflow Routing:** Provides conditional "if/then" logic for dynamic process adjustments.

*   **Core: Node-Based Workflow Engine:** Define complex, multi-step processes using JSON-based workflows, where each node executes a specific task and passes its output to the next.

*   **Multi-LLM & Multi-Tool Orchestration:** Seamlessly integrate different language models and external tools within a single workflow to optimize task performance.

*   **Modular & Reusable Workflows:** Build self-contained workflows for common tasks and reuse them within larger, complex workflows.

*   **Stateful Conversation Memory:** Maintains context with a chronological summary, a rolling summary, and a searchable vector database for Retrieval-Augmented Generation (RAG).

*   **Adaptable API Gateway:** Connect your existing front-end applications effortlessly with OpenAI and Ollama-compatible API endpoints.

*   **Flexible Backend Connectors:** Easily connect to various LLM backends like OpenAI, Ollama, and KoboldCpp through a simple configuration system.

*   **MCP Server Tool Integration using MCPO:** New support for MCP server tool calling using MCPO, allowing tool use mid-workflow (thanks to [iSevenDays](https://github.com/iSevenDays)). More info can be found in the [ReadMe](Public/modules/README_MCP_TOOLS.md).

## Getting Started

*   **[User Documentation Setup Starting Guide](Docs/_User_Documentation/Setup/_Getting-Start_Wilmer-Api.md)**
*   **[Setting up Wilmer with Open WebUI](Docs/_User_Documentation/Setup/Open-WebUI.md)**
*   **[Setting up Wilmer with SillyTavern](Docs/_User_Documentation/Setup/SillyTavern.md).**

## Key Benefits

*   Enhance the capabilities of your existing front-end applications.
*   Utilize a single prompt to orchestrate multiple LLMs and external tools.
*   Easily manage and optimize your token usage.

## Additional Information

*   **[User Documentation](Docs/_User_Documentation/README.md)**
*   **[Developer Documentation](Docs/Developer_Docs/README.md)**

## Contact

For feedback, requests, or to say hello: WilmerAI.Project@gmail.com