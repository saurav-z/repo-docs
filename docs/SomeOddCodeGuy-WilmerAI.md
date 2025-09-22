# WilmerAI: Intelligent Prompt Routing and Complex Task Orchestration

**Tired of basic AI chatbots? WilmerAI uses advanced workflows to understand the context of your entire conversation, not just the last message, for smarter, more effective AI interactions.**  Learn more about WilmerAI on [GitHub](https://github.com/SomeOddCodeGuy/WilmerAI).

## Key Features:

*   **Advanced Contextual Routing:**
    *   Analyzes the **entire conversation history** for deeper understanding of user intent.
    *   Uses **Prompt Routing** to select the best specialized workflow and **In-Workflow Routing** for conditional logic.
*   **Core: Node-Based Workflow Engine:**
    *   Processes requests through **JSON-defined workflows** with sequences of steps (nodes).
    *   Enables complex, chained-thought processes by passing node outputs as inputs to the next.
*   **Multi-LLM & Multi-Tool Orchestration:**
    *   Each node can connect to a **different LLM endpoint or execute a tool.**
    *   Allows for orchestration of the best model for each part of a task.
*   **Modular & Reusable Workflows:**
    *   Build self-contained workflows for common tasks.
    *   Execute them as a single, reusable node inside other, larger workflows, which simplifies design.
*   **Stateful Conversation Memory:**
    *   Employs a **three-part memory system** (chronological summary, rolling summary, vector database) to provide context.
*   **Adaptable API Gateway:**
    *   Exposes OpenAI- and Ollama-compatible API endpoints.
    *   Connects to existing front-end applications and tools without modification.
*   **Flexible Backend Connectors:**
    *   Connects to various LLM backends including OpenAI, Ollama, and KoboldCpp.
    *   Uses a configuration system of Endpoints, API Types, and Presets.
*   **MCP Server Tool Integration:**
    *   Experimental support for MCP server tool calling using MCPO.

## Workflow Power:

*   **Semi-Autonomous Workflows** - Determine what tools and when to use them.
*   **Iterative LLM Calls** - Improve performance with follow-up questions and chained reasoning.
*   **Distributed LLMs** - Leverage multiple LLMs across your hardware for enhanced capabilities.

## Visual Examples:

*   [Simple Assistant Workflow](Doc_Resources/Media/Images/Wilmer-Assistant-Workflow-Example.jpg)
*   [Prompt Routing Example](Doc_Resources/Media/Images/Wilmer-Categorization-Workflow-Example.png)
*   [Group Chat to Different LLMs](Doc_Resources/Media/Images/Wilmer-Groupchat-Workflow-Example.png)
*   [UX Workflow Example](Doc_Resources/Media/Images/Wilmer-Simple-Coding-Workflow-Example.jpg)
*   [No-RAG vs RAG](Doc_Resources/Media/Gifs/Search-Gif.gif)

## Quick Setup

*   **Guides:** Find setup guides in the [User Documentation](Docs/_User_Documentation/README.md)
    *   [Wilmer API Setup](Docs/_User_Documentation/Setup/_Getting-Start_Wilmer-Api.md)
    *   [Wilmer with Open WebUI](Docs/_User_Documentation/Setup/Open-WebUI.md)
    *   [Wilmer With SillyTavern](Docs/_User_Documentation/Setup/SillyTavern.md)
*   **Video:**
    [![WilmerAI and Open WebUI Install on Fresh Windows 11 Desktop](https://img.youtube.com/vi/KDpbxHMXmTs/0.jpg)](https://www.youtube.com/watch?v=KDpbxHMXmTs "WilmerAI and Open WebUI Install on Fresh Windows 11 Desktop")

## API Endpoints:

WilmerAI offers the following APIs for easy integration:

*   OpenAI Compatible v1/completions (requires Wilmer Prompt Template)
*   OpenAI Compatible chat/completions
*   Ollama Compatible api/generate (requires Wilmer Prompt Template)
*   Ollama Compatible api/chat

## Backend Connections:

Connect to a variety of LLM backends:

*   OpenAI
*   Ollama
*   KoboldCpp

## License

WilmerAI is licensed under the GNU General Public License v3.

## Contact

For feedback, requests, or inquiries: [WilmerAI.Project@gmail.com](mailto:WilmerAI.Project@gmail.com)