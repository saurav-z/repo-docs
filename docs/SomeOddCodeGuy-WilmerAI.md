# WilmerAI: Unlock Advanced Conversational AI with Contextual Routing and Customizable Workflows

**Tired of chatbots that misunderstand your requests? WilmerAI uses advanced routing and workflows to deliver nuanced, context-aware responses.**

[![WilmerAI and Open WebUI Install on Fresh Windows 11 Desktop](https://img.youtube.com/vi/KDpbxHMXmTs/0.jpg)](https://www.youtube.com/watch?v=KDpbxHMXmTs "WilmerAI and Open WebUI Install on Fresh Windows 11 Desktop")

[Visit the WilmerAI GitHub Repository](https://github.com/SomeOddCodeGuy/WilmerAI) to learn more.

## Key Features

*   **Advanced Contextual Routing:**
    *   Understands the *entire* conversation history for superior intent recognition.
    *   Uses *Prompt Routing* to select appropriate workflows (Coding, Factual, Creative, etc.).
    *   Employs *In-Workflow Routing* for dynamic decision-making based on node outputs.

*   **Core: Node-Based Workflow Engine:**
    *   Executes user requests through workflows defined in JSON files.
    *   Nodes perform specific tasks, with outputs passed as inputs for complex processes.
    *   Enables the creation of modular and reusable workflows.

*   **Multi-LLM & Multi-Tool Orchestration:**
    *   Connects each workflow node to *different* LLM endpoints or external tools.
    *   Optimizes tasks by assigning the best model for each step.

*   **Stateful Conversation Memory:**
    *   Maintains context with a chronological summary file, rolling summary, and a searchable vector database.

*   **Adaptable API Gateway:**
    *   Exposes OpenAI- and Ollama-compatible API endpoints for seamless integration with existing applications.

*   **Flexible Backend Connectors:**
    *   Connects to various LLM backends (OpenAI, Ollama, KoboldCpp) through configurable Endpoints, API Types, and Presets.

*   **MCP Server Tool Integration:**
    *   New experimental support for MCP server tool calling using MCPO.

## Why WilmerAI?

Built to overcome the limitations of traditional routers, WilmerAI utilizes workflows to give granular control over how LLMs complete tasks. It empowers users to experiment with prompting styles and customize categorization workflows to optimize the routing process.

## Get Started

*   **Setup Guides:**
    *   [User Documents Setup Starting Guide](Docs/_User_Documentation/Setup/_Getting-Start_Wilmer-Api.md)
    *   [Wilmer with Open WebUI](Docs/_User_Documentation/Setup/Open-WebUI.md)
    *   [Wilmer With SillyTavern](Docs/_User_Documentation/Setup/SillyTavern.md)
*   **API Endpoints:**
    *   OpenAI Compatible v1/completions (requires [Wilmer Prompt Template](Public/Configs/PromptTemplates/wilmerai.json))
    *   OpenAI Compatible chat/completions
    *   Ollama Compatible api/generate (requires [Wilmer Prompt Template](Public/Configs/PromptTemplates/wilmerai.json))
    *   Ollama Compatible api/chat

## Contact

For questions or feedback, reach out to: WilmerAI.Project@gmail.com