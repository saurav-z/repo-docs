# WilmerAI: Advanced Semantic Prompt Routing and Workflow Orchestration

**Unlock the power of AI with WilmerAI, a node-based workflow engine designed to intelligently route and orchestrate complex tasks for LLMs.**  [See the original repository](https://github.com/SomeOddCodeGuy/WilmerAI).

## Key Features

*   **Advanced Contextual Routing:** WilmerAI analyzes the entire conversation history to select the most appropriate workflow, understanding intent beyond just the last message.
    *   **Prompt Routing:** Directs requests to specialized workflows (e.g., "Coding," "Factual," "Creative").
    *   **In-Workflow Routing:** Enables dynamic, conditional logic based on the output of previous nodes.
*   **Core: Node-Based Workflow Engine:** Processes requests via JSON-defined workflows, allowing for complex, chained processes.
*   **Multi-LLM & Multi-Tool Orchestration:** Connects each node in a workflow to different LLM endpoints or tools, enabling specialized task execution.
*   **Modular & Reusable Workflows:** Build and execute self-contained workflows as reusable nodes within larger systems.
*   **Stateful Conversation Memory:** Utilizes a three-part memory system (chronological summary, rolling summary, vector database) for context and accuracy.
*   **Adaptable API Gateway:** Exposes OpenAI- and Ollama-compatible API endpoints for seamless integration with existing applications.
*   **Flexible Backend Connectors:** Supports various LLM backends (OpenAI, Ollama, KoboldCpp) using a simple configuration system.
*   **MCP Server Tool Integration:** Experimental support for MCP server tool calling using MCPO, enabling tool use mid-workflow, thanks to [iSevenDays](https://github.com/iSevenDays).

## Why Use WilmerAI?

WilmerAI goes beyond simple prompt routing by enabling the creation of semi-autonomous workflows that give you granular control over LLM interactions. You can easily integrate multiple LLMs, tools, and custom scripts to achieve complex results.

## Powerful Workflow Examples

*   **[No-RAG vs RAG GIF](Doc_Resources/Media/Gifs/Search-Gif.gif)**

*   **Simple Assistant Workflow**

    ![Single Assistant Routing to Multiple LLMs](Doc_Resources/Media/Images/Wilmer-Assistant-Workflow-Example.jpg)

*   **Prompt Routing Example**

    ![Prompt Routing Example](Doc_Resources/Media/Images/Wilmer-Categorization-Workflow-Example.png)

*   **Group Chat to Different LLMs**

    ![Groupchat to Different LLMs](Doc_Resources/Media/Images/Wilmer-Groupchat-Workflow-Example.png)

*   **Example of a UX Workflow Where A User Asks for a Website**

    ![Oversimplified Example Coding Workflow](Doc_Resources/Media/Images/Wilmer-Simple-Coding-Workflow-Example.jpg)

## Documentation & Setup

*   **User Documentation:** [/Docs/_User_Documentation/README.md](Docs/_User_Documentation/README.md)
*   **Developer Documentation:** [/Docs/Developer_Docs/README.md](Docs/Developer_Docs/README.md)

### Quick-ish Setup Guides
*   **WilmerAI Getting Started:** [User Documents Setup Starting Guide](Docs/_User_Documentation/Setup/_Getting-Start_Wilmer-Api.md)
*   **Wilmer with Open WebUI:** [Open WebUI Setup Guide](Docs/_User_Documentation/Setup/Open-WebUI.md)
*   **Wilmer with SillyTavern:** [SillyTavern Setup Guide](Docs/_User_Documentation/Setup/SillyTavern.md)

### Youtube Videos

[![WilmerAI and Open WebUI Install on Fresh Windows 11 Desktop](https://img.youtube.com/vi/KDpbxHMXmTs/0.jpg)](https://www.youtube.com/watch?v=KDpbxHMXmTs "WilmerAI and Open WebUI Install on Fresh Windows 11 Desktop")

## Wilmer API Endpoints

*   **OpenAI Compatible:** v1/completions and chat/completions (*requires [Wilmer Prompt Template](Public/Configs/PromptTemplates/wilmerai.json)*)
*   **Ollama Compatible:** api/generate and api/chat (*requires [Wilmer Prompt Template](Public/Configs/PromptTemplates/wilmerai.json)*)

## Wilmer Connects To:
*   OpenAI Compatible v1/completions
*   OpenAI Compatible chat/completions
*   Ollama Compatible api/generate
*   Ollama Compatible api/chat
*   KoboldCpp Compatible api/v1/generate (*non-streaming generate*)
*   KoboldCpp Compatible /api/extra/generate/stream (*streaming generate*)

## Important Considerations

*   WilmerAI does not currently provide token usage tracking.  Monitor your LLM API usage.
*   The quality of WilmerAI is directly influenced by the LLMs and configurations used.

## Contact

For inquiries, contact WilmerAI.Project@gmail.com.