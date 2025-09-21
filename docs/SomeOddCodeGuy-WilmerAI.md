# WilmerAI: Advanced Semantic Prompt Routing and Task Orchestration

**Unlock the power of context and workflows for sophisticated AI interactions with WilmerAI.**

[![WilmerAI and Open WebUI Install on Fresh Windows 11 Desktop](https://img.youtube.com/vi/KDpbxHMXmTs/0.jpg)](https://www.youtube.com/watch?v=KDpbxHMXmTs "WilmerAI and Open WebUI Install on Fresh Windows 11 Desktop")

## Key Features

*   **Advanced Contextual Routing:** WilmerAI analyzes the *entire* conversation history to understand user intent, enabling intelligent routing to specialized workflows. It uses:
    *   **Prompt Routing:** Categorizes prompts to select the most appropriate workflow.
    *   **In-Workflow Routing:** Provides conditional "if/then" logic to dynamically choose the next step based on the output of a previous node.
*   **Core: Node-Based Workflow Engine:**  Build complex processes with workflows defined in JSON files, each containing a sequence of steps (nodes) for detailed task execution.
*   **Multi-LLM & Multi-Tool Orchestration:**  Connect each workflow node to diverse LLM endpoints or external tools, optimizing tasks by leveraging the best model for each step.
*   **Modular & Reusable Workflows:** Create and reuse workflows for common tasks, simplifying the design of intricate AI agents.
*   **Stateful Conversation Memory:**  Maintain context with a chronological summary, rolling summary, and a searchable vector database for Retrieval-Augmented Generation (RAG), all working together.
*   **Adaptable API Gateway:** Integrate with existing front-end applications seamlessly via OpenAI- and Ollama-compatible API endpoints.
*   **Flexible Backend Connectors:**  Connect to various LLM backends (OpenAI, Ollama, KoboldCpp) through an easy configuration system with **Endpoints**, **API Types**, and **Presets**.
*   **MCP Server Tool Integration using MCPO:** Experimental support for MCP server tool calling using MCPO. Big thanks to [iSevenDays](https://github.com/iSevenDays). More info can be found in the [ReadMe](Public/modules/README_MCP_TOOLS.md)

## Getting Started

### Setup Guides

*   **WilmerAI API:**  Follow the [User Documents Setup Starting Guide](Docs/_User_Documentation/Setup/_Getting-Start_Wilmer-Api.md) for a quick API setup.
*   **Wilmer with Open WebUI:** [Click here](Docs/_User_Documentation/Setup/Open-WebUI.md) for a written guide.
*   **Wilmer with SillyTavern:** [Click here](Docs/_User_Documentation/Setup/SillyTavern.md) for a setup guide.

##  Learn More

*   **User Documentation:** [/Docs/_User_Documentation/](Docs/_User_Documentation/README.md)
*   **Developer Documentation:** [/Docs/Developer_Docs/](Docs/Developer_Docs/README.md)

## Contact

Reach out with feedback and suggestions: WilmerAI.Project@gmail.com

[Original Repository](https://github.com/SomeOddCodeGuy/WilmerAI)