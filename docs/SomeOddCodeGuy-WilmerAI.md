# WilmerAI: The Semantic Routing Engine for LLM Orchestration

**Unlock the power of advanced context-aware workflows to revolutionize how you interact with language models.** Explore the original repository here: [SomeOddCodeGuy/WilmerAI](https://github.com/SomeOddCodeGuy/WilmerAI)

## Key Features

*   **Advanced Contextual Routing:**  WilmerAI analyzes the *entire* conversation history to understand user intent, not just the most recent message, for superior routing.
    *   **Prompt Routing:** Selects the appropriate specialized workflow (e.g., "Coding," "Factual," "Creative") at the start of a conversation.
    *   **In-Workflow Routing:** Provides conditional "if/then" logic for dynamic decision-making within workflows.

*   **Core: Node-Based Workflow Engine:** Processes requests through JSON-defined workflows, enabling complex, multi-step processes.

*   **Multi-LLM & Multi-Tool Orchestration:** Directs each node to different LLM endpoints or tools for optimized task execution.

*   **Modular & Reusable Workflows:**  Create self-contained workflows for common tasks and easily incorporate them into larger workflows.

*   **Stateful Conversation Memory:** Utilizes a three-part memory system (chronological summary, rolling summary, and vector database) for long-term context.

*   **Adaptable API Gateway:** Exposes OpenAI- and Ollama-compatible endpoints for easy integration with existing front-end tools.

*   **Flexible Backend Connectors:** Connects to various LLM backends (OpenAI, Ollama, KoboldCpp) via configurable endpoints, API types, and presets.

*   **MCP Server Tool Integration using MCPO:** Experimental support for calling server tools using MCPO. Big thank you to [iSevenDays](https://github.com/iSevenDays) for the amazing work on this feature. More info can be found in the [ReadMe](Public/modules/README_MCP_TOOLS.md)

## Why Choose WilmerAI?

*   **Beyond Simple Routing:** WilmerAI goes beyond keyword-based categorization by leveraging complete conversation history.
*   **Granular Control:** Users have maximum control over the paths LLMs take.
*   **Multiple LLMs Working Together:** Orchestrate multiple LLMs within a single workflow.
*   **Customizable Categorization:** Built with user-defined workflows, with as many nodes and LLMs involved as the user wants, to break down the conversation.
*   **Semi-Autonomous Workflows:** Wilmer became focused on semi-autonomous Workflows, giving the user granular control of the path the LLMs take, and allow maximum use of the user's own domain knowledge and experience.

## Visual Examples

*   [Simple Assistant Workflow](Doc_Resources/Media/Images/Wilmer-Assistant-Workflow-Example.jpg)
*   [Prompt Routing Example](Doc_Resources/Media/Images/Wilmer-Categorization-Workflow-Example.png)
*   [Group Chat to Different LLMs](Doc_Resources/Media/Images/Wilmer-Groupchat-Workflow-Example.png)
*   [Coding Workflow Example](Doc_Resources/Media/Images/Wilmer-Simple-Coding-Workflow-Example.jpg)

## Getting Started

### Quick Setup Guides

*   [User Documents Setup Starting Guide](Docs/_User_Documentation/Setup/_Getting-Start_Wilmer-Api.md)
*   [Setting up Wilmer with Open WebUI](Docs/_User_Documentation/Setup/Open-WebUI.md)
*   [Setting up Wilmer with SillyTavern](Docs/_User_Documentation/Setup/SillyTavern.md)

### Video Tutorial

[![WilmerAI and Open WebUI Install on Fresh Windows 11 Desktop](https://img.youtube.com/vi/KDpbxHMXmTs/0.jpg)](https://www.youtube.com/watch?v=KDpbxHMXmTs "WilmerAI and Open WebUI Install on Fresh Windows 11 Desktop")

## API Endpoints

*   **OpenAI Compatible:** v1/completions (*requires [Wilmer Prompt Template](Public/Configs/PromptTemplates/wilmerai.json)*), chat/completions
*   **Ollama Compatible:** api/generate (*requires [Wilmer Prompt Template](Public/Configs/PromptTemplates/wilmerai.json)*), api/chat

## Backend Connections

*   OpenAI (v1/completions, chat/completions)
*   Ollama (api/generate, api/chat)
*   KoboldCpp (api/v1/generate - non-streaming, /api/extra/generate/stream - streaming)

## Important Notes

*   **Token Usage:** WilmerAI does *not* track or report token usage. Monitor your LLM API dashboards.
*   **LLM Dependency:** The quality of WilmerAI is directly tied to the quality of the connected LLMs, presets, and prompt templates.

## Contact

For feedback, requests, or inquiries, contact: WilmerAI.Project@gmail.com

## Documentation

*   [User Documentation](Docs/_User_Documentation/README.md)
*   [Developer Documentation](Docs/Developer_Docs/README.md)

## Third Party Libraries

WilmerAI utilizes the following libraries.  See [ThirdParty-Licenses/README.md](ThirdParty-Licenses/README.md) for licensing details.

*   Flask
*   requests
*   scikit-learn
*   urllib3
*   jinja2
*   pillow

## License

WilmerAI is licensed under the GNU General Public License v3.  See the [LICENSE](LICENSE) file for details.