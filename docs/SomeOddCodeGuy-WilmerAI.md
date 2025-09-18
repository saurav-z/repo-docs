# WilmerAI: The Advanced Semantic Prompt Router & Workflow Engine

**Unlock the power of intelligent conversation routing and multi-LLM orchestration with WilmerAI, transforming your backend logic without front-end changes.** ([Back to original repo](https://github.com/SomeOddCodeGuy/WilmerAI))

## Key Features

*   **Advanced Contextual Routing:** WilmerAI's core feature, intelligently directs user requests based on the entire conversation history using:
    *   **Prompt Routing:** Analyzes the user's prompt to select the most suitable specialized workflow (e.g., "Coding," "Factual," "Creative").
    *   **In-Workflow Routing:** Provides conditional logic within workflows, enabling dynamic step selection based on node outputs.

*   **Core: Node-Based Workflow Engine:** Processes requests using JSON-defined workflows, allowing complex, chained-thought processes.

*   **Multi-LLM & Multi-Tool Orchestration:** Connects each workflow node to different LLMs or tools, enabling the best model for each step.

*   **Modular & Reusable Workflows:** Build self-contained workflows for common tasks and execute them as nodes within larger workflows for simplified agent design.

*   **Stateful Conversation Memory:** Provides context for long conversations via a three-part memory system: chronological summary, rolling summary, and a vector database.

*   **Adaptable API Gateway:** Compatible with OpenAI and Ollama APIs, allowing seamless integration with existing applications.

*   **Flexible Backend Connectors:** Connects to various LLM backends (OpenAI, Ollama, KoboldCpp) via an endpoint/API type/preset configuration system.

*   **MCP Server Tool Integration:** Experimental tool-calling support with MCPO.

## How WilmerAI Works

WilmerAI excels by going beyond simple keyword-based routing. It examines the full context of a conversation, enabling a deeper understanding of user intent. At its heart lies a node-based workflow engine, allowing you to define and orchestrate multi-step processes. Each node can leverage various LLMs, external tools, and custom scripts. This architecture allows for the creation of sophisticated backend logic while remaining compatible with your current front-end applications.

## Examples and Visualizations

*   **Simple Assistant Workflow**

    [Example of A Simple Assistant Workflow Using the Prompt Router](Docs/Examples/Images/Wilmer-Assistant-Workflow-Example.jpg)

*   **Prompt Routing Example**

    [Example of How Routing Might Be Used](Docs/Examples/Images/Wilmer-Categorization-Workflow-Example.png)

*   **Group Chat to Different LLMs**

    [Groupchat to Different LLMs](Docs/Examples/Images/Wilmer-Groupchat-Workflow-Example.png)

*   **UX Workflow Example**

    [Example of a UX Workflow Where A User Asks for a Website](Docs/Examples/Images/Wilmer-Simple-Coding-Workflow-Example.jpg)

## Quick Setup

### YouTube Videos

[![WilmerAI and Open WebUI Install on Fresh Windows 11 Desktop](https://img.youtube.com/vi/KDpbxHMXmTs/0.jpg)](https://www.youtube.com/watch?v=KDpbxHMXmTs "WilmerAI and Open WebUI Install on Fresh Windows 11 Desktop")

### Guides

*   **Getting Started with the API:** [User Documents Setup Starting Guide](Docs/_User_Documentation/Setup/_Getting-Start_Wilmer-Api.md)
*   **Setting up with Open WebUI:** [WilmerAI with Open WebUI](Docs/_User_Documentation/Setup/Open-WebUI.md)
*   **Setting up with SillyTavern:** [WilmerAI with SillyTavern](Docs/_User_Documentation/Setup/SillyTavern.md)

## Important Notes

*   **Token Usage:**  WilmerAI does not track or report token usage. Monitor your LLM API dashboards.
*   **LLM Dependency:** WilmerAI's quality depends heavily on the connected LLMs and your prompt design.

## Contact

WilmerAI.Project@gmail.com

## Third-Party Libraries

WilmerAI utilizes the following libraries: Flask, requests, scikit-learn, urllib3, jinja2, and pillow. Further details, including licensing information, are available in the ThirdParty-Licenses folder.

## License

Licensed under the GNU General Public License v3.