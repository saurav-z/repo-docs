# WilmerAI: Expert Semantic Prompt Routing and Workflow Orchestration for LLMs

Tired of basic AI chatbots? **WilmerAI revolutionizes how you interact with language models by intelligently routing your requests and orchestrating complex, multi-step workflows.** [Explore WilmerAI on GitHub](https://github.com/SomeOddCodeGuy/WilmerAI).

## Key Features:

*   **Context-Aware Routing:**
    *   **Prompt Routing:** Analyzes the *entire* conversation history to understand the user's true intent and direct the query to the appropriate specialized workflow (e.g., "Coding," "Factual," "Creative").
    *   **In-Workflow Routing:** Dynamically chooses the next step based on the output of a previous node, creating adaptable, efficient processes.
*   **Node-Based Workflow Engine:**
    *   Utilizes JSON-defined workflows, consisting of steps ("nodes"), to orchestrate complex logic. Each node can trigger different LLMs, tools, scripts, and more.
*   **Multi-LLM & Tool Integration:**
    *   Effortlessly combines different LLMs and external tools (e.g., search, APIs, custom scripts) within a single workflow for optimal results.
*   **Modular and Reusable Workflows:**
    *   Create self-contained workflows for common tasks and easily integrate them into larger, more complex workflows.
*   **Conversational Memory:**
    *   Maintains context with a chronological summary, a continuously updated rolling summary, and a vector database for Retrieval-Augmented Generation (RAG).
*   **API Gateway:**
    *   Exposes OpenAI- and Ollama-compatible API endpoints for seamless integration with existing applications.
*   **Backend Connectors:**
    *   Connects to various LLM backends (OpenAI, Ollama, KoboldCpp) using a configurable system of Endpoints, API Types, and Presets.
*   **MCPO Tool Calling:** Experimental support for MCP server tool calling, allowing tool use mid-workflow. Big thank you
    to [iSevenDays](https://github.com/iSevenDays)

---

## Key Benefits:

*   **Enhanced Accuracy:** Understand and respond to user prompts with greater precision.
*   **Complex Task Automation:** Automate intricate processes beyond simple question-answering.
*   **Flexible Integration:** Easily integrate with your existing front-end tools.
*   **Modular Design:** Simplify the creation of powerful and adaptable AI solutions.

---

## Getting Started

WilmerAI is a powerful but evolving project. Please be aware that there may be bugs.

1.  **Installation:** Follow the instructions in the original README to set up the project.
2.  **Configuration:** Customize WilmerAI using JSON configuration files located in the `Public` folder.
3.  **User Setup:** Select a pre-made user configuration from `Public/Configs/Users` and configure the endpoints in `Public/Configs/Endpoints`.
4.  **API Connection:** Connect your favorite LLM application (SillyTavern, Open WebUI, etc.) to Wilmer's API endpoints.

---

## Example Applications:

*   **Advanced Chatbots:** Build smarter, more conversational chatbots.
*   **Task Automation:** Automate complex tasks involving multiple LLMs and tools.
*   **Custom AI Assistants:** Create specialized AI assistants for various needs.

---

## Important Notes

WilmerAI uses LLMs to do the bulk of the heavy lifting. Please be sure to keep track of how many tokens you are using via any dashboard provided to you by your LLM APIs, especially early on as you get used to this software.

---

## Further Exploration

*   **User Documentation:** [/Docs/\_User\_Documentation/README.md](Docs/_User_Documentation/README.md)
*   **Developer Documentation:** [/Docs/Developer\_Docs/README.md](Docs/Developer_Docs/README.md)

---