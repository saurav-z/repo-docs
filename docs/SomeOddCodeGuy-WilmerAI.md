# WilmerAI: Expert Semantic Prompt Routing & Workflow Orchestration

**Unlock advanced, context-aware AI interactions with WilmerAI, a node-based workflow engine for sophisticated prompt routing and multi-LLM orchestration.** ([Original Repo](https://github.com/SomeOddCodeGuy/WilmerAI))

## Key Features:

*   **Advanced Contextual Routing:** Understands the full context of a conversation, not just the last message, for superior routing and intent recognition.
*   **Node-Based Workflow Engine:** Powers complex, multi-step processes using JSON-defined workflows.  Each node can trigger LLMs, external tools, scripts, and other workflows.
*   **Multi-LLM & Multi-Tool Orchestration:** Orchestrates the optimal LLM and tools for each step of a task, enabling a customized and powerful AI solution.
*   **Modular & Reusable Workflows:** Build and reuse self-contained workflows for common tasks to streamline complex agent design.
*   **Stateful Conversation Memory:**  Leverages a multi-part memory system for effective long-conversation context (chronological summary, rolling summary, and vector database).
*   **Adaptable API Gateway:** Provides OpenAI and Ollama-compatible API endpoints for easy integration with existing front-end applications.
*   **Flexible Backend Connectors:** Connects to various LLM backends, including OpenAI, Ollama, and KoboldCpp, with a simple configuration.
*   **Tool Integration:** Support for MCP Server Tool Calling using MCPO, allowing tool use mid-workflow. (Big thank you to [iSevenDays](https://github.com/iSevenDays))

## Why Use WilmerAI?

WilmerAI goes beyond simple keyword-based routing, providing a powerful framework for orchestrating complex AI tasks. It enables you to:

*   **Build semi-autonomous AI assistants** that excel at iterative problem-solving.
*   **Leverage distributed LLMs** by connecting to a wide range of language models and services.
*   **Customize your AI interactions** with fully flexible workflows that go beyond what's currently available.

## Example Workflows - Visualized

*(Click on images to view the animated GIFs)*

*   **No-RAG vs RAG:**
    ![No-RAG vs RAG](Docs/Gifs/Search-Gif.gif)
*   **Simple Assistant Workflow:**
    ![Single Assistant Routing to Multiple LLMs](Docs/Examples/Images/Wilmer-Assistant-Workflow-Example.jpg)
*   **Prompt Routing Example:**
    ![Prompt Routing Example](Docs/Examples/Images/Wilmer-Categorization-Workflow-Example.png)
*   **Group Chat to Different LLMs:**
    ![Groupchat to Different LLMs](Docs/Examples/Images/Wilmer-Groupchat-Workflow-Example.png)
*   **Coding Workflow Example:**
    ![Oversimplified Example Coding Workflow](Docs/Examples/Images/Wilmer-Simple-Coding-Workflow-Example.jpg)

## Getting Started

### 1. Installation

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the Server:**
    ```bash
    python server.py
    ```
    Or, use the provided `.bat` (Windows) or `.sh` (macOS) scripts.

### 2. Configuration:

1.  **Endpoints:** Configure your LLM API endpoints in the `Public/Configs/Endpoints` directory.  Example endpoints are provided in the `_example-endpoints` folder.
2.  **User:** Select the user profile you want to use in `Public/Configs/Users/_current-user.json`. Choose from provided templates for easy setup.

**Find pre-made user configurations in `Public/Configs/Users/` to quickly get started.**

## Connecting to WilmerAI

WilmerAI exposes OpenAI and Ollama-compatible APIs:

*   OpenAI v1/completions (*requires [Wilmer Prompt Template](Public/Configs/PromptTemplates/wilmerai.json)*)
*   OpenAI chat/completions
*   Ollama api/generate (*requires [Wilmer Prompt Template](Public/Configs/PromptTemplates/wilmerai.json)*)
*   Ollama api/chat

WilmerAI can connect to:

*   OpenAI Compatible v1/completions
*   OpenAI Compatible chat/completions
*   Ollama Compatible api/generate
*   Ollama Compatible api/chat
*   KoboldCpp Compatible api/v1/generate (*non-streaming generate*)
*   KoboldCpp Compatible /api/extra/generate/stream (*streaming generate*)

### Connecting in SillyTavern

*   **Text Completion:** Connect as OpenAI Compatible v1/Completions or Ollama api/generate (use WilmerAI prompt template).
*   **Chat Completions (Not Recommended):** Configure in SillyTavern and in your Wilmer user settings.

### Connecting in Open WebUI

Connect to Wilmer as if it were an Ollama instance.

## Important Notes

*   **Token Usage:**  WilmerAI does not track or report token usage. Monitor this yourself through your LLM API dashboards.
*   **LLM Dependency:** The quality of WilmerAI's outputs depends heavily on the quality of the connected LLMs and the prompt template you use.
*   **Current Development:**  The project is under active development.  Refer to the maintainer's notes in the original README for current development and future plans.

## Contact

*   WilmerAI.Project@gmail.com

## Additional Information

*   **User Documentation:** /Docs/\_User\_Documentation/README.md
*   **Developer Documentation:** /Docs/Developer\_Docs/README.md
*   **Third Party Libraries:**  Detailed license information is in the README of the ThirdParty-Licenses folder.
*   **License:**  GNU General Public License v3.