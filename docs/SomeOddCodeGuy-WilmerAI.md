# WilmerAI: Expert Contextual Routing for Language Models

**Tired of LLMs that lack context? WilmerAI revolutionizes how you interact with language models by providing advanced contextual routing and powerful workflow orchestration, enabling deeper understanding and more complex task execution.**

[Visit the original repository](https://github.com/SomeOddCodeGuy/WilmerAI)

## Key Features:

*   **Advanced Contextual Routing:** WilmerAI analyzes the entire conversation history to understand the true intent behind each request, allowing it to select the most appropriate specialized workflow.
*   **Node-Based Workflow Engine:**  Define complex, multi-step processes through JSON-based workflows. Each node can orchestrate different LLMs, call external tools, run custom scripts, and more.
*   **Multi-LLM & Multi-Tool Orchestration:** Seamlessly integrate various LLMs (OpenAI, Ollama, KoboldCpp) and tools within a single workflow for optimal task performance.
*   **Modular & Reusable Workflows:** Build and reuse workflows for common tasks, simplifying the creation of sophisticated AI agents.
*   **Stateful Conversation Memory:**  Maintain context across long conversations with a three-part memory system: a chronological summary, a rolling summary, and a vector database for Retrieval-Augmented Generation (RAG).
*   **Adaptable API Gateway:** Easily connect existing front-end applications and tools through OpenAI- and Ollama-compatible API endpoints.
*   **Flexible Backend Connectors:**  Connect to various LLM backends (OpenAI, Ollama, KoboldCpp) using a simple but powerful configuration system of Endpoints, API Types, and Presets.
*   **MCP Server Tool Integration:**  New experimental support for MCP server tool calling using MCPO, allowing tool use mid-workflow (thanks to iSevenDays).

## Quick Start:

WilmerAI is easy to set up.  Follow these steps to get started:

1.  **Install:** Run the provided script for Windows (`.bat`) or macOS (`.sh`), or manually install dependencies with `pip install -r requirements.txt` and run `python server.py`.
2.  **Choose a User Template:**  Select a pre-configured user from the `Public/Configs/Users` directory (e.g., `_example_simple_router_no_memory`).
3.  **Configure Endpoints:**  Update the endpoints in `Public/Configs/Endpoints` to match your LLM API keys and settings.
4.  **Set Current User:** Set your desired user in `Public/Configs/Users/_current-user.json`.
5.  **Run and Connect:**  Run Wilmer and connect to the OpenAI or Ollama compatible endpoints using your preferred front-end application.

## More Information:

*   **User Documentation:** `/Docs/_User_Documentation/README.md`
*   **Developer Documentation:** `/Docs/Developer_Docs/README.md`

## Important Notes:

*   **Token Usage:** WilmerAI does not track token usage. Monitor your LLM API dashboards for cost control.
*   **LLM Dependency:** The quality of WilmerAI's outputs depends on the quality of the connected LLMs and the prompt templates and configurations.
*   **Maintainer's Note:** This is a personal project under active development. Expect occasional bugs and updates. See the maintainer's note in the original README for more details.

## Contact

For feedback, requests, or inquiries, contact WilmerAI.Project@gmail.com.