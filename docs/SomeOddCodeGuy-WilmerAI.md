# WilmerAI: Supercharge Your Language Model Interactions with Advanced Routing & Orchestration

Tired of basic AI chatbots? **WilmerAI revolutionizes language model interactions by intelligently routing and orchestrating complex workflows.** [Check out the original repository](https://github.com/SomeOddCodeGuy/WilmerAI).

## Key Features

*   **Advanced Contextual Routing:** Directs user requests using sophisticated, context-aware logic, analyzing the entire conversation history to understand intent.

    *   **Prompt Routing:** Categorizes prompts to specialized workflows (e.g., "Coding," "Factual," "Creative").
    *   **In-Workflow Routing:** Enables conditional logic within workflows, allowing dynamic step selection.
*   **Node-Based Workflow Engine:** Processes requests through modular JSON-defined workflows, creating complex, chained-thought processes.
*   **Multi-LLM & Multi-Tool Orchestration:** Connect to different LLMs or tools within a single workflow. Use a local model for summarization and a cloud model for reasoning.
*   **Modular & Reusable Workflows:** Build and reuse self-contained workflows for common tasks, simplifying complex agent design.
*   **Stateful Conversation Memory:** Maintains context with a chronological summary file, a continuously updated rolling summary, and a vector database for RAG.
*   **Adaptable API Gateway:** Compatible with OpenAI and Ollama API endpoints, enabling integration with existing front-end applications.
*   **Flexible Backend Connectors:** Supports multiple LLM backends (OpenAI, Ollama, KoboldCpp) through a simple configuration system.
*   **MCP Server Tool Integration:** Experimental support for MCP server tool calling using MCPO, allowing tool use mid-workflow. Big thank you to [iSevenDays](https://github.com/iSevenDays)

## Why Choose WilmerAI?

WilmerAI goes beyond simple chatbots by enabling powerful features like multi-step processes, access to a wide variety of LLMs, and access to various tools. It allows for fine-grained control over the LLM experience.

## Setup & Usage

### Installation

1.  **Prerequisites:** Ensure you have Python installed.
2.  **Installation Options:**
    *   **Using Scripts:** Run the provided `.bat` (Windows) or `.sh` (macOS) files.
    *   **Manual Installation:** `pip install -r requirements.txt` then `python server.py`.
3.  **Configuration:**  All essential configurations are managed through JSON files in the "Public" folder.

### Connecting in SillyTavern

*   Connect using OpenAI Compatible v1/Completions (Requires [Wilmer Prompt Template](Public/Configs/PromptTemplates/wilmerai.json))
*   Connect using Ollama api/generate (Requires [Wilmer Prompt Template](Public/Configs/PromptTemplates/wilmerai.json))

### Connecting in Open WebUI

*   Connect as an Ollama instance.

## Resources

*   **User Documentation:** [/Docs/_User_Documentation/README.md](Docs/_User_Documentation/README.md)
*   **Developer Documentation:** [/Docs/Developer_Docs/README.md](Docs/Developer_Docs/README.md)

## Important Considerations

*   **Token Usage:** WilmerAI does not track or report accurate token usage. Monitor your LLM API dashboards.
*   **LLM Dependency:** WilmerAI's quality is directly influenced by the quality of the connected LLMs and the configuration.

## Contact

For questions and feedback: WilmerAI.Project@gmail.com