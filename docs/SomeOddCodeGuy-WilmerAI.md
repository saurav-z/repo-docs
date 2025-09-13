# WilmerAI: Revolutionizing Semantic Prompt Routing and Task Orchestration

**Unleash the power of context with WilmerAI, a node-based workflow engine designed to intelligently route and orchestrate LLMs for advanced conversational AI.**

[Go to the original repository](https://github.com/SomeOddCodeGuy/WilmerAI)

## Key Features

*   **Advanced Contextual Routing:**
    *   Analyzes the **entire conversation history** to understand the true intent behind user queries, moving beyond single-keyword categorization.
    *   **Prompt Routing:** Directs user requests to appropriate specialized workflows (e.g., "Coding," "Factual," "Creative").
    *   **In-Workflow Routing:** Offers conditional logic within workflows, enabling dynamic decision-making based on node outputs.

*   **Core: Node-Based Workflow Engine:**
    *   Processes requests through workflows defined in **JSON files**, creating sequences of steps or "nodes."
    *   Nodes can be configured to connect to different LLMs or execute external tools.
    *   This is what allows for complex, chained-thought processes.

*   **Multi-LLM & Multi-Tool Orchestration:**
    *   Orchestrates different LLMs and tools within a single workflow.
    *   Allows you to select the best model for each task stage, such as using a small, fast local model for summarization and a large, powerful cloud model for the final reasoning.

*   **Modular & Reusable Workflows:**
    *   Build self-contained workflows for common tasks and integrate them into larger workflows.
    *   Simplifies the design of complex agents.

*   **Stateful Conversation Memory:**
    *   Includes a 3-part memory system to provide context for long conversations and ensure accurate routing.
        *   Chronological summary file
        *   Continuously updated "rolling summary" of the entire chat.
        *   Searchable vector database for Retrieval-Augmented Generation (RAG).

*   **Adaptable API Gateway:**
    *   Exposes OpenAI- and Ollama-compatible API endpoints.
    *   Connects seamlessly with existing front-end applications and tools.

*   **Flexible Backend Connectors:**
    *   Connects to various LLM backends (OpenAI, Ollama, KoboldCpp) using a simple configuration of Endpoints, API Types, and Presets.

*   **MCP Server Tool Integration using MCPO:**
    *   Experimental support for MCP server tool calling via MCPO.

## Why Choose WilmerAI?

WilmerAI provides a powerful and flexible framework for building sophisticated conversational AI applications. By leveraging its node-based workflow engine, you can create intelligent agents that understand context, orchestrate multiple LLMs, and integrate with external tools. It allows for:

*   **Enhanced Contextual Understanding:** Go beyond keyword-based routing by analyzing the entire conversation.
*   **Customizable Workflows:** Define complex workflows to achieve specific goals.
*   **Improved Performance:** Leverage multiple LLMs and tools for enhanced results.
*   **Seamless Integration:** Connect to existing front-end applications.
*   **Flexibility:** Adapt workflows and routing mechanisms to match your user's preferences.

## Getting Started

### Step 1: Installing the Program

1.  **Prerequisites:** Ensure you have Python installed (tested with 3.10 and 3.12).
2.  **Installation Options:**
    *   **Scripts (Recommended):** Run the provided `.bat` (Windows) or `.sh` (macOS) file, which sets up a virtual environment and installs dependencies from `requirements.txt`.
    *   **Manual Installation:**
        1.  Install dependencies: `pip install -r requirements.txt`
        2.  Run the server: `python server.py`
3.  Script arguments for .bat, .sh and .py files:
    *   `--ConfigDirectory`: Directory where your config files can be found. By default, this is the "Public" folder within the Wilmer root.
    *   `--LoggingDirectory`: The directory where file logs, if enabled, are stored. Be default file logging is turned OFF, and in the event that they are enabled in the user json, they default to going to the "logs" folder in the Wilmer root
    *   `--User`: The user that you want to run under.

### Step 2:  Quick Start with Pre-made Users

1.  **Choose a Template User:** Select a pre-configured user from `Public/Configs/Users`. Example user categories:
    *   `\_example\_simple\_router\_no\_memory`
    *   `\_example\_general\_workflow`
    *   `\_example\_coding\_workflow`
    *   `\_example\_wikipedia\_multi\_step\_workflow`
    *   `\_example\_assistant\_with\_vector\_memory`
    *   `\_example\_game\_bot\_with\_file\_memory`
2.  **Update Endpoints:** Configure your LLM endpoints in `Public/Configs/Endpoints`.
3.  **Set Current User:**  Update `Public/Configs/Users/_current-user.json` with the name of your chosen user.
4.  **User Configuration:** Customize the user's settings within the JSON file.

Run Wilmer, connect, and start exploring!

## API Endpoints

WilmerAI exposes the following API endpoints:

*   OpenAI Compatible `/v1/completions` (requires Wilmer Prompt Template)
*   OpenAI Compatible `/chat/completions`
*   Ollama Compatible `/api/generate` (requires Wilmer Prompt Template)
*   Ollama Compatible `/api/chat`

WilmerAI can connect to:

*   OpenAI Compatible `/v1/completions`
*   OpenAI Compatible `/chat/completions`
*   Ollama Compatible `/api/generate`
*   Ollama Compatible `/api/chat`
*   KoboldCpp Compatible `/api/v1/generate` (non-streaming)
*   KoboldCpp Compatible `/api/extra/generate/stream` (streaming)

## Connecting to WilmerAI (Examples)

**SillyTavern:**

*   Use OpenAI Compatible v1/Completions or Ollama api/generate.
*   For Text Completion, use a WilmerAI-specific Prompt Template.
*   For Chat Completion, configure settings.

**Open WebUI:**

*   Connect as an Ollama instance.

## Important Notes

*   **Token Usage:** Monitor token usage within your LLM API dashboards, as WilmerAI does not track token consumption.
*   **LLM Quality:**  The quality of your connected LLMs directly impacts WilmerAI's performance.
*   **Documentation:** Comprehensive user documentation available at `/Docs/_User_Documentation/README.md` and Developer documentation at `/Docs/Developer_Docs/README.md`

## Contact

WilmerAI.Project@gmail.com

---

**(Remainder of the original README content, including third-party libraries, license, and maintainer's notes, can remain unchanged.)**