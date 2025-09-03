# WilmerAI: Revolutionizing LLM Inference with Contextual Routing

**Unlock advanced semantic prompt routing and orchestrate complex tasks with WilmerAI, your gateway to next-generation language model interactions. [Visit the GitHub Repository](https://github.com/SomeOddCodeGuy/WilmerAI)**

WilmerAI is a powerful application designed to transform how you interact with language models. It excels in advanced semantic prompt routing and complex task orchestration, offering a more intelligent and versatile approach compared to traditional LLM applications.

## Key Features

*   **Advanced Contextual Routing:**
    *   Analyzes the entire conversation history to understand the *true* intent behind a query.
    *   Routes prompts to specialized workflows (e.g., coding, factual, creative) based on context, not just keywords.
    *   Provides conditional "if/then" logic within workflows, allowing dynamic process control.

*   **Node-Based Workflow Engine:**
    *   At the heart of WilmerAI, enabling structured and efficient processing of requests.
    *   Uses JSON-defined workflows, composed of interconnected "nodes" that each perform a specific task.
    *   Orchestrates multiple LLMs, external tools, and custom scripts within a single workflow, with outputs passed between nodes.

*   **Multi-LLM & Multi-Tool Orchestration:**
    *   Each node can call on different LLM endpoints or execute external tools.
    *   Facilitates the use of the *best* model for each part of a task, boosting overall results.
    *   Allows you to leverage all of your resources for any given single prompt.

*   **Modular & Reusable Workflows:**
    *   Create self-contained workflows for common tasks.
    *   Execute workflows as reusable nodes in larger, more complex workflows.
    *   Simplify the design of complex agent applications.

*   **Stateful Conversation Memory:**
    *   Keeps track of conversations and improves contextual awareness and routing accuracy.
    *   Chronological summary file, a continuously updated "rolling summary", and vector database for RAG.

*   **Adaptable API Gateway:**
    *   Exposes OpenAI and Ollama-compatible API endpoints.
    *   Connects seamlessly with existing front-end applications and tools.

*   **Flexible Backend Connectors:**
    *   Connects to various LLM backends (OpenAI, Ollama, KoboldCpp).
    *   Uses a simple but powerful configuration system of Endpoints, API Types, and Presets.

*   **MCP Server Tool Integration using MCPO:**
    *   Experimental support for MCP server tool calling using MCPO.
    *   Enables tool use mid-workflow, expanding task capabilities.
    *   Thanks to [iSevenDays](https://github.com/iSevenDays) for their contributions.

## Getting Started

WilmerAI requires Python and offers easy setup through:

*   **Option 1: Using Provided Scripts:** Run .bat (Windows) or .sh (macOS) files.
*   **Option 2: Manual Installation:** Install dependencies with `pip install -r requirements.txt` and run `python server.py`.

### Quick-ish Setup

1.  **Configure Endpoints:** Set up LLM API connections in the "Public/Configs/Endpoints" directory.
2.  **Choose a User:** Define your settings in `Public/Configs/Users/_current-user.json`.

Refer to the full [README](https://github.com/SomeOddCodeGuy/WilmerAI) for detailed instructions.

## Example Applications

*   Semantic Routing
*   Workflow Orchestration
*   Multi-LLM Interactions

## Contact

For feedback, requests, or support, reach out to WilmerAI.Project@gmail.com.

## License & Third-Party Libraries

WilmerAI is licensed under the GNU General Public License. It utilizes several third-party libraries listed in the original README and licensed accordingly.