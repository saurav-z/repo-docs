# WilmerAI: Expert Contextual Routing for LLM Applications

**Unlock advanced semantic understanding and orchestration capabilities for your LLM applications with WilmerAI: Where conversations are truly understood.** ([Original Repo](https://github.com/SomeOddCodeGuy/WilmerAI))

## Key Features

*   **Advanced Contextual Routing:**
    *   **Prompt Routing:** Analyzes the entire conversation history to select the most suitable specialized workflow (e.g., "Coding," "Factual," "Creative").
    *   **In-Workflow Routing:** Implements conditional logic for dynamic decision-making based on node outputs.
    *   **Context-Aware:** Makes routing decisions based on the entire conversation history, not just the latest message.

*   **Core: Node-Based Workflow Engine:**
    *   **Workflow Definition:** Processes requests through JSON-defined workflows, each a sequence of steps (nodes).
    *   **Modular Design:** Nodes perform distinct tasks, with outputs passed as input for chained, complex processes.

*   **Multi-LLM & Multi-Tool Orchestration:**
    *   **LLM Agnostic:** Connects to diverse LLM endpoints and external tools within each node, optimizing for specific tasks.
    *   **Task Specialization:** Orchestrates the best model for each step—e.g., a local model for summarization, a cloud model for reasoning.

*   **Modular & Reusable Workflows:**
    *   **Workflow Composability:** Builds self-contained workflows for common tasks that can be reused as nodes in larger workflows.
    *   **Simplified Design:** Simplifies complex agent design through building blocks of workflows.

*   **Stateful Conversation Memory:**
    *   **Comprehensive Memory:** Maintains a three-part memory system to facilitate long conversations and accurate routing.
        *   Chronological Summary File
        *   Rolling Summary
        *   Vector Database for RAG

*   **Adaptable API Gateway:**
    *   **API Compatibility:** Exposes OpenAI- and Ollama-compatible API endpoints, compatible with existing frontends.

*   **Flexible Backend Connectors:**
    *   **LLM Agnostic:** Connects to LLM backends, including OpenAI, Ollama, and KoboldCpp, via a simple and configurable system of Endpoints, API Types and Presets.

*   **MCP Server Tool Integration using MCPO:**
    *   **Tool Calling:** Supports MCP server tool-calling mid-workflow, allowing tool usage in between workflow nodes.

## Why Choose WilmerAI?

WilmerAI provides a robust solution for developers seeking advanced semantic prompt routing and sophisticated task orchestration. Its design allows users to have granular control of the path the LLMs take. It allows multiple LLMs to work together within a single workflow, giving the user maximum use of domain knowledge and experience.

## Getting Started

### Installation

1.  **Install Dependencies:** `pip install -r requirements.txt`
2.  **Run the Server:** `python server.py`

For convenience, you can utilize the included `.bat` (Windows) and `.sh` (macOS) scripts for simplified setup.
Use the command line arguments to customize the install location, as documented in the original README.

### Configuration

All configurations reside in the "Public" folder, containing essential JSON files. Configure endpoints and users within the `Endpoints` and `Users` directories, including specifying your chosen LLM API keys. For detailed setup instructions, review the provided documentation.

## Connect Your Front-End

WilmerAI is designed to connect to a wide range of LLM frontend interfaces such as SillyTavern and Open WebUI. Both tools are capable of using standard OpenAI chat completion and Ollama endpoints.

## Resources

*   **User Documentation:** \[Docs/_User\_Documentation/README.md]
*   **Developer Documentation:** \[Docs/Developer\_Docs/README.md]

## Contact

Reach out with feedback or questions: WilmerAI.Project@gmail.com

## License and Copyright

WilmerAI is licensed under the GNU General Public License v3.