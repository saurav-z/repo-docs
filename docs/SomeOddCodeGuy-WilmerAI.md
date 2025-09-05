# WilmerAI: Expert Semantic Prompt Routing and Orchestration

**Unlock advanced AI capabilities with WilmerAI, a node-based workflow engine that understands context and orchestrates multiple LLMs to deliver powerful, multi-step responses.  [View the original repository](https://github.com/SomeOddCodeGuy/WilmerAI)**

## Key Features

*   **Advanced Contextual Routing:**
    *   Analyzes entire conversation history for precise intent understanding, surpassing single-keyword routing.
    *   Includes a **Prompt Router** that selects specialized workflows and **In-Workflow Routing** for dynamic decision-making, all based on the conversation's context.

*   **Node-Based Workflow Engine:**
    *   Core of WilmerAI, processing requests through JSON-defined workflows (sequences of steps).
    *   Each "node" executes a specific task, with outputs feeding into subsequent nodes, enabling intricate, chained processes.

*   **Multi-LLM & Multi-Tool Orchestration:**
    *   Each workflow node connects to a different LLM endpoint or external tool.
    *   Optimize tasks by assigning specialized models for each step, leveraging the strengths of various LLMs.

*   **Modular & Reusable Workflows:**
    *   Construct self-contained workflows for common tasks.
    *   Integrate workflows as nodes within larger workflows, simplifying complex agent design.

*   **Stateful Conversation Memory:**
    *   Employs a three-part memory system: chronological summary, rolling chat summary, and searchable vector database for effective context management.

*   **Adaptable API Gateway:**
    *   Exposes OpenAI and Ollama-compatible API endpoints for easy integration with existing front-ends.

*   **Flexible Backend Connectors:**
    *   Connects to various LLM backends, including OpenAI, Ollama, and KoboldCpp, through customizable endpoints, API types, and presets.

*   **MCP Server Tool Integration using MCPO:** Supports MCPO server tool calling, for use with iSevenDays's MCP tool-calling feature, allowing tool use mid-workflow.

---

## Power of Workflows:

*   **Dynamic LLM Control:** Orchestrate various LLMs within a single call, as seen with the ability to call the Offline Wikipedia API
*   **Iterative Enhancement:** Refine responses through iterative LLM interactions for improved performance.
*   **Distributed LLMs:** Leverage multiple machines and APIs for powerful and scalable workflows.

## Getting Started

### Quick Setup

1.  **Install Dependencies:** Ensure Python is installed and install dependencies using `pip install -r requirements.txt`.
2.  **Run the Server:** Execute the server script using `python server.py` or the provided `.bat` or `.sh` files.
3.  **Configure Endpoints:** Set up your LLM endpoints in the `Public/Configs/Endpoints` directory.
4.  **Choose a User:** Select a pre-made user configuration from `Public/Configs/Users/_current-user.json` or run with the `--User` argument.

For detailed instructions, see the [User Documentation](/Docs/_User_Documentation/README.md) and [Developer Documentation](/Docs/Developer_Docs/README.md).

---

## Maintainer's Note
  *   As the project is undergoing active development, please be aware that pull requests modifying the Middleware modules are not accepted at this time. However, any changes to the iSevenDays's new MCP tool-calling feature, or adding new custom users or prompt templates within the Public directory are still welcome.

## Contact

For questions, feedback, or collaboration, please reach out to:

WilmerAI.Project@gmail.com

---

**[View the original repository](https://github.com/SomeOddCodeGuy/WilmerAI)**