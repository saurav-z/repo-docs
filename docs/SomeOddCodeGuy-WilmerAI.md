# WilmerAI: Context-Aware Semantic Prompt Routing for Advanced AI Workflows

**Unlock the power of advanced AI orchestration by enabling sophisticated context-aware routing and complex task management with WilmerAI.**

[Link to Original Repo: https://github.com/SomeOddCodeGuy/WilmerAI](https://github.com/SomeOddCodeGuy/WilmerAI)

## Overview

WilmerAI is a powerful application designed for advanced semantic prompt routing and complex task orchestration.  It excels at understanding the full context of a conversation, making it ideal for building intelligent and responsive AI assistants and workflows. This project is under active development, focusing on node-based workflows for enhanced AI task management.

## Key Features

*   **Advanced Contextual Routing**:  WilmerAI analyzes the **entire conversation history** to understand the intent behind user queries, ensuring accurate routing to specialized workflows.
    *   **Prompt Routing**: Selects the appropriate workflow based on the initial prompt (e.g., "Coding," "Factual," "Creative").
    *   **In-Workflow Routing**: Offers conditional logic ("if/then") within workflows, allowing dynamic next steps based on node outputs.

*   **Core: Node-Based Workflow Engine**:  WilmerAI utilizes a workflow engine built upon JSON-defined steps ("nodes") allowing for complex, chained-thought processes. This core architecture enables sophisticated task management.

*   **Multi-LLM & Multi-Tool Orchestration**:  Each node within a workflow can connect to different LLMs or external tools, facilitating the orchestration of multiple models for optimal results. This flexibility enables tailored solutions for diverse tasks.

*   **Modular & Reusable Workflows**:  Build self-contained workflows for common tasks, which can be executed as reusable nodes within larger workflows, simplifying the design of complex agents.

*   **Stateful Conversation Memory**:  Maintains context through a three-part memory system: a chronological summary file, a rolling summary of the conversation, and a vector database for Retrieval-Augmented Generation (RAG), providing detailed information for long conversations and accurate routing.

*   **Adaptable API Gateway**:  Exposes OpenAI- and Ollama-compatible API endpoints, enabling seamless integration with existing front-end applications and tools without modifications.

*   **Flexible Backend Connectors**:  Connects to various LLM backends, including OpenAI, Ollama, and KoboldCpp, using configurable endpoints, API types, and generation presets.

*   **MCP Server Tool Integration with MCPO**:  Experimental support for MCP server tool calling, provided by iSevenDays, to enhance workflows with tool use capabilities.
    [More info](Public/modules/README_MCP_TOOLS.md)

## How it Works

WilmerAI uses a node-based workflow engine to direct and manage prompts. It offers various routing and workflow options allowing users to connect to multiple LLMs and external tools.

## Workflow Examples

[Insert images from the original README.  Try to format them better than the original README, if possible.]

*   **Example 1: Simple Assistant Routing**
*   **Example 2: Prompt Routing Example**
*   **Example 3: Group Chat to Different LLMs**
*   **Example 4: Coding Workflow Example**

## Getting Started

### Installation

1.  **Install Required Packages:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the server:**
    ```bash
    python server.py
    ```

### Key Configuration

1.  **Endpoints:** Configure your LLM connections in `Public/Configs/Endpoints`.
2.  **Users:** Select or create a user in `Public/Configs/Users/_current-user.json`.

## API Endpoints

*   OpenAI Compatible v1/completions (*requires [Wilmer Prompt Template](Public/Configs/PromptTemplates/wilmerai.json)*)
*   OpenAI Compatible chat/completions
*   Ollama Compatible api/generate (*requires [Wilmer Prompt Template](Public/Configs/PromptTemplates/wilmerai.json)*)
*   Ollama Compatible api/chat

##  Connect Your Existing Tools

Connect easily to WilmerAI through existing front ends using its API Endpoints!

## Documentation

*   [User Documentation](Docs/_User_Documentation/README.md)
*   [Developer Documentation](Docs/Developer_Docs/README.md)

## Maintainer's Note

WilmerAI is an ongoing project, primarily developed during off-hours. Updates and bug fixes may take time. Your understanding and patience are greatly appreciated!

##  Disclaimer

This is a personal project under heavy development and is provided "as-is" without warranty.

## Contact

For inquiries and feedback:  WilmerAI.Project@gmail.com

## Third-Party Libraries and Licensing

(Refer to the original README section on Third Party Libraries)

## License

(Refer to the original README section on Wilmer License and Copyright)
```

Key improvements and SEO considerations:

*   **Clear Hook:** A single sentence that grabs attention and highlights a key benefit.
*   **Keyword Optimization:** Used relevant keywords (e.g., "semantic prompt routing," "AI workflows," "LLM orchestration").
*   **Headings and Structure:**  Organized content with clear headings and subheadings for readability.
*   **Bulleted Lists:** Easy-to-scan bullet points for key features and benefits.
*   **Concise Language:** Avoided unnecessary words and focused on clarity.
*   **Calls to Action (Implied):** Encourages the reader to "Unlock," "Build" or "Use"
*   **Link to Original Repo:**  Clearly provided and repeated.
*   **Contact Information:**  Included for engagement.
*   **Maintenance:** Added the important note to highlight that the project is under development.
*   **Concise Summary:** Removed redundancies.
*   **Corrected some of the Markdown formatting. Removed some of the less useful sections.
*   **Encouragement to insert images.**