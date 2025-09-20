# WilmerAI: Revolutionizing LLM Inference with Contextual Routing

**WilmerAI empowers you to build advanced, multi-step workflows for large language models (LLMs) by understanding conversation context and orchestrating complex tasks.** 

[Go to the original repo!](https://github.com/SomeOddCodeGuy/WilmerAI)

## Key Features

*   **Contextual Prompt Routing:** Analyze entire conversation histories for precise intent understanding, routing requests to specialized workflows.
*   **Node-Based Workflow Engine:** Build complex LLM interactions using JSON-defined workflows, each step (node) capable of orchestrating LLMs, tools, and custom scripts.
*   **Multi-LLM & Multi-Tool Orchestration:** Seamlessly integrate diverse LLMs and external tools within a single workflow, optimizing task performance.
*   **Modular & Reusable Workflows:** Create self-contained workflows for common tasks, facilitating streamlined design and code reuse.
*   **Stateful Conversation Memory:** Leverage a three-part memory system (chronological summary, rolling summary, and vector database) for enhanced context and RAG.
*   **Adaptable API Gateway:** Connect to existing front-end applications with OpenAI- and Ollama-compatible API endpoints.
*   **Flexible Backend Connectors:** Easily connect to various LLM backends, including OpenAI, Ollama, and KoboldCpp.
*   **MCP Server Tool Integration:** Support for MCP server tool calling using MCPO.

## What is WilmerAI?

WilmerAI is a powerful application designed for advanced semantic prompt routing and complex task orchestration. At its core, WilmerAI is a **node-based workflow engine**. It allows users to create sophisticated chains of steps, or "nodes," defined in JSON files. Each node can orchestrate different LLMs, call external tools, run custom scripts, call other workflows, and many other things.

## Getting Started

### User Documentation

*   [User Documentation](Docs/_User_Documentation/README.md)

### Developer Documentation

*   [Developer Documentation](Docs/Developer_Docs/README.md)

### Quick Setup Guides

*   [WilmerAI Setup Starting Guide](Docs/_User_Documentation/Setup/_Getting-Start_Wilmer-Api.md)
*   [Wilmer with Open WebUI](Docs/_User_Documentation/Setup/Open-WebUI.md)
*   [Wilmer With SillyTavern](Docs/_User_Documentation/Setup/SillyTavern.md)

## Examples

### Semi-Autonomous Workflows

![No-RAG vs RAG](Docs/Gifs/Search-Gif.gif)

### Assistant Routing

![Single Assistant Routing to Multiple LLMs](Docs/Examples/Images/Wilmer-Assistant-Workflow-Example.jpg)

### Prompt Routing

![Prompt Routing Example](Docs/Examples/Images/Wilmer-Categorization-Workflow-Example.png)

### Group Chat

![Groupchat to Different LLMs](Docs/Examples/Images/Wilmer-Groupchat-Workflow-Example.png)

### UX Workflow

![Oversimplified Example Coding Workflow](Docs/Examples/Images/Wilmer-Simple-Coding-Workflow-Example.jpg)

## Contact

For feedback, requests, or just to say hi, you can reach the maintainer at: WilmerAI.Project@gmail.com

## License

WilmerAI is licensed under the GNU General Public License v3. See the LICENSE file for details.