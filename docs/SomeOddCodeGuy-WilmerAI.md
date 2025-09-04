# WilmerAI: The Advanced Semantic Prompt Router and Workflow Engine

**Unlock the power of context and multi-LLM orchestration with WilmerAI, a cutting-edge application for advanced semantic prompt routing and complex task orchestration.**

[View the original repository](https://github.com/SomeOddCodeGuy/WilmerAI)

## Key Features

*   **Advanced Contextual Routing:** Directs user requests based on the entire conversation history, not just the latest message, for superior understanding of intent.  This is achieved through a prompt router that intelligently selects specialized workflows based on context, and by the use of in-workflow routing that allows conditional "if/then" logic, allowing a process to dynamically choose its next step based on the output of a previous node.

*   **Node-Based Workflow Engine:**  At the core of WilmerAI is a powerful workflow engine, driven by JSON-defined sequences of steps (nodes). Each node executes specific tasks, and their outputs fuel subsequent steps, enabling sophisticated, chained-thought processes.

*   **Multi-LLM & Multi-Tool Orchestration:**  Workflows can leverage a variety of LLMs and external tools. This means you can connect your own or proprietary LLMs to run on a single prompt, with the potential for a single prompt to utilize 5+ computers, including proprietary APIs, depending on how you build your workflow.

*   **Modular & Reusable Workflows:** Build self-contained workflows for common tasks, then integrate them as reusable nodes within larger, complex workflows, simplifying the development of sophisticated AI agents.

*   **Stateful Conversation Memory:**  WilmerAI maintains comprehensive conversation memory via a chronological summary file, a continually updated "rolling summary" and a searchable vector database for enhanced context and Retrieval-Augmented Generation (RAG).

*   **Adaptable API Gateway:** Seamlessly integrate your existing applications with OpenAI- and Ollama-compatible API endpoints.

*   **Flexible Backend Connectors:**  Connect to a range of LLM backends (OpenAI, Ollama, KoboldCpp) through a flexible configuration system based on Endpoints, API Types, and Presets.

*   **MCPO Server Tool Integration:** Experimental support for MCP server tool calling using MCPO, allowing tool use mid-workflow. Big thank you to [iSevenDays](https://github.com/iSevenDays) for the amazing work on this feature. More info can be found in the [ReadMe](Public/modules/README_MCP_TOOLS.md)

## How it Works

WilmerAI's architecture revolves around its core workflow engine and advanced routing capabilities. When a user submits a prompt, the system analyzes the context, including the full conversation history, to determine the most appropriate workflow to execute. These workflows are constructed using a node-based system, where each node performs a specific task, such as calling an LLM, running a script, or interacting with an external tool.

This flexible framework enables complex, multi-step processes that can leverage multiple LLMs, enhance the quality of responses, and provide users with powerful and customizable AI interactions.

## Getting Started

1.  **Installation:** Follow the easy steps to install the program using the provided scripts (.bat for Windows, .sh for macOS, or manual installation with pip).
2.  **Configuration:** Explore the Public directory's JSON configuration files.
3.  **Endpoint Setup:** Configure your LLM endpoints in the `Endpoints` folder.
4.  **User Selection:** Choose your preferred user template in the `Users` folder and set it as the current user.
5.  **Launch:** Run WilmerAI, connect your application via the API endpoints, and begin experiencing advanced prompt routing and task orchestration.

## User Documentation

*   Detailed user instructions can be found in [/Docs/_User_Documentation/](/Docs/_User_Documentation/README.md)
## Developer Documentation
*   In-depth developer documentation can be found in [/Docs/Developer_Docs/](/Docs/Developer_Docs/README.md)

## Contact

For any questions or feedback, please reach out to WilmerAI.Project@gmail.com

## License
    WilmerAI
    Copyright (C) 2025 Christopher Smith

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

---