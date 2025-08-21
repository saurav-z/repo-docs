# WilmerAI: Unleash the Power of Multi-LLM Workflows

**Transform your AI interactions with WilmerAI, a flexible middleware application that orchestrates complex workflows across multiple Language Models.** [Explore the Original Repo](https://github.com/SomeOddCodeGuy/WilmerAI)

WilmerAI allows you to connect your front-end applications to a wide range of Language Model (LLM) APIs, including OpenAI, Ollama, KoboldCpp, and more, enabling sophisticated and customizable AI experiences.

**Key Features:**

*   ✅ **Dynamic Prompt Routing**: Route prompts to specific categories or personas using user-defined workflows.
*   ✅ **Customizable Workflows**: Create tailored workflows to execute a sequence of LLM calls and tool integrations.
*   ✅ **Multi-LLM Orchestration**: Leverage multiple LLMs in a single call to enhance results through collaboration.
*   ✅ **Offline Wikipedia Integration**: Utilize the Offline Wikipedia API for enhanced factual accuracy (RAG).
*   ✅ **Persistent Chat Summaries**: Generate and maintain concise summaries of conversations to provide context.
*   ✅ **Hot-swapping for VRAM Efficiency**: Utilize Ollama's hotswapping to manage VRAM efficiently across LLMs.
*   ✅ **Customizable Presets**: Easily modify and adapt presets to fit specific LLM API requirements.
*   ✅ **Image Processing Support**: Experimental multi-modal support via Ollama to analyze images within your workflows.
*   ✅ **Mid-Workflow Conditional Logic**: Implement branching logic and kick off custom workflows based on conditions.
*   ✅ **MCP Server Tool Integration**: Experimental support for tool calling using MCPO, offering enhanced functionality.

## Core Functionality

WilmerAI sits between your front-end (or any LLM-calling program) and LLM APIs. It exposes OpenAI- and Ollama-compatible API endpoints and connects to various LLM APIs like OpenAI, KoboldCpp, and Ollama on the backend.

### How it Works

1.  Your front end sends a prompt to Wilmer.
2.  Wilmer processes the prompt through a series of custom workflows.
3.  Workflows can call multiple LLMs, tools, or internal functions.
4.  The final response is returned to your front end.

This approach allows you to build advanced AI assistants, combining multiple LLMs and tools to generate more accurate, comprehensive, and contextually aware responses.

### Why Choose WilmerAI?

WilmerAI provides a powerful and flexible framework for building advanced AI applications. Its workflow-based design allows for granular control over the AI interaction process, enabling customization to specific use cases. Whether you're building a coding assistant, a role-playing chatbot, or an AI-powered research tool, WilmerAI offers the tools you need to achieve your goals.

## Getting Started

### Prerequisites

*   **Python**: Ensure Python is installed on your system (3.10 or 3.12 recommended).

### Installation

1.  **Option 1: Using Scripts**
    *   Windows: Run the provided `.bat` file.
    *   macOS: Run the provided `.sh` file.

2.  **Option 2: Manual Installation**
    ```bash
    pip install -r requirements.txt
    python server.py
    ```

### Configuration

WilmerAI is configured through JSON files located in the `Public` folder. This includes endpoint definitions, user profiles, routing configurations, and workflow definitions. Customizing these files is key to tailoring WilmerAI to your needs.

### Connecting to Wilmer

WilmerAI exposes OpenAI-compatible v1/completions, OpenAI chat/completions, Ollama api/generate and Ollama api/chat endpoints. Configure your front-end application to connect to these endpoints.

**Quick Start Guide:**

1.  **Endpoints:** Define the LLM API endpoints in the `Public/Configs/Endpoints` folder (examples are provided).
2.  **User Configuration:** Set your current user in `Public/Configs/Users/_current-user.json`.
3.  **Routing:** Configure routing in the `Public/Configs/Routing` folder.
4.  **Workflows**: Create and customize workflows in the `Public/Workflows` folder, within your user's specific workflows folder.

### Further Resources

*   **YouTube Videos**:
    *   [WilmerAI Setup Tutorial](https://www.youtube.com/watch?v=v2xYQCHZwJM)
    *   [WilmerAI Tutorial YouTube PlayList](https://www.youtube.com/playlist?list=PLjIfeYFu5Pl7J7KGJqVmHM4HU56nByb4X)

## Important Notes and Troubleshooting

*   **Token Usage**: WilmerAI does not track token usage, so please monitor your LLM API dashboards for cost management.
*   **LLM Impact**: The quality of WilmerAI's output depends on the LLMs and prompt templates you use.
*   **Bugs**: This is an actively developed project and may contain bugs.

## Third-Party Libraries

*   Flask
*   requests
*   scikit-learn
*   urllib3
*   jinja2
*   pillow

(See README of ThirdParty-Licenses folder for license details)

## License and Copyright

WilmerAI is licensed under the GNU General Public License v3.

© 2025 Christopher Smith

For additional details, feedback, or support, contact:

WilmerAI.Project@gmail.com