# WilmerAI: Expertly Routing Your Language Model Inference

**WilmerAI is a powerful middleware application that allows you to orchestrate complex workflows involving multiple Language Model (LLM) APIs, enhancing the capabilities of your front-end applications and LLM projects. ([Original Repo](https://github.com/SomeOddCodeGuy/WilmerAI))**

## Key Features

*   **Flexible Prompt Routing:** Direct prompts to custom categories (coding, math, etc.) or personas for group chats, enabling tailored responses.
*   **Customizable Workflows:** Design workflows that go beyond routing, allowing every prompt to follow a specific, multi-step process.
*   **Multi-LLM Orchestration:** Send a single prompt to multiple LLMs simultaneously within a single workflow to create the optimal response.
*   **RAG Integration:** Leverage the Offline Wikipedia API to improve factual accuracy by adding context from Wikipedia.
*   **Conversation Memory:** The Conversation Memory node will generate "memories", by chunking your messages and then summarizing them and saving them to a file.
*   **Model Hotswapping:** Maximize VRAM by leveraging Ollama's hotswapping feature, enabling complex workflows on resource-constrained systems.
*   **Customizable Presets:** Easily configure and customize LLM API calls via user-defined JSON files, adding flexibility.
*   **Ollama Vision Support:** Provides experimental image processing support when using Ollama as the frontend and backend, allowing for more detailed analysis.
*   **Mid-Workflow Conditionals:** Initiate new workflows based on conditions met mid-conversation, providing flexibility.
*   **Tool Integration:** Experimental support for MCP server tool calling using MCPO, allowing tool use mid-workflow, provided by [iSevenDays](https://github.com/iSevenDays).

## Core Concepts

WilmerAI facilitates complex AI applications by acting as an intermediary between your front-end (or any application that makes LLM API calls) and the LLM APIs. WilmerAI exposes OpenAI- and Ollama-compatible API endpoints, and connects to the supported LLM APIs like OpenAI, KoboldCpp, and Ollama.

Workflows are the foundation of WilmerAI. These customizable processes can be made of individual nodes that make calls to multiple LLMs. From your perspective, a single prompt is sent to WilmerAI, which then orchestrates one or more workflows with multiple LLMs, returning a single result.

## Quick Start

1.  **Installation:** Follow the provided setup instructions via Python.
2.  **Setup:** Configure endpoints and models within the `Public/Configs` directory.
3.  **User Configuration:** Set up user profiles, routing, and workflows via JSON files.
4.  **Connection:** Connect your front-end to WilmerAI using the OpenAI- or Ollama-compatible API endpoints.

## Understanding Workflows

The heart of WilmerAI's power lies in its flexible and customizable workflows.

### Workflow Structure

Workflows are defined using JSON files made up of "nodes." Each node performs a specific task, such as calling an LLM, running a Python script, or searching a memory.

**Two primary formats are supported:**

*   **New Format (Recommended)**: Utilizes dictionary-based structure for top-level configuration and variables to organize nodes.
*   **Old Format (Still Supported)**: A simple list of nodes, ensuring full backward compatibility.

### Node Properties

Each node is a JSON object with properties to define how the node behaves:

*   **`type`** (Required): Specifies the function of the node. Common types include `Standard`, `PythonModule`, `VectorMemorySearch`, and `CustomWorkflow`.
*   **`title`**: Descriptive name for the node (used for debugging).
*   **`systemPrompt`**: The system prompt sent to the LLM API.
*   **`prompt`**: The user prompt to send.
*   **`lastMessagesToSendInsteadOfPrompt`**: Specifies how many recent messages to send to the LLM.
*   **`endpointName`**: The LLM API endpoint to use.
*   **`preset`**: The preset to use for the API call (e.g., temperature, token limits).
*   **`maxResponseSizeInTokens`**: Overrides the preset for max response tokens.
*   **`returnToUser`**: If set to `true`, outputs from a node are returned to the user.
*   **`useRelativeTimestamps`**: Prepends timestamps to messages (e.g., `[Sent 5 minutes ago]`).
*   **`workflowName`**: Used in `CustomWorkflow` nodes to specify sub-workflows.
*   **`scoped_variables`**: Variables from parent workflows for child workflows.

### Memory Management

WilmerAI features a robust memory system to store and retrieve context for more intelligent conversations.

*   **Vector Memories:**  Create highly-relevant, keyword-based search results through the vector database.

*   **Chat Summary:**  Create a rolling summary of the conversation.

*   **Long-Term Memory:** Classic memory file.

### Conditional Custom Workflow Node

The **`ConditionalCustomWorkflow` Node** allows for dynamic selection and execution of sub-workflows based on a variable:

*   `conditionalKey` (string, required): Variable whose value determines which workflow to execute.
*   `conditionalWorkflows` (object, required): Maps values of `conditionalKey` to workflow files.
*   `Default` (string, optional): Fallback workflow.
*   `is_responder` (boolean, optional, default: `false`):  Controls output response.
*   `scoped_variables` (array of strings, optional): Passed variables into child workflows.
*   `routeOverrides` (object, optional): Overrides prompts for specific routes.

## API Endpoints

WilmerAI offers these API endpoints for seamless integration with LLM-based applications:

*   OpenAI Compatible v1/completions
*   OpenAI Compatible chat/completions
*   Ollama Compatible api/generate
*   Ollama Compatible api/chat
*   KoboldCpp Compatible api/v1/generate (non-streaming)
*   KoboldCpp Compatible /api/extra/generate/stream (streaming)

## Important Notes

*   **Token Usage:** WilmerAI does not track or report token usage. Monitor token usage via your LLM API dashboards.
*   **LLM Dependency:**  The quality of WilmerAI depends on the quality of the connected LLMs.
*   **Development:** WilmerAI is under active development. Expect occasional bugs and breaking changes.

## YouTube Videos

*   WilmerAI Setup Tutorial ([Link](https://www.youtube.com/watch?v=v2xYQCHZwJM))
*   WilmerAI Tutorial YouTube PlayList ([Link](https://www.youtube.com/playlist?list=PLjIfeYFu5Pl7J7KGJqVmHM4HU56nByb4X))

## Contact

For any questions, feedback, or inquiries, contact: WilmerAI.Project@gmail.com