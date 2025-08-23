# WilmerAI: Expertly Routing Your LLM Inferences

**Unlock the power of advanced workflows and connect multiple Large Language Models (LLMs) with WilmerAI, your intelligent intermediary for seamless and efficient AI interactions.** ([Original Repo](https://github.com/SomeOddCodeGuy/WilmerAI))

## Key Features:

*   **Versatile API Endpoints:** Compatible with OpenAI and Ollama APIs for easy integration.
*   **Custom Workflows:** Design unique multi-step processes to orchestrate LLMs for complex tasks.
*   **Prompt Routing:** Route prompts to specific domains or personas, optimizing for various use cases.
*   **Multi-LLM Orchestration:** Leverage numerous LLMs within a single call, maximizing performance.
*   **Offline Wikipedia Integration:** Utilize the Offline Wikipedia API for enhanced factual accuracy (RAG).
*   **Contextual Chat Summaries:** Generate real-time summaries to simulate "memory" for extended conversations.
*   **Model Hotswapping:** Maximize VRAM usage with Ollama's model hotswapping capabilities.
*   **Customizable Presets:** Tailor model parameters with easily editable JSON configuration.
*   **Image Processing via Ollama:** Experiment with processing images in your workflows through Ollama.
*   **Conditional Mid-Workflow Workflows:** Design workflows that adapt to different conditions.
*   **MCP Server Tool Integration:** Experimental support for MCP server tool calling for tool use mid-workflow.

## What is WilmerAI?

WilmerAI is a powerful application that sits between your frontend (or other LLM-calling programs) and your LLM APIs, allowing you to construct semi-autonomous workflows, connect to a variety of LLMs, and create a more powerful, adaptable, and cost-effective AI experience.

## Getting Started

### Setup

1.  **Installation**: Follow the simple instructions, using the provided scripts (Windows .bat, macOS .sh) or by manually installing Python dependencies (`pip install -r requirements.txt`) and running `python server.py`.
2.  **Pre-made Users (Fast Route)**: Leverage pre-configured example user setups to streamline your initial experience, providing a quick way to get familiar with all the potential features.  Remember to update the endpoints configuration!
3.  **Endpoints**: Configure API endpoint connections, defining API type, and model settings.
4.  **Users**: Create or modify user settings and user specific endpoints for each of your different LLMs.
5.  **Routing**: Define how prompts will be directed with routing, or bypass routing entirely to use a single, pre-defined workflow.
6.  **Workflows**: Explore comprehensive workflows, or develop your own to take advantage of the maximum potential WilmerAI has to offer.

### Core Concepts

*   **Workflows**: Construct pipelines to use a variety of different APIs to achieve a particular outcome.
*   **Nodes**: Each workflow is built upon nodes, each having a specific function such as a LLM call or tool use.
*   **Memory System**: Leverage the "DiscussionId" tag and the built-in creator and retriever nodes to generate a permanent memory and rolling chat summary to vastly improve LLM responses.

## Advanced Features and Capabilities

*   **Vector Memory**: The most powerful option for RAG, build advanced keyword based searches by leveraging SQLite.
*   **Rolling Chat Summary**: Gives the AI a bird's eye view of everything that has happened so far.
*   **Python Module Caller Node**: Call any `.py` file with the `Invoke(*args, **kwargs)` method that returns a string.
*   **Workflow Lock**: Locks the workflow at a certain point during asynchronous operations, so that you don't encounter race conditions of two instances of a workflow crashing into each other via consecutive calls.

## Disclaimer

This is an actively developed personal project, provided "as is" without any warranty.

## Get Involved

For any queries, requests, and feedback, contact WilmerAI.Project@gmail.com

## Resources

*   [Setup Tutorial](https://www.youtube.com/watch?v=v2xYQCHZwJM)
*   [Tutorial Youtube PlayList](https://www.youtube.com/playlist?list=PLjIfeYFu5Pl7J7KGJqVmHM4HU56nByb4X)