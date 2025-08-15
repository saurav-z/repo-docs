# WilmerAI: Supercharge Your LLM Workflows

WilmerAI is your versatile intermediary, orchestrating complex LLM interactions to deliver enhanced results, acting as a powerful workflow engine. [Explore the original repository on GitHub](https://github.com/SomeOddCodeGuy/WilmerAI).

## Key Features:

*   **Flexible Prompt Routing:** Route prompts to custom categories or personas using configurable workflows, enabling tailored responses.
*   **Customizable Workflows:** Create intricate, multi-step workflows, including calls to multiple LLMs and tools for advanced processing.
*   **Multi-LLM Orchestration:** Leverage the power of numerous LLMs, working in tandem, to produce superior outputs.
*   **Offline Wikipedia Integration:** Utilize the Offline Wikipedia API for RAG-based information retrieval, improving factual accuracy.
*   **Conversational Memory & Summarization:** Implement chat summaries and memory nodes for continuous context and enhanced conversational consistency.
*   **Model Hotswapping:** Utilize Ollama's hotswapping to make optimal use of your VRAM, allowing you to work with more models.
*   **Customizable Presets:** Easily configure and modify LLM parameters using JSON-based presets for diverse API support.
*   **Multimodal Support via Ollama:** Experiment with image processing capabilities via the api/chat endpoint.
*   **Mid-Workflow Logic:** Implement conditional workflows and advanced logic based on node outputs, enabling dynamic prompt routing.
*   **MCP Server Tool Integration:** Utilize the MCPO to call tools mid-workflow

## Why Choose WilmerAI?

Built during the Llama 2 era, WilmerAI evolved from a simple router into a comprehensive workflow system. It facilitates
complex prompt processing, multi-LLM interactions, and contextual awareness. With WilmerAI, users gain granular control
over the path LLMs take, which also allows for the creation of more complex and customizable categorization than standard
routers. By enabling the collaboration of multiple LLMs, WilmerAI empowers a single prompt with the capabilities of a
complex workflow.

## Key Use Cases

*   **Advanced Code Generation:** Complex workflows tailored for coding tasks, including multi-model collaboration and image-based UX analysis.
*   **Roleplaying & Conversational AI:** Single and dual model workflows for conversation, including character memory and persona management.
*   **Factual Information Retrieval:** Workflows that RAG against the OfflineWikipediaTextApi.

## Getting Started

### Quick Setup

1.  **Installation:** Follow the provided scripts or install dependencies using `pip install -r requirements.txt` and then run `python server.py`.
2.  **Endpoint Configuration:** Configure your LLM API endpoints in the `Public/Configs/Endpoints` directory.
3.  **User Setup:**
    *   Choose a pre-made user configuration from `Public/Configs/Users/` or create a new one.
    *   Edit `Public/Configs/Users/_current-user.json` to specify your active user.
4.  **Routing and Workflows:** Customize routing logic within the `Routing` and `Workflows` folders to fit your needs.

### Connecting to WilmerAI

WilmerAI exposes APIs compatible with both OpenAI and Ollama, supporting streaming and non-streaming connections. You can connect WilmerAI to frontends like SillyTavern and Open WebUI using the following settings:

*   **OpenAI v1/completions**: Configure text completion as an OpenAI compatible v1/completions endpoint.
*   **OpenAI chat/completions**: Use settings for OpenAI chat/completions.
*   **Ollama api/generate**: Connect with the Ollama generate API.
*   **Ollama api/chat**: Connect with the Ollama chat API.

Detailed setup instructions and troubleshooting tips are available in the original README, and video tutorials are linked
in the documentation.

## Disclaimer

This project is under active development and may contain bugs. It's provided "as-is" without warranty. The maintainer, who
has a full-time job, will update the project in their free time.