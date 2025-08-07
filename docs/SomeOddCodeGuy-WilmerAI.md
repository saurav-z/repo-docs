# WilmerAI: Your Gateway to Advanced LLM Workflows

> Harness the power of multiple Language Models (LLMs) to create sophisticated, custom workflows for enhanced AI interactions. [View the original repo](https://github.com/SomeOddCodeGuy/WilmerAI)

WilmerAI acts as a flexible intermediary between your front-end application and various LLM APIs, enabling you to build intricate workflows that leverage the strengths of different models and tools.

## Key Features

*   **Prompt Routing:** Direct prompts to custom categories (coding, math, personas, etc.) for tailored responses.
*   **Custom Workflows:** Design unique sequences of LLM calls, allowing for iterative refinement and complex operations.
*   **Multi-LLM Orchestration:** Combine responses from multiple LLMs within a single workflow for richer output.
*   **RAG Integration:** Supports the [OfflineWikipediaTextApi](https://github.com/SomeOddCodeGuy/OfflineWikipediaTextApi) for enhanced factual accuracy via Retrieval-Augmented Generation (RAG).
*   **Conversation Memory:** Utilizes chat summaries and recent memories to maintain context across extended conversations.
*   **Ollama Hotswapping:** Maximize VRAM usage with Ollama's model hotswapping, enabling complex workflows on systems with limited resources.
*   **Flexible Presets:** Configure LLM parameters via customizable JSON files, easily adapting to new models and APIs.
*   **Multi-Modal Support (Ollama):** Experimental image processing support using Ollama, even for models without native image handling.
*   **Conditional Workflows:** Initiate new workflows mid-process based on conditions for dynamic and adaptable interactions.
*   **MCP Tool Integration:** New and experimental support for MCP server tool calling using MCPO, allowing tool use mid-workflow. Big thank you to [iSevenDays](https://github.com/iSevenDays) for the amazing work on this feature. More info can be found in the [ReadMe](Public/modules/README_MCP_TOOLS.md)

## Why Use WilmerAI?

WilmerAI was designed to overcome the limitations of traditional semantic routers by allowing users to control the flow of data and get the most out of their LLMs. It does so by allowing multiple models to work together, orchestrated by the workflow itself,  making categorization more complex and customizable. WilmerAI routes to many LLMs via a whole workflow, allowing it to be far more versatile than a standard router.

## Core Concepts

*   **Workflows:** Chain multiple LLM calls and tool integrations for complex processing.
*   **Routing:** Categorize prompts and direct them to the appropriate workflows.
*   **Endpoints:** Define connections to different LLM APIs (OpenAI, Ollama, KoboldCpp, etc.).
*   **Presets:** Customize LLM parameters for each API or workflow.
*   **Memory Management:** Use the chat summary and recent memory nodes for more informed responses.

## Quick Setup

1.  **Install:**  Follow the instructions in the [Quick-ish Setup](#quick-ish-setup) section to install the program, either using the provided scripts or manually.

2.  **Choose a User Template:** In the `Public/Configs/Users` folder, select and configure a pre-made user template. Templates are for `assistant-single-model`, `assistant-multi-model`, `convo-roleplay-single-model`, `convo-roleplay-dual-model`, `group-chat-example`, `openwebui-norouting-single-model`, `openwebui-norouting-dual-model`, `openwebui-routing-multi-model`, `openwebui-routing-single-model`, `socg-openwebui-norouting-coding-complex-multi-model`, `socg-openwebui-norouting-coding-dual-multi-model`, `socg-openwebui-norouting-coding-reasoning-multi-model`, `socg-openwebui-norouting-coding-single-multi-model`, `socg-openwebui-norouting-general-multi-model`,  and `socg-openwebui-norouting-general-offline-wikipedia`.
    *   Update the `Endpoints` as directed in the [Endpoints](#endpoints) section.
    *   Set the user in `Public/Configs/Users/_current-user.json`.

3.  **Configure Endpoints:**  Within `Public/Configs/Endpoints` you'll find examples. Modify these to match your LLM API details.

4.  **Build Workflows:** Define your desired workflow logic in the `Public/Workflows` directory, within your user's name.

## Wilmer AI API Endpoints

### How Do You Connect To Wilmer?

Wilmer exposes several different APIs on the front end, allowing you to connect most applications in the LLM space
to it.

Wilmer exposes the following APIs that other apps can connect to it with:

- OpenAI Compatible v1/completions (*requires [Wilmer Prompt Template](Public/Configs/PromptTemplates/wilmerai.json)*)
- OpenAI Compatible chat/completions
- Ollama Compatible api/generate (*requires [Wilmer Prompt Template](Public/Configs/PromptTemplates/wilmerai.json)*)
- Ollama Compatible api/chat

### What Wilmer Can Connect To

On the backend, Wilmer is capable to connecting to various APIs, where it will send its prompts to LLMs. Wilmer
currently is capable of connecting to the following API types:

- OpenAI Compatible v1/completions
- OpenAI Compatible chat/completions
- Ollama Compatible api/generate
- Ollama Compatible api/chat
- KoboldCpp Compatible api/v1/generate (*non-streaming generate*)
- KoboldCpp Compatible /api/extra/generate/stream (*streaming generate*)

Wilmer supports both streaming and non-streaming connections, and has been tested using both Sillytavern
and Open WebUI.

## Additional Documentation

*   **[YouTube Videos](#youtube-videos)**

## Disclaimer

> This is a personal project under active development. Expect potential bugs, incomplete code, and other issues. The software is provided "as is," without warranty. Updates are made in the maintainer's free time.

## Contact

For questions, feedback, or collaboration: WilmerAI.Project@gmail.com