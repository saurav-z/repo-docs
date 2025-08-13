<!-- SEO-optimized README for WilmerAI -->

# WilmerAI: Expertly Routing Your LLM Inference 🚀

**Unleash the power of sophisticated, multi-LLM workflows to revolutionize how you interact with and orchestrate your language models. Enhance your AI experience by leveraging a single interface to manage multiple LLMs, crafting custom workflows, and optimizing performance.**

[Link to Original Repo](https://github.com/SomeOddCodeGuy/WilmerAI)

## Key Features 🔑

*   **Prompt Routing:** Direct prompts to custom categories (coding, math, personas, etc.) via user-defined workflows.
*   **Custom Workflows:** Design tailored workflows for each task.
*   **Multi-LLM Orchestration:** Combine the strengths of multiple LLMs in a single prompt for enhanced results.
*   **Offline Wikipedia API Integration:** Leverage the [OfflineWikipediaTextApi](https://github.com/SomeOddCodeGuy/OfflineWikipediaTextApi) for RAG-enhanced factual responses.
*   **Continuous Chat Summaries:** Generate up-to-date summaries to maintain context in extended conversations.
*   **Model Hotswapping:** Maximize VRAM utilization with Ollama's model-swapping feature.
*   **Customizable Presets:** Easily configure LLM parameters via JSON files.
*   **Vision Support via Ollama:** Process images with the power of Ollama's multi-modal abilities.
*   **Mid-Workflow Control:** Implement workflow branching with Conditional Custom Workflow Nodes.
*   **MCP Server Tool Integration using MCPO:** New and experimental support for tool usage mid-workflow. Big thank you to [iSevenDays](https://github.com/iSevenDays) for the amazing work on this feature. More info can be found in the [ReadMe](Public/modules/README_MCP_TOOLS.md)

## Core Concepts & Architecture 🛠️

WilmerAI serves as an intelligent intermediary between your front-end application (e.g., SillyTavern, Open WebUI) and the various LLM APIs you're using. It exposes both OpenAI and Ollama compatible API endpoints for seamless integration, and on the backend, it is capable of communicating with services like OpenAI, KoboldCpp, and Ollama to route prompts to LLMs.

*   **Front-End Integration:** Your front end connects to WilmerAI via standard API endpoints.
*   **Workflow Execution:** Prompts pass through a series of user-defined workflows, potentially involving multiple LLMs and/or tools.
*   **Response Delivery:** WilmerAI processes the responses and delivers the final result back to your application.

### Visualizing the Workflow

[Insert a visual, like the "No-RAG vs RAG" GIF]

*   **Enhanced Performance:** Iterative LLM calls and workflows are key to improving the quality of answers.
*   **Distributed LLMs:** Exploit multiple machines and proprietary APIs for the best results.

## Setup & Configuration ⚙️

### Quick Start

1.  **Installation:** Follow the steps in the `Quick-ish Setup` section.
2.  **Configuration:** Use the pre-made user configurations in `Public/Configs/Users` and configure Endpoints in `Public/Configs/Endpoints`.
3.  **Routing:** Use the routing config files in `Public/Configs/Routing` or create new ones.
4.  **Set the User:** Choose your current user in `Public/Configs/Users/_current-user.json` or using the `--User` argument.

### Detailed Steps

*   [Refer to the README original for detailed instructions.]

## API Endpoints 🔗

WilmerAI supports various API endpoints to connect.

### Supported API Types

*   OpenAI Compatible v1/completions (*requires [Wilmer Prompt Template](Public/Configs/PromptTemplates/wilmerai.json)*)
*   OpenAI Compatible chat/completions
*   Ollama Compatible api/generate (*requires [Wilmer Prompt Template](Public/Configs/PromptTemplates/wilmerai.json)*)
*   Ollama Compatible api/chat
*   KoboldCpp Compatible api/v1/generate (*non-streaming generate*)
*   KoboldCpp Compatible /api/extra/generate/stream (*streaming generate*)

## Troubleshooting 🚧

### [Refer to the original README for the troubleshooting section.]

## Videos & Guides 📹

*   **Setup Tutorial:** [WilmerAI Setup Tutorial](https://www.youtube.com/watch?v=v2xYQCHZwJM)
*   **Comprehensive Tutorial:** [WilmerAI Tutorial Youtube PlayList](https://www.youtube.com/playlist?list=PLjIfeYFu5Pl7J7KGJqVmHM4HU56nByb4X)
*   **Connecting in SillyTavern:** [Follow instructions in the README.]
*   **Connecting in Open WebUI:** [Follow instructions in the README.]

## Disclaimer & Important Considerations ⚠️

*   **Under Development:** This project is still in active development and is subject to change.
*   **As-Is:** Software is provided "as-is," without any warranties.
*   **Token Usage:** WilmerAI does not track or report token usage. Monitor your LLM API usage separately.
*   **Quality Dependency:** WilmerAI’s outputs depend on the quality of the connected LLMs and the prompts.

## Contact 📧

For support and feedback, contact: WilmerAI.Project@gmail.com

## Third Party Libraries & License 📜

[Refer to the original README.]