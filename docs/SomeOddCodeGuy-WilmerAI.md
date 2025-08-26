# WilmerAI: Expertly Routing Your LLM Inference

**Unlock the power of advanced, customizable workflows for your Large Language Models (LLMs) by seamlessly routing prompts and orchestrating multi-LLM interactions.** [View the original repo](https://github.com/SomeOddCodeGuy/WilmerAI)

## Key Features

*   **Dynamic Prompt Routing:** Categorize prompts into domains (coding, math, personas) for tailored processing, using flexible workflows for the most complex logic.
*   **Customizable Workflows:** Craft unique sequences of actions, with each step of a workflow having a specific purpose, for complete control.
*   **Multi-LLM Orchestration:** Leverage multiple LLMs within a single workflow, including your own local hardware or proprietary APIs, for enhanced results.
*   **Offline Wikipedia API Support:** Integrate the [OfflineWikipediaTextApi](https://github.com/SomeOddCodeGuy/OfflineWikipediaTextApi) for fact-based responses via Retrieval-Augmented Generation (RAG).
*   **Contextual Chat Memory:**  The Chat Summary node will summarize your chats and create "memories", by chunking your messages and then summarizing them and saving them to a file. It will then take those summarized chunks and generate an ongoing, constantly updating, summary of the entire conversation. This allows conversations that far exceed the LLM's context to continue to maintain some level of consistency.
*   **Ollama Hotswap for VRAM Efficiency:** Use Ollama's hotswapping to run complex workflows on systems with limited VRAM.
*   **Customizable Presets:** Configure LLM parameters through easy-to-edit JSON files, adaptable to new LLM features.
*   **Vision Multi-Modal Support (Ollama):** Process images through the Ollama API, even when the LLM has limited multimodal capabilities.
*   **Mid-Workflow Conditional Routing:** Build dynamic workflows that change course based on LLM responses.
*   **MCP Tool Integration:** Experimental support for MCP server tool calling using MCPO, allowing tool use mid-workflow. More info can be found in the [ReadMe](Public/modules/README_MCP_TOOLS.md)

## What is WilmerAI?

WilmerAI is a flexible middleware designed to bridge the gap between your front-end applications (or any program calling LLMs) and your chosen LLM APIs. It exposes OpenAI- and Ollama-compatible endpoints, allowing you to connect to a wide range of LLM providers like OpenAI, KoboldCpp, and Ollama. WilmerAI enables sophisticated workflows that may involve multiple LLMs and tools working together to produce complex results. It simplifies the interaction with the models.

## Quick Setup

To get started:

1.  **Installation:** Install Python and run the setup scripts, or manually install requirements.txt.
2.  **Configure Endpoints:** Set up your LLM API endpoints (in `Public/Configs/Endpoints`).
3.  **Create a User:** Define user configurations (in `Public/Configs/Users`).  Set current user in `_current-user.json`.
4.  **Customize Routing:** Configure prompt categorization (in `Public/Configs/Routing`) for better workflow performance.
5.  **Build Your Workflows:** Customize workflows to suit your use case (in `Public/Workflows/<username>`).

See more in-depth setup instructions in the original README, including using the provided .bat and .sh files.

## Important Notes

*   **Disclaimer:** This project is under heavy development and may contain bugs. It is provided "as-is" without any warranty.
*   **Token Usage:**  WilmerAI doesn't track token usage. Monitor API token consumption through your LLM provider's dashboard.
*   **Model Quality:** The quality of your LLM responses directly depends on the connected LLMs, presets, and prompt templates.