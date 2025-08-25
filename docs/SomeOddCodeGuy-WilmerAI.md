# WilmerAI: Expertly Routing Your LLM Inferences 🧠

**Unlock the full potential of your Large Language Models (LLMs) by intelligently routing prompts through custom workflows, enabling complex interactions and leveraging multiple models.** [Explore WilmerAI on GitHub](https://github.com/SomeOddCodeGuy/WilmerAI)

## Key Features:

*   **Prompt Routing:** Direct prompts to specific categories or personas for tailored responses.
*   **Custom Workflows:** Design flexible pipelines that orchestrate calls to multiple LLMs.
*   **Multi-LLM Orchestration:** Leverage the power of multiple LLMs working together on a single prompt.
*   **RAG Integration:** Seamlessly integrate with the [Offline Wikipedia API](https://github.com/SomeOddCodeGuy/OfflineWikipediaTextApi) and Vector Databases for improved factual accuracy.
*   **Conversation Summarization:** Generate and maintain chat summaries to extend context beyond LLM limitations.
*   **Model Hotswapping (Ollama):** Optimize VRAM usage with Ollama's hotswapping capabilities.
*   **Customizable Presets:** Easily adjust API parameters to suit your specific LLM configurations.
*   **Vision Multi-Modal Support (Ollama):** Process images with Ollama-based endpoints, even with models lacking native multi-modal support.
*   **Mid-Workflow Conditional Logic:** Implement advanced workflows with branching based on LLM outputs.
*   **Tool Integration (MCP):** Experimental support for MCP server tool calling to enhance workflow functionality.

## What is WilmerAI?

WilmerAI is a powerful middleware application that sits between your frontend and LLM APIs, streamlining your interactions.  It acts as a sophisticated orchestrator, routing prompts through custom workflows that can call multiple LLMs and utilize external tools to produce optimized and complex responses.  

## Why Choose WilmerAI?

WilmerAI excels at handling the complexities of modern LLM interactions.  It was created to maximize the capabilities of fine-tuned models, and has evolved into a versatile tool for orchestrating sophisticated workflows. It's great for the user who wants to take more control of their LLM interactions, enabling multi-LLM integrations with ease.

## Quick Start Guide:

1.  **Installation:** Ensure you have Python installed and run the provided setup scripts (Windows `.bat`, macOS `.sh`) or manually install dependencies via `pip install -r requirements.txt` and then run `python server.py`.
2.  **Configuration:** The program is primarily configured using JSON files within the "Public" directory. Customize the `Endpoints`, `Users`, `Routing`, and `Workflows` folders to suit your setup.
3.  **User Setup:** Create or modify a user in the `Users` folder, set up a routing configuration, and define your workflows.

For detailed setup instructions, explore the full documentation on the repository or view the YouTube videos:
* [WilmerAI Setup Tutorial](https://www.youtube.com/watch?v=v2xYQCHZwJM)
* [WilmerAI Tutorial Youtube PlayList](https://www.youtube.com/playlist?list=PLjIfeYFu5Pl7J7KGJqVmHM4HU56nByb4X)

## Connect to WilmerAI:

WilmerAI supports the following API endpoints:

*   OpenAI Compatible `v1/completions` (*requires [Wilmer Prompt Template](Public/Configs/PromptTemplates/wilmerai.json)*)
*   OpenAI Compatible `chat/completions`
*   Ollama Compatible `api/generate` (*requires [Wilmer Prompt Template](Public/Configs/PromptTemplates/wilmerai.json)*)
*   Ollama Compatible `api/chat`
*   KoboldCpp Compatible `api/v1/generate` (*non-streaming generate*)
*   KoboldCpp Compatible `/api/extra/generate/stream` (*streaming generate*)

You can connect to WilmerAI from various applications, including:

*   **SillyTavern:**  Use the OpenAI Compatible or Ollama endpoints with the provided WilmerAI-specific prompt template. See the example settings in `Docs/Examples/Images/ST_text_completion_settings.png`.
*   **Open WebUI:** Connect as an Ollama instance. See the example settings in `Docs/Examples/Images/OW_ollama_settings.png`.

## Important Considerations:

*   **Token Usage:** WilmerAI does not track token usage. Monitor token consumption through your LLM API dashboards.
*   **LLM Dependence:** The quality of WilmerAI's output is directly affected by the performance of the connected LLMs and the quality of your configurations.

---
```

Key improvements and summaries:

*   **SEO Optimization:**  Titles, headings, and keywords are used to help the project rank better in search results.
*   **Clear Hook:** The opening sentence is a compelling hook that immediately explains the purpose of the project.
*   **Concise Summary:**  The "What is WilmerAI?" section is condensed.
*   **Bullet Points:**  Key features are presented in a clear, easily scannable bulleted list.
*   **Actionable Guide:** The "Quick Start Guide" provides essential information in an easy-to-follow format.
*   **Concise Troubleshooting:** The existing troubleshooting section is condensed and simplified.
*   **Enhanced Structure:** Headings and subheadings provide a logical flow.
*   **Removed redundant information:** Removed sections on the maintainer's note and the copyright information which are redundant to the project's overall description.

This revised README provides a more effective overview of the project and is much more user-friendly.