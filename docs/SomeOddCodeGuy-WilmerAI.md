# WilmerAI: Expertly Routing Language Models for Complex Workflows

**Unleash the power of multi-LLM workflows with WilmerAI, a flexible and extensible application that intelligently orchestrates AI interactions.**  [View on GitHub](https://github.com/SomeOddCodeGuy/WilmerAI)

## Key Features

*   **Prompt Routing:**  Categorize and direct prompts to specialized workflows based on domain or persona, ensuring the right LLM handles the task.
*   **Custom Workflows:** Design intricate chains of LLM calls, tools, and processing steps to achieve sophisticated results, offering granular control over the AI interaction.
*   **Multi-LLM Coordination:**  Leverage the power of multiple LLMs working in tandem within a single call, maximizing performance and leveraging diverse AI capabilities.
*   **Offline Wikipedia Integration:** Seamlessly incorporate the [OfflineWikipediaTextApi](https://github.com/SomeOddCodeGuy/OfflineWikipediaTextApi) for enhanced factual accuracy through Retrieval-Augmented Generation (RAG).
*   **Conversation Memory:**  Maintain long-term consistency with continually updated chat summaries, enabling LLMs to understand and recall the full context of extended conversations.
*   **Model Hotswapping:** Optimize VRAM usage through Ollama's hotswapping, allowing you to run complex workflows even with limited hardware.
*   **Customizable Presets:** Tailor LLM interactions with easily configurable JSON presets, allowing you to adapt and optimize responses for various models and use cases.
*   **Vision Multi-Modal Support:** Harness image processing capabilities via Ollama, enabling visual understanding and analysis within your workflows.
*   **Mid-Workflow Conditional Branching:** Dynamically redirect workflows based on LLM responses, enabling intelligent decision-making within complex processes.
*   **Tool Integration:** Experimentally supports MCP server tool calling using MCPO. (Thank you to [iSevenDays](https://github.com/iSevenDays) for the amazing work on this feature.)

## Why Use WilmerAI?

WilmerAI moves beyond basic prompt routing.  Designed for maximum use of finetunes via prompt routing, it now focuses on **semi-autonomous workflows**, empowering users with unparalleled control over LLM interactions. You are the conductor, leading your AI symphony for superior, multi-model outputs.

## Quick Start

WilmerAI requires a bit of setup, mostly controlled by JSON configuration files. For quick deployment:

1.  **Install:** Follow the installation instructions in the original [README](https://github.com/SomeOddCodeGuy/WilmerAI#quick-ish-setup)
2.  **Configure Endpoints:** Update the `Endpoints` folder with your LLM API details.
3.  **Choose a User:** Edit `_current-user.json` to select a pre-built or custom user configuration.
4.  **Run:** Launch WilmerAI and connect your preferred front-end application.

## Key Concepts

*   **Endpoints:** Configure connections to various LLM APIs (OpenAI, Ollama, KoboldCpp, etc.).
*   **Users:** Manage individual configurations, including workflows and routing.
*   **Routing:** Define categories and triggers for initiating specific workflows.
*   **Workflows:** Assemble sequential steps involving LLMs, tools, and memory management.
*   **Memory:** Utilize file-based (long-term) and smart vector memory, as well as rolling summaries, to create rich, context-aware conversations.
*   **Presets:** Customize prompt parameters for optimal performance.

## [View Full Documentation](https://github.com/SomeOddCodeGuy/WilmerAI)

For a complete walkthrough of all features, as well as troubleshooting tips, be sure to view the full ReadMe.

## Important Considerations

*   **Token Tracking:** WilmerAI does not track token usage. Monitor your LLM API dashboards for cost management.
*   **Quality Dependent:** The quality of your results directly depends on the LLMs, prompts, and preset configurations you choose.
*   **Development Status:** This is an active project with ongoing development. Please be patient with any potential bugs and always inspect the code before using it.