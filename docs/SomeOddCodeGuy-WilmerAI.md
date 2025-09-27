# WilmerAI: Unleash the Power of Context-Aware Workflows for Advanced LLM Orchestration

**WilmerAI is a powerful application that goes beyond simple prompt routing, enabling you to build intelligent, multi-step workflows that orchestrate Large Language Models (LLMs) for complex tasks.** 🔗 [View the original repository](https://github.com/SomeOddCodeGuy/WilmerAI)

## Key Features:

*   **Advanced Contextual Routing:** Understands conversation history to direct requests to the most appropriate specialized workflow, using prompt routing and in-workflow conditional logic.
*   **Node-Based Workflow Engine:** Processes requests through JSON-defined workflows, enabling complex, chained-thought processes.
*   **Multi-LLM & Multi-Tool Orchestration:** Connect each node in a workflow to different LLM endpoints or external tools for optimized task execution.
*   **Modular & Reusable Workflows:** Build self-contained workflows for common tasks and reuse them within larger workflows.
*   **Stateful Conversation Memory:** Provides context for long conversations with a summary file, rolling summary, and a vector database for Retrieval-Augmented Generation (RAG).
*   **Adaptable API Gateway:** Exposes OpenAI and Ollama-compatible API endpoints, compatible with your existing front-ends.
*   **Flexible Backend Connectors:** Supports connections to various LLM backends, including OpenAI, Ollama, and KoboldCpp, through a simple configuration of Endpoints, API Types, and Presets.
*   **MCP Server Tool Integration:** Uses MCPO for experimental support in mid-workflow tool usage.

## The Power of Workflows:

WilmerAI leverages the power of workflows to unlock unprecedented control and flexibility in your LLM interactions.

*   **Semi-Autonomous Workflows:** Determine which tools to use and when, orchestrating multiple LLMs.
    ![No-RAG vs RAG](Doc_Resources/Media/Gifs/Search-Gif.gif) *Click the image to play gif if it doesn't start automatically*

*   **Iterative LLM Calls:** Improve performance by automatically asking follow-up questions.
*   **Distributed LLMs:** Utilize multiple machines and LLM APIs to make your workflows more powerful.

## Dive into the Potential:

*   **[User Documentation](Docs/_User_Documentation/README.md)**
*   **[Developer Documentation](Docs/Developer_Docs/README.md)**

## Quick-ish Setup:

*   **[Youtube Video Installation](https://www.youtube.com/watch?v=KDpbxHMXmTs)**

*   **Guides:**

    *   [Wilmer API Setup](Docs/_User_Documentation/Setup/_Getting-Start_Wilmer-Api.md)
    *   [Wilmer with Open WebUI](Docs/_User_Documentation/Setup/Open-WebUI.md)
    *   [Wilmer With SillyTavern](Docs/_User_Documentation/Setup/SillyTavern.md)

---