# WilmerAI: Orchestrate Your LLMs with Advanced Workflows

**Unleash the power of interconnected Language Models!** WilmerAI is a versatile application that acts as a central hub for managing and routing prompts to your preferred LLM APIs, allowing for complex, multi-LLM workflows.  [Visit the original repository](https://github.com/SomeOddCodeGuy/WilmerAI) for more details.

## Key Features

*   **Modular Prompt Routing:** Categorize and direct prompts to various workflows based on domain or persona.
*   **Customizable Workflows:** Build intricate workflows that involve multiple LLMs and tools for sophisticated responses.
*   **Multi-LLM Orchestration:** Leverage several LLMs in a single call to enhance quality and achieve specialized results.
*   **RAG with OfflineWikipediaTextApi Integration:** Seamlessly incorporates the [OfflineWikipediaTextApi](https://github.com/SomeOddCodeGuy/OfflineWikipediaTextApi) for contextually relevant factual responses.
*   **Persistent Chat Summaries & Vector Memory:** Maintain context with auto-generated summaries and state-of-the-art vector memory search.
*   **Flexible Model Hotswapping:** Maximize VRAM usage by using Ollama's hotswapping capabilities.
*   **Highly Customizable Presets:** Easily modify and manage LLM API parameters through JSON-based presets.
*   **Vision Multi-Modal Support via Ollama:** Experimental support for image processing when using Ollama as the front-end API, and having an Ollama backend API to send it to.
*   **Conditional Mid-Workflow Logic:** Conditionally trigger workflows based on LLM responses for adaptable processing.
*   **Tool Integration with MCP:** Integration with MCP server tool calling using MCPO, allowing tool use mid-workflow.
   
## Get Started

*   **Installation:** Simple setup using provided scripts or manual dependency installation. See the README at the original repo for step-by-step instructions.
*   **Connect to WilmerAI**:  Use the following APIs:
    *   OpenAI Compatible v1/completions (*requires [Wilmer Prompt Template](Public/Configs/PromptTemplates/wilmerai.json)*)
    *   OpenAI Compatible chat/completions
    *   Ollama Compatible api/generate (*requires [Wilmer Prompt Template](Public/Configs/PromptTemplates/wilmerai.json)*)
    *   Ollama Compatible api/chat
*   **Pre-Made Users**: Start quickly with pre-configured user profiles in the Public/Configs directory.
*   **Endpoints and Models**: Configure your LLM API endpoints to match your preferred models. See the README at the original repo for detailed steps.
*   **Workflows**: Customize workflows to route prompts and create complex and multi-model responses. See the README at the original repo for full explanations.

## Important Considerations

*   **Token Usage:** WilmerAI does not track or report token usage, so monitor your LLM API dashboards.
*   **LLM Dependence:** The quality of responses directly depends on the connected LLMs and well-written prompts.

## Community Resources

*   [Setup Tutorial](https://www.youtube.com/watch?v=v2xYQCHZwJM)
*   [Tutorial Playlist](https://www.youtube.com/playlist?list=PLjIfeYFu5Pl7J7KGJqVmHM4HU56nByb4X)

---

**Disclaimer:**  This is a personal project under development and is provided "as is" without any warranty.