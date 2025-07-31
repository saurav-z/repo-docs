# WilmerAI: Expertly Routing Language Models for Advanced AI Workflows

Tired of single-model limitations? **WilmerAI lets you orchestrate complex workflows, connecting multiple LLMs across various systems to generate a single, powerful response.** This personal project empowers you to build sophisticated AI assistants with unparalleled flexibility and control. [Explore the original repository on GitHub](https://github.com/SomeOddCodeGuy/WilmerAI).

## Key Features:

*   **Dynamic Prompt Routing:** Direct prompts to specialized categories (e.g., coding, factual, personas) for tailored responses.
*   **Customizable Workflows:** Design workflows that chain multiple LLMs, perform iterative calls, and integrate custom Python scripts.
*   **Multi-LLM Orchestration:** Leverage the power of several models at once by running separate nodes within a workflow.
*   **RAG Integration:** Seamlessly integrate with the Offline Wikipedia API for improved factual accuracy and knowledge retrieval.
*   **Continual Chat Summaries:** Maintain context in long conversations with dynamically generated summaries that surpass the context window of LLMs.
*   **Maximize VRAM Usage:** Leverage Ollama's hotswapping capabilities to run several models simultaneously, even with limited VRAM.
*   **Customizable Presets:** Easily adapt to new model updates and API settings.
*   **Vision & Multi-Modal Support:** Utilizes Ollama's image API endpoints to work with images.
*   **Mid-Workflow Conditional Logic:** Implement dynamic branching within workflows based on LLM outputs and provide custom prompt overrides.
*   **MCP Server Tool Integration using MCPO:** Integrate advanced tool use for advanced processing.
*   **Flexible API Endpoints:**  Compatible with OpenAI and Ollama API standards.

## What's New (Updated: June 29, 2025):

*   **Improved Reasoning Model Support:**  Wilmer now gracefully handles reasoning models and can strip out unwanted "thinking" blocks from LLM responses.
*   **New MCP Server Tool Integration:**  Full support for tool integration and utilization using MCPO.

## Get Started:

### Key Setup Steps:

1.  **Installation:** Use the provided `.bat` (Windows) or `.sh` (macOS) files, or manually install dependencies via `pip install -r requirements.txt` and run `python server.py`.
2.  **Configuration:** Customize settings via JSON configuration files within the `Public` folder.  Copy your `Public` folder to preserve your settings when updating.
3.  **Endpoints and Models:** Configure LLM API endpoints in the `Endpoints` folder (e.g., `SmallModelEndpoint.json`).  Refer to the Endpoints and API Types sections for details.
4.  **Create/Configure User:** Create a user file in the `Users` folder and update `_current-user.json` to specify the active user.
5.  **Routing:** Define prompt routing logic in the `Routing` folder.
6.  **Workflows:**  Customize multi-stage processing and LLM usage with the powerful `Workflow` framework.

### Connecting to Wilmer

Wilmer exposes OpenAI and Ollama compatible endpoints:

*   **SillyTavern:** Connect as OpenAI Compatible v1/Completions or Ollama api/generate.  Import the WilmerAI-specific prompt template (Docs/SillyTavern/InstructTemplate).
*   **Open WebUI:** Connect as an Ollama instance.

## Examples

*   **Single Assistant Routing to Multiple LLMs**:  A single prompt is sent to a category that selects from multiple models.
*   **Silly Tavern Groupchat to Different LLMs**:  Character settings from SillyTavern will call different LLMs.
*   **An Oversimplified Example Coding Workflow**:  Leverage different tools for code generation.
*   **An Oversimplified Conversation/Roleplay Workflow**:  Create a roleplay assistant to engage in a conversation.

## Important Considerations:

*   **Token Usage:** WilmerAI does not track token usage. Monitor token consumption through your LLM API dashboards.
*   **LLM Quality:**  The quality of your connected LLMs directly impacts Wilmer's performance.
*   **Maintenance Notice:** Please keep in mind that fixes or updates may take a week or two to be pushed.

## Contact

For questions or feedback, email WilmerAI.Project@gmail.com

## License

[See the full license text](LICENSE.txt)