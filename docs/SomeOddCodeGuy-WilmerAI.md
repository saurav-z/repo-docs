# WilmerAI: Expertly Routing Your LLM Interactions

**Unleash the power of multi-LLM workflows for complex tasks, all through a simple, OpenAI-compatible API.**

[View the Original Repo](https://github.com/SomeOddCodeGuy/WilmerAI)

## Key Features

*   **Prompt Routing:** Direct prompts to specific domains (coding, math, etc.) or personas for advanced conversational AI.
*   **Custom Workflows:** Design intricate chains of operations, leveraging multiple LLMs and tools in a single call.
*   **Multi-LLM Collaboration:** Enable multiple LLMs to work in tandem on a single prompt for enhanced performance.
*   **Offline Wikipedia Integration:** Seamlessly integrate with the [OfflineWikipediaTextApi](https://github.com/SomeOddCodeGuy/OfflineWikipediaTextApi) for enhanced factual responses and RAG capabilities.
*   **Contextual Memory:** Generate chat summaries and "memories" to provide long-term conversational context, allowing interactions that exceed LLM limitations.
*   **Model Hotswapping:** Optimize VRAM usage with Ollama's hotswapping, facilitating the use of complex workflows on systems with limited resources.
*   **Customizable Presets:** Configure LLM parameters with custom JSON files to control model behavior precisely.
*   **Multi-Modal Vision Support:** Experimental support for processing multiple images per message via Ollama, enhancing versatility.
*   **Conditional Workflows:** Trigger dynamic workflows mid-conversation based on LLM responses.
*   **Tool Integration:** Experimental support for tool use mid-workflow via MCPO.

## What is WilmerAI?

WilmerAI is a middleware application designed to sit between your front-end or any LLM-calling program and various Large Language Model (LLM) APIs. It exposes OpenAI- and Ollama-compatible API endpoints, allowing you to connect to LLM APIs like OpenAI, KoboldCpp, and Ollama. You type a prompt into your front end, which is connected to Wilmer. The prompt gets sent to Wilmer first, which runs it through a series of workflows. Each workflow may make calls to multiple LLMs, after which the final response comes back to you. From your perspective, it looks like a (likely long-running) one-shot call to an LLM. But in reality, it could involve many LLMs—and even tools—performing complex work.

## Get Started

### Installation

WilmerAI is easy to set up. Follow these steps:

1.  **Prerequisites:** Ensure you have Python installed (versions 3.10 and 3.12 are recommended).
2.  **Installation Options:**
    *   **Using Provided Scripts:** Use the `.bat` (Windows) or `.sh` (macOS) file to create a virtual environment, install dependencies, and run Wilmer.
    *   **Manual Installation:**

        ```bash
        pip install -r requirements.txt
        python server.py
        ```
3.  **Configuration:** Customize settings using JSON configuration files in the `Public` folder. See the documentation for user setup and workflow specifics.

### Connecting to Wilmer

Wilmer provides several API endpoints for easy integration:

*   **OpenAI Compatible:** `/v1/completions`, `/chat/completions`
*   **Ollama Compatible:** `/api/generate`, `/api/chat`
*   **KoboldCpp Compatible:** `/api/v1/generate`, `/api/extra/generate/stream`

Connect using the settings for text completions or chat completions in your preferred front-end application (SillyTavern, Open WebUI, etc.).

## Examples & Visualizations

Explore example workflows that demonstrate WilmerAI's power:

*   Simple Assistant Workflows
*   Prompt Routing Scenarios
*   Group Chat configurations
*   UX-driven Workflow Examples

## Important Notes

*   **Disclaimer:** This is a personal project in active development, provided "as is" without warranty.
*   **Token Usage:** WilmerAI does not track or report token usage. Monitor your LLM API dashboards for accurate cost management.
*   **LLM Quality:** WilmerAI's performance is directly dependent on the quality and accuracy of the connected LLMs.
*   **Contribute:** Until October 2025, WilmerAI will not accept any new Pull Requests that modify anything within the Middleware modules; some exceptions may apply.

## Contact

For feedback, questions, or support, please contact: WilmerAI.Project@gmail.com

## License

WilmerAI is licensed under the GNU General Public License v3.