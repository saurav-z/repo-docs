# WilmerAI: Expertly Routing Your Language Model Inferences

**Tired of basic LLM interactions?** WilmerAI is a powerful middleware application that sits between your front-end or any LLM calling program and various LLM APIs, enabling sophisticated workflows and unlocking advanced LLM capabilities. Check out the original repo [here](https://github.com/SomeOddCodeGuy/WilmerAI)!

## Key Features

*   **Prompt Routing:** Dynamically route prompts to custom categories (coding, math, personas, etc.) for tailored responses.
*   **Custom Workflows:** Design and execute unique, multi-step workflows for complex tasks and enhanced results.
*   **Multi-LLM Orchestration:** Leverage multiple LLMs in a single call for comprehensive processing and superior outputs.
*   **Offline Wikipedia Integration:** Seamlessly integrate the [OfflineWikipediaTextApi](https://github.com/SomeOddCodeGuy/OfflineWikipediaTextApi) for enriched factual responses.
*   **Persistent Chat Summaries:** Generate and maintain chat summaries to provide consistent context even in long conversations.
*   **Dynamic Model Hotswapping:** Maximize VRAM usage with Ollama's hotswapping for complex workflows on limited hardware.
*   **Customizable Presets:** Fine-tune LLM behavior with easily configurable JSON-based presets.
*   **Vision & Multi-Modal Support:** Experimental image processing via Ollama integration.
*   **Mid-Workflow Conditional Routing:** Implement conditional logic within workflows for dynamic, adaptive processing.
*   **MCP Server Tool Integration:** Experimental support for MCP server tool calling using MCPO.

## Why Choose WilmerAI?

WilmerAI goes beyond simple prompt routing by offering semi-autonomous workflows, empowering users with granular control over LLM interactions. It enables the combination of multiple LLMs for creating better answers, and allowing for more complex and customizable categorization.

WilmerAI enables the power of workflows and enables you to achieve:

*   **Enhanced Categorization:** Route to many LLMs via a whole workflow, rather than just one LLM
*   **Granular Control:** Leverage your domain knowledge and experience to control how LLMs take the path they will
    take, using semi-autonomous workflows.
*   **Flexibility:** Route to many via a whole workflow instead of one, or route directly to any workflow you want to
    use.

## Quick Start

### Installation

1.  **Prerequisites:** Ensure you have Python installed (3.10 and 3.12 recommended).
2.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application:**

    ```bash
    python server.py
    ```

### Configuration

1.  **Endpoints:** Configure your LLM API endpoints in the `Public/Configs/Endpoints` directory.
2.  **Users:** Create a user configuration file in `Public/Configs/Users` or update the `_current-user.json` file to
    select a pre-built user configuration.
3.  **Routing (Optional):** Customize prompt routing in the `Public/Configs/Routing` directory.
4.  **Workflows:** Customize and add the user specific workflow configuration to the `Public/Workflows/` directory, under
    your specific user's folder.
5.  **Connect:** Utilize the endpoints to connect in SillyTavern or Open WebUI.

### Connect in SillyTavern

To connect as a Text Completion:
*   Connect as OpenAI Compatible v1/Completions. Set the parameters as:
    ```text
    Server URL: http://<wilmer-ip>:<wilmer-port>
    API Key: (leave blank)
    Model:  (leave blank)
    ```

OR
*   Connect as Ollama api/generate. Set the parameters as:
    ```text
    Server URL: http://<wilmer-ip>:<wilmer-port>
    API Key: (leave blank)
    Model:  (leave blank)
    ```
    Select WilmerAI compatible prompt template.

For Chat Completion, follow the instructions given at the original repo.

### Connect in Open WebUI

To connect in Open WebUI, connect as if it were an Ollama Instance.
*   Server URL: http://<wilmer-ip>:<wilmer-port>

## Important Notes

*   **Token Usage:** WilmerAI does not track or report token usage. Monitor your LLM API dashboards.
*   **LLM Dependence:** WilmerAI's quality depends on the quality of the connected LLMs, presets, and prompt templates.
*   **Maintenance:** This is a passion project, so updates and bug fixes may take some time.

## Support and Resources

*   **Email:** WilmerAI.Project@gmail.com
*   **YouTube Videos:**
    *   [WilmerAI Setup Tutorial](https://www.youtube.com/watch?v=v2xYQCHZwJM)
    *   [WilmerAI Tutorial Youtube PlayList](https://www.youtube.com/playlist?list=PLjIfeYFu5Pl7J7KGJqVmHM4HU56nByb4X)