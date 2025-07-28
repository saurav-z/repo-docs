# WilmerAI: Expertly Routing LLMs for Enhanced AI Experiences

**Unleash the power of multiple LLMs and customize your AI workflows with WilmerAI!** ([Original Repo](https://github.com/SomeOddCodeGuy/WilmerAI))

## Key Features

*   🚀 **Advanced Prompt Routing:** Direct prompts to specific categories, personas, or create custom workflows.
*   🤝 **Multi-LLM Synergy:**  Leverage multiple LLMs simultaneously for richer, more diverse responses.
*   📚 **RAG Integration:**  Seamlessly integrate with the Offline Wikipedia API for enhanced factual accuracy.
*   🔄 **Iterative Call Chains:**  Create workflows that refine responses through iterative LLM calls, optimizing performance.
*   🌐 **Distributed Computing:**  Utilize multiple machines and LLMs to create powerful, scalable AI assistants.
*   🧠 **Continual Memory with Chat Summaries:**  Maintain persistent conversation context for extended interactions.
*   ⚙️ **Model Hotswapping with Ollama:** Optimize VRAM usage by leveraging Ollama's model swapping capabilities.
*   🛠️ **Customizable Presets:**  Easily tailor LLM parameters through JSON-based presets.
*   🖼️ **Vision Support with Ollama:** Experimental multi-modal support for images.
*   🔗 **Mid-Workflow Logic:** Conditional workflows and Custom Workflow Nodes for increased versatility.
*   🤖 **Tool Calling with MCP Server Integration:** New and experimental support for MCP server tool calling.

## What is WilmerAI?

WilmerAI is a powerful middleware that sits between your applications and LLM APIs, whether it be front-ends like Open WebUI or SillyTavern, or a Python application or agent. Wilmer uses prompt routing and custom workflows to allow you to send a single prompt through as many LLMs across as many computers as you have access to, and get back a single response. WilmerAI exposes OpenAI and Ollama compatible endpoints, so it should be connectible to most tools and front ends.

## Latest Updates & Maintainer's Note

As of June 29, 2025, WilmerAI now gracefully handles reasoning models, removing thinking blocks during both the workflow and when the user receives their result. This is a major upgrade for modern LLMs. 
Until September 2025, no new Pull Requests will be accepted that modify anything in the Middleware modules. Updates to iSevenDays' new MCP tool calling feature, or adding new custom users or prompt templates within the Public directory, are still welcome.

## Getting Started

### Installation

1.  **Prerequisites:** Python 3.10 or 3.12 is recommended.
2.  **Choose an Installation Method:**
    *   **Option 1 (Recommended):** Run the `.bat` (Windows) or `.sh` (macOS) file to create a virtual environment and install dependencies.
    *   **Option 2 (Manual):**
        ```bash
        pip install -r requirements.txt
        python server.py
        ```

### Quick Configuration Guide

1.  **Endpoints:** Configure your LLM API endpoints in the `Public/Configs/Endpoints` directory.
2.  **User Setup:**
    *   Copy an existing user JSON file from `Public/Configs/Users` and customize it.
    *   Set the `_current-user.json` file to your new user's name.
3.  **Routing (Optional):** Create a routing configuration in `Public/Configs/Routing` and reference it in your user JSON file.
4.  **Workflows:**  Create a folder with your user's name in the `Workflows` folder; this folder will contain the JSON file for each workflow you intend to use.
5.  **Run Wilmer:** Launch `server.py` or the `.bat`/`.sh` files.

## Connecting to WilmerAI

WilmerAI supports various API types for seamless integration:

*   OpenAI Compatible v1/completions (*requires [Wilmer Prompt Template](Public/Configs/PromptTemplates/wilmerai.json)*)
*   OpenAI Compatible chat/completions
*   Ollama Compatible api/generate (*requires [Wilmer Prompt Template](Public/Configs/PromptTemplates/wilmerai.json)*)
*   Ollama Compatible api/chat
*   KoboldCpp Compatible api/v1/generate (*non-streaming generate*)
*   KoboldCpp Compatible /api/extra/generate/stream (*streaming generate*)

### Connecting in SillyTavern

#### Text Completion

*   **Connect as OpenAI Compatible v1/Completions:** [See SillyTavern Settings](Docs/Examples/Images/ST_text_completion_settings.png)
*   **Connect as Ollama api/generate:** [See SillyTavern Settings](Docs/Examples/Images/ST_ollama_text_completion_settings.png)

    Use the [WilmerAI-specific Prompt Template](Docs/SillyTavern/InstructTemplate).

#### Chat Completions (Not Recommended)

*   **Connect as Open AI Chat Completions:** [See SillyTavern Settings](Docs/Examples/Images/ST_chat_completion_settings.png)

    Adjust the truncate length and set "Message Content" under "Character Names Behavior" in Presets.
    Or set `chatCompleteAddUserAssistant` to true.

### Connecting in Open WebUI

Connect to Wilmer as you would an Ollama instance: [See Open WebUI Settings](Docs/Examples/Images/OW_ollama_settings.png)

## Additional Resources
*   [WilmerAI Setup Tutorial](https://www.youtube.com/watch?v=v2xYQCHZwJM)
*   [WilmerAI Tutorial Youtube PlayList](https://www.youtube.com/playlist?list=PLjIfeYFu5Pl7J7KGJqVmHM4HU56nByb4X)

## Important Considerations

*   **Token Usage:** Monitor token usage from your LLM APIs.
*   **LLM Quality:** The quality of your connected LLMs directly impacts WilmerAI's output.

## Troubleshooting

*   **No Memories/Summaries:** Verify the presence of memory/summary nodes in your workflow and the correct file paths.
*   **No Response:** Ensure streaming settings match between Wilmer and the front end.
*   **Preset Errors:** Use presets that are supported by your LLM API.
*   **Memory/Truncate Length Issues:** Be mindful of token limits within Wilmer and your LLMs.
*   **Errors:** Review the console output for error messages.
*   **Stalled Responses:** Confirm the correct endpoint address, port, and active user.

## Contact

For support, feedback, or inquiries, contact: WilmerAI.Project@gmail.com