# WilmerAI: Expertly Routing Your Language Model Inferences

🚀 **Unleash the power of semi-autonomous workflows to orchestrate complex LLM interactions and unlock unparalleled control over your AI responses!** This project sits between your frontend and LLM APIs, exposing OpenAI and Ollama compatible API endpoints, and connecting to LLMs like OpenAI, KoboldCpp, and Ollama.  [Explore the WilmerAI repo](https://github.com/SomeOddCodeGuy/WilmerAI).

## Key Features

*   ✅ **Prompt Routing:** Direct prompts to custom categories (coding, math, personas) using user-defined workflows.
*   ✅ **Custom Workflows:** Create tailored workflows for specific tasks, bypassing routing for direct control.
*   ✅ **Multi-LLM Orchestration:** Leverage multiple LLMs in a single workflow to enhance responses.
*   ✅ **Offline Wikipedia API Support:** Integrate with the [OfflineWikipediaTextApi](https://github.com/SomeOddCodeGuy/OfflineWikipediaTextApi) for RAG-based responses.
*   ✅ **Conversation Memory (with Summaries):** Maintains context with continually generated chat summaries, extending beyond LLM context limits.
*   ✅ **Model Hotswapping (Ollama):** Maximize VRAM usage by hotswapping models, enabling complex workflows even on limited hardware.
*   ✅ **Customizable Presets:** Configure LLM behavior through readily customizable JSON files.
*   ✅ **Vision Multi-Modal Support (Ollama):** Process images via Ollama, even when your LLM does not directly support images.
*   ✅ **Mid-Workflow Conditional Flows:** Dynamically branch workflows based on LLM responses.
*   ✅ **MCP Server Tool Integration:** Experimental support for MCP server tool calling.

## Getting Started

### Prerequisites

*   Python (3.10 or 3.12 recommended)
*   [OfflineWikipediaTextApi](https://github.com/SomeOddCodeGuy/OfflineWikipediaTextApi) (Optional)

### Installation

**Option 1: Using Provided Scripts (Recommended)**

- **Windows:** Run the provided `.bat` file.
- **macOS:** Run the provided `.sh` file.
- **linux:** Not tested, but can manually install the requirements as outlined below.

**Option 2: Manual Installation**

1.  Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

2.  Run the server:

    ```bash
    python server.py
    ```

### Configuration

1.  **Endpoints:** Configure LLM API endpoints in the `Public/Configs/Endpoints` directory using pre-built user configs, or build your own.
2.  **User Setup:**
    *   Choose an example user from `Public/Configs/Users` to get started.
    *   Set the desired `_current-user.json` in `Public/Configs/Users`.
3.  **Routing (Optional):** Customize routing behavior within the `Public/Configs/Routing` directory by editing the routing configurations.
4.  **Workflows (Advanced):** Customize and create your workflows in the `Public/Workflows/your_user` directory (e.g., `Public/Workflows/socg`).

### Connecting to Wilmer

*   **OpenAI/Ollama Compatible v1/Completions:** Configure front-end applications (like SillyTavern) to connect as OpenAI/Ollama-compatible text completions. Use a WilmerAI-specific prompt template (see `Docs/SillyTavern/InstructTemplate`).

*   **OpenAI Chat Completions:**  Connect as OpenAI Chat Completions; configure settings for maximum token limit.

*   **Open WebUI:** Simply connect Open WebUI to Wilmer as if it were an Ollama instance.

## Deep Dive: Understanding Workflows, Memories, and Chat Summaries

### Workflows

Workflows organize LLM interactions by defining the flow of prompts, endpoints, and operations.  Each workflow consists of nodes that perform specific actions like:

*   Calling an LLM via an endpoint.
*   Retrieving information from an API, like the offline Wikipedia API.
*   Using tools like generating memories.
*   Executing python modules.

### Memory and Chat Summaries

*   **Recent Memories:** Wilmer creates summaries and stores them, generating  `DiscussionId_memories.json`, along with `[DiscussionId]####[/DiscussionId]` tags within your prompt/system prompt.

*   **Chat Summaries:** Updates the entire conversation by summarizing existing memories in the file `[DiscussionId]_chatsummary.json`.

### Parallel Processing (Advanced)

By using parallel processing, you can divide memory-heavy tasks across multiple LLMs simultaneously.

## Troubleshooting

*   **Memory/Summary Files:** Check your workflows and user settings. Ensure the target folder and files are configured, or create them for first run.

*   **Frontend Responses:** Verify that streaming settings align between Wilmer and your frontend (e.g., SillyTavern).

*   **Preset Errors:**  Check preset configurations in your workflows and make sure your LLM supports them.

*   **Token Length Errors:**  Be mindful of context limits. WilmerAI *does not* have a token limit.
*   **Runtime Errors**:  Check logs, endpoints, and user profiles. If there is an error it could be related to a non-valid API call.

## Additional Information

*   **YouTube Tutorials:** Comprehensive setup tutorials and workflow examples.

    *   [WilmerAI Setup Tutorial](https://www.youtube.com/watch?v=v2xYQCHZwJM)
    *   [WilmerAI Tutorial Youtube PlayList](https://www.youtube.com/playlist?list=PLjIfeYFu5Pl7J7KGJqVmHM4HU56nByb4X)
*   **Documentation:**  See the README for further instructions!

## Contact

For feedback, requests, or questions, reach out to WilmerAI.Project@gmail.com

## Third-Party Libraries & Licensing

See the `ThirdParty-Licenses` folder for information regarding license details.