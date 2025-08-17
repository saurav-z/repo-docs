# WilmerAI: Expertly Routing Language Model Inference for Enhanced AI Workflows

> **Unleash the power of advanced AI workflows with WilmerAI, a versatile application designed to orchestrate and optimize interactions with various Language Models, enabling complex, multi-step AI tasks.** [Visit the original repository](https://github.com/SomeOddCodeGuy/WilmerAI)

## Key Features

*   **Prompt Routing:** Dynamically route prompts to tailored workflows based on category or persona.
*   **Custom Workflows:** Design multi-step workflows using multiple LLMs and tools.
*   **Multi-LLM Orchestration:** Leverage multiple LLMs in a single workflow for complex tasks.
*   **Offline Wikipedia Integration:** Utilize the [Offline Wikipedia API](https://github.com/SomeOddCodeGuy/OfflineWikipediaTextApi) for enhanced factual responses.
*   **Contextual Chat Summaries:** Generate continual summaries for maintaining context in long conversations.
*   **Hotswapping Models:** Maximize VRAM usage by leveraging Ollama's hotswapping for complex multi-model workflows.
*   **Configurable Presets:** Customize model parameters via JSON files for various LLM types.
*   **Multi-Modal Vision Support:** Experimental support of image processing with Ollama.
*   **Mid-Workflow Conditional Execution:** Kick off new workflows based on conditions, enabling flexible, dynamic responses.
*   **MCP Server Tool Integration:** Experimental support for tool calling mid-workflow, allowing flexible prompt responses.

## Getting Started

WilmerAI sits between your front-end (or any other LLM calling program) and various LLM APIs (OpenAI, Ollama, KoboldCpp, etc.), routing prompts through customizable workflows.

### 1. Installation

*   **Option 1: Using Scripts**
    Run the included `.bat` file (Windows) or `.sh` file (macOS) to automatically set up a virtual environment and install dependencies.
*   **Option 2: Manual Installation**
    1.  Install dependencies: `pip install -r requirements.txt`
    2.  Run the server: `python server.py`

### 2. Configuration

*   **Endpoints:** Configure your LLM API endpoints in the `Public/Configs/Endpoints` folder (example endpoints are available in the `_example-endpoints` folder).
*   **Users:** Create a user configuration file in `Public/Configs/Users/` (copy and rename an existing file), and specify the active user in `Public/Configs/Users/_current-user.json`.
*   **Routing:** Configure prompt routing categories and workflows in the `Public/Configs/Routing` folder.
*   **Workflows:** Design your multi-step workflows in `Public/Workflows/{username}/` with the appropriate endpoints and parameters.

### 3. Connecting Your Front-End

WilmerAI exposes OpenAI and Ollama compatible API endpoints. Connect your front-end application to Wilmer using the appropriate settings.

*   **Text Completion:** Connect to the OpenAI Compatible v1/completions or Ollama api/generate endpoints. You may need a specific prompt template (example provided in `Docs/SillyTavern/InstructTemplate`).
*   **Chat Completion:** Connect to OpenAI Compatible chat/completions.

### 4. Advanced Features

*   **Workflow Locks:** Employ workflow locks to manage asynchronous operations, especially in multi-model setups.
*   **Custom Workflows:** Execute entire workflows within other workflows, for modular and complex logic.
*   **Image Processing:** Leverage the ImageProcessor node with Ollama for multimodal workflows.
*   **Python Module Caller:** Use Python to extend Wilmer's capabilities.

## Resources

*   **YouTube Tutorials:**  Setup and usage tutorials.
    *   [WilmerAI Setup Tutorial](https://www.youtube.com/watch?v=v2xYQCHZwJM)
    *   [WilmerAI Tutorial Youtube PlayList](https://www.youtube.com/playlist?list=PLjIfeYFu5Pl7J7KGJqVmHM4HU56nByb4X)
*   **Documentation:**  Consult the README for detailed explanations of configurations, workflow types, and troubleshooting tips.

## Disclaimer

This is a personal project under active development. Please note that this software is provided "as is" without any warranty.

## Contact

For feedback and support, contact: WilmerAI.Project@gmail.com