# WilmerAI: Crafting Intelligent Workflows for LLMs

**Unlock the power of complex, multi-LLM workflows with WilmerAI, a flexible and customizable middleware designed to optimize your language model interactions.** [View the original repo](https://github.com/SomeOddCodeGuy/WilmerAI)

## Key Features:

*   **Advanced Prompt Routing:** Direct your prompts to custom categories or personas, facilitating nuanced interactions.
*   **Customizable Workflows:** Design workflows that precisely orchestrate LLM calls, allowing for iterative refinement and complex task execution.
*   **Multi-LLM Orchestration:** Leverage multiple LLMs concurrently, distributing workloads and maximizing performance.
*   **RAG Support:** Seamless integration with the Offline Wikipedia API and vector database.
*   **Contextual Memory:** Utilize continual chat summaries to maintain context and enhance consistency, even beyond LLM token limits.
*   **Hotswapping:** Utilize Ollama's hotswapping to run multiple models on limited VRAM.
*   **Custom Presets:** Tailor API calls with easily configurable presets.
*   **Vision Capabilities:** Experimental image processing support through Ollama's vision API.
*   **Conditional Workflows:** Implement mid-workflow logic and select sub-workflows dynamically.
*   **Tool Integration:** Experimental tool calling with MCP support.

## What is WilmerAI?

WilmerAI acts as a sophisticated intermediary between your front-end applications and various Large Language Model (LLM) APIs. It exposes OpenAI and Ollama-compatible endpoints, enabling you to connect to a wide range of LLMs, including OpenAI, KoboldCpp, and Ollama. Wilmer processes your prompts through a series of interconnected workflows, where each workflow may call multiple LLMs and other tools. This results in a seamless, single-call experience for users, but with the power of distributed processing and complex task management under the hood.

## Getting Started

### Installation

WilmerAI is a Python-based application.

**Installation:**
*   Ensure you have Python installed (3.10 or 3.12 recommended).
*   **Option 1 (Recommended):** Run the provided `.bat` (Windows) or `.sh` (macOS) scripts, which create a virtual environment and install dependencies.
*   **Option 2 (Manual):**
    1.  Install dependencies: `pip install -r requirements.txt`
    2.  Start the server: `python server.py`

### Configuration

*   All settings are configured via JSON files located in the `Public` folder.
*   When updating, simply copy your `Public` folder to preserve your settings.
*   **Key Files:**
    *   `Endpoints`: Defines LLM API endpoints (e.g., OpenAI, Ollama).
    *   `Users`: Configures user-specific settings, including workflow selection.
    *   `Routing`: Sets up prompt categorization and routing rules.
    *   `Workflows`: Defines the sequence of operations and LLM calls for each task.

### Connecting to WilmerAI

WilmerAI provides several API endpoints to connect to various front-end apps. These include:

*   OpenAI Compatible v1/completions (requires Wilmer Prompt Template)
*   OpenAI Compatible chat/completions
*   Ollama Compatible api/generate (requires Wilmer Prompt Template)
*   Ollama Compatible api/chat

### Setting Up Users

1.  **Create a User:** Duplicate an existing user's JSON file in the `Users` folder and rename it.
2.  **Update `_current-user.json`:** Set your new username in this file or use the `--User` command-line argument.
3.  **Create Routing:** Create a routing configuration file in the `Routing` folder, referencing it in your user's settings.
4.  **Build Workflows:** Create a new folder within the `Workflows` directory that matches your username and add workflows you want to use

### Example Workflows

WilmerAI offers several example workflows, found in the user config files, that provide various methods of running the LLM, including:
*   Single-model conversational assistants
*   Multi-model collaborative agents
*   Roleplaying scenarios
*   Group chats with distinct personas

## Advanced Concepts

*   **Workflows:** Modular JSON-based configurations defining a sequence of LLM calls.
*   **Memory System:** Integrated system for managing contextual information. The system supports long-term, rolling summary, and vector databases.
*   **Presets:** Highly customizable configurations that allow you to modify the parameters used in each of your LLM calls.
*   **Dynamic Prompts:** Employ variable placeholders within prompts (e.g., `{agent1Output}`, `{categoryNameBulletpoints}`) to inject real-time data.

## Important Considerations

*   **Token Usage:** WilmerAI does not track token usage. Monitor your LLM API dashboards for cost control.
*   **LLM Quality:** The quality of WilmerAI's output is heavily dependent on the quality of the LLMs used.

## Support & Resources

*   **YouTube Videos:** Explore the WilmerAI Setup Tutorial and WilmerAI Tutorial YouTube playlist.
*   **Contact:** WilmerAI.Project@gmail.com

## Roadmap

*   Continue improvements to the memory system
*   Expanded documentation