# WilmerAI: Your Intelligent LLM Workflow Orchestrator

**Seamlessly connect your frontend to multiple LLMs and unlock advanced workflows with WilmerAI, giving you unprecedented control over your AI interactions.** [View the original repo](https://github.com/SomeOddCodeGuy/WilmerAI)

## Key Features

*   **Advanced Prompt Routing:** Direct prompts to specific domains or personas using customizable routing configurations.
*   **Customizable Workflows:** Build intricate workflows using multiple LLMs and tools, tailoring AI responses to your exact needs.
*   **Multi-LLM Orchestration:** Leverage multiple LLMs in a single call, achieving complex tasks through distributed processing.
*   **Offline Wikipedia Integration:** Enhance factual responses with the Offline Wikipedia API for improved RAG capabilities.
*   **Contextual Chat Summarization:** Generate continuous chat summaries to maintain long-running conversations beyond LLM context limits.
*   **Model Hotswapping with Ollama:** Optimize VRAM usage and run complex workflows on systems with limited resources.
*   **Preset Customization:** Easily modify and adapt your configurations using customizable preset files.
*   **Image Processing via Ollama:** Experiment with image processing using Ollama as a frontend.
*   **Mid-Workflow Conditional Workflows:** Design workflows that adapt based on LLM responses, enabling dynamic decision-making.
*   **MCP Server Tool Integration (Experimental):** Utilize server-side tool calling with MCP server integration.

## About WilmerAI

WilmerAI (What If Language Models Expertly Routed All Inference?) is a project focused on building sophisticated workflows that harness the power of multiple LLMs to enhance AI-driven applications. Launched in late 2023, Wilmer evolved from a fine-tuning router to a robust workflow engine, enabling users to orchestrate complex tasks, improve response quality, and integrate various tools. The project is actively maintained and continually enhanced to meet the evolving demands of the AI landscape.

## Quick Start

### Step 1: Installation

1.  **Option 1 (Recommended):** Run the provided `.bat` (Windows) or `.sh` (macOS) files, or the provided `.py` file. These scripts will create a virtual environment and install all dependencies.
2.  **Option 2 (Manual):**
    *   Install dependencies: `pip install -r requirements.txt`
    *   Run the program: `python server.py`

### Step 2: Configuration

1.  **Endpoints:** Configure your LLM API endpoints in the `Public/Configs/Endpoints` directory.
2.  **Users:** Choose a pre-made user or create a new user configuration in the `Public/Configs/Users` directory and update the `_current-user.json` file accordingly.

## Troubleshooting & Tips

*   **Memory & Summary Files:** Verify workflow nodes in your workflow.
*   **Front-End Compatibility:** Ensure streaming settings match between WilmerAI and your front-end.
*   **Preset Compatibility:** Use presets that are designed for the API.
*   **Troubleshooting:** Check the troubleshooting tips in the original README to address issues.

## Important Notes

*   WilmerAI does not track token usage. Monitor your API dashboards.
*   The quality of outputs depends heavily on connected LLMs and presets.

---

**Get started with WilmerAI today and transform the way you interact with LLMs!**