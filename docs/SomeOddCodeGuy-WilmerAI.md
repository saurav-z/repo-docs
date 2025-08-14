# WilmerAI: Expertly Routing Your LLM Inference

**Unlock the power of interconnected LLMs with WilmerAI, a flexible middleware designed to orchestrate complex workflows and elevate your AI interactions.  [Visit the original repo for more details](https://github.com/SomeOddCodeGuy/WilmerAI).**

## Key Features

*   **Dynamic Prompt Routing:**  Direct prompts to custom categories (coding, math, personas, etc.) or leverage a complete workflow override.
*   **Customizable Workflows:**  Design intricate, multi-step workflows to chain LLM calls, integrating tools, and fine-tuning responses.
*   **Parallel LLM Processing:**  Utilize multiple LLMs simultaneously for faster, more robust results.
*   **Offline Wikipedia Integration:** Enhance factual accuracy with seamless integration with the [OfflineWikipediaTextApi](https://github.com/SomeOddCodeGuy/OfflineWikipediaTextApi).
*   **Chat Summaries & Memory:**  Employ continual chat summaries to retain contextual consistency across extensive conversations.
*   **Model Hotswapping:** Maximize VRAM usage with Ollama's hotswapping to run complex workflows on more hardware.
*   **Customizable Presets:** Tailor LLM behavior with flexible JSON-based preset configurations.
*   **Multi-Modal Vision Support (Ollama):** Experiment with image processing via the Ollama API.
*   **Mid-Workflow Conditional Logic:** Dynamically adjust workflows based on LLM responses, enabling branching and adaptive responses.
*   **MCP Tool Server Integration:** Leverage new MCPO features for enhanced tool calling support using MCP server.
*   **Workflow Locks:** Enable asynchronous operations for improved response times by locking certain parts of the workflow while other tasks run.

## Why Choose WilmerAI?

WilmerAI moves beyond the limitations of simple LLM routers by offering powerful, semi-autonomous workflows that allow you to define the precise path your prompts take.  This provides:

*   **Fine-grained control:**  Orchestrate your LLMs.
*   **Enhanced Accuracy:**  Leverage the collective intelligence of multiple LLMs.
*   **Greater Flexibility:**  Adapt to various tasks through custom workflows.
*   **Domain Expertise:** Integrate your own knowledge and experience into the process.

## Quick Start

1.  **Install Dependencies:**  `pip install -r requirements.txt`
2.  **Configure Endpoints:**  Update JSON files in the `Public/Configs/Endpoints` directory with your LLM API details.
3.  **Select a User:**  Choose and set your current user via `Public/Configs/Users/_current-user.json` or the command-line argument.
4.  **Set up Routing:**  Choose a routing file in the `Public/Configs/Routing` directory, or create your own.
5.  **Set up workflows:** Set up the workflow json files in the `Public/Workflows/username` directory

Refer to the original repository's documentation for detailed setup instructions, API endpoint configurations, and example workflows.

---