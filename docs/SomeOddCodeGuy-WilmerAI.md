# WilmerAI: Expertly Routing Language Model Inference

**Unlock the power of interconnected LLMs with WilmerAI, a flexible and customizable application that orchestrates your AI workflows.** [Visit the original repo for more details.](https://github.com/SomeOddCodeGuy/WilmerAI)

## Key Features

*   **Customizable Workflows:** Design intricate workflows to chain multiple LLMs, tools, and resources for complex tasks.
*   **Prompt Routing:** Direct prompts to specific categories or personas, utilizing tailored presets and workflows.
*   **Distributed LLM Support:** Leverage multiple LLMs, including local and cloud-based APIs, to work together.
*   **Flexible API Endpoints:** Compatible with OpenAI, Ollama, and KoboldCpp APIs for seamless integration with various frontends.
*   **Dynamic Memory Management:** Continuously generate chat summaries and recent memories to enhance context and consistency.
*   **Offline Wikipedia Integration:** Utilize the Offline Wikipedia API for enhanced factual recall and RAG capabilities.
*   **Vision Multi-Modal Support:** Experimental image processing using Ollama, enabling image analysis and integration into conversations.
*   **Mid-Workflow Logic:** Implement conditional workflows and execution paths based on LLM responses.
*   **Tool Integration**: Experimental integration with MCP server for advanced tool usage.
*   **Model Hotswapping**: Utilizing Ollama's hotswapping, users can run complex workflows even on systems with smaller amounts of VRAM

## How WilmerAI Works

WilmerAI acts as an intelligent intermediary between your frontend applications (like SillyTavern or Open WebUI) and LLM APIs. It exposes OpenAI and Ollama compatible endpoints, while on the backend, it connects to various LLM APIs such as OpenAI, KoboldCpp, and Ollama. The user can type a prompt into your front end, which is connected to Wilmer, which will then send this prompt to a series of workflows. Each workflow may make calls to multiple LLMs, after which the final response comes backs to you. From your perspective, it looked like a (likely long running) one shot call to an LLM. But in reality, it could be many LLM and even tools doing complex work.

**Think of it as a command center for your LLMs.**

### Example:
1.  **Frontend Input:** You enter a prompt into your chosen application.
2.  **Wilmer's Role:** Wilmer receives the prompt.
3.  **Workflow Execution:** Wilmer routes the prompt through a series of pre-defined workflows.
4.  **LLM Interactions:** The workflows execute, potentially calling multiple LLMs (OpenAI, KoboldCpp, Ollama) and tools.
5.  **Output:** The final, enhanced response is returned to your frontend.

## Why Use WilmerAI?

WilmerAI was created to maximize the potential of Language Models by enabling users to utilize multiple models, tools, and resources to accomplish complex tasks. Wilmer's architecture is focused on enabling flexible and user-defined workflows.

## Getting Started

1.  **Installation:** Follow the instructions in the original [README](https://github.com/SomeOddCodeGuy/WilmerAI) to set up WilmerAI using either the provided scripts or manual installation. Python is required.
2.  **Configuration:**
    *   **Endpoints:** Configure your LLM API endpoints within the `Public/Configs/Endpoints` directory.
    *   **Users:** Set up user profiles in the `Public/Configs/Users` directory, specifying routing, workflows, and settings.
    *   **Routing:** Define prompt routing rules within the `Public/Configs/Routing` directory.
    *   **Workflows:** Customize workflows within the `Public/Workflows` directory to define the flow of interactions between LLMs.
3.  **Connect:** Connect your chosen frontend application (SillyTavern, Open WebUI, etc.) to WilmerAI via the available API endpoints (OpenAI or Ollama compatible).

## Further Exploration

*   **YouTube Tutorials:** Explore the available [WilmerAI Setup Tutorial](https://www.youtube.com/watch?v=v2xYQCHZwJM) and the comprehensive [WilmerAI Tutorial Youtube PlayList](https://www.youtube.com/playlist?list=PLjIfeYFu5Pl7J7KGJqVmHM4HU56nByb4X) for in-depth guidance and examples.
*   **Troubleshooting:** Consult the original README's troubleshooting section for common issues and solutions.
*   **Configuration Files:** Dive into the configuration files to understand how to customize endpoints, users, routing, and workflows.

---

## Disclaimer

This project is under active development and provided "as is" without warranty. The maintainer and contributors are not liable for any issues arising from its use.

---

## Contact

For inquiries, feedback, or support, contact:

WilmerAI.Project@gmail.com