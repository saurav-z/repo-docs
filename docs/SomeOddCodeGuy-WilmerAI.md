# WilmerAI: Orchestrating LLMs for Advanced AI Workflows

**Transform your AI interactions with WilmerAI, a powerful middleware that routes, manages, and enhances your prompts across multiple Large Language Models (LLMs).**

[Go to the original repo](https://github.com/SomeOddCodeGuy/WilmerAI)

## Key Features

*   **Intelligent Prompt Routing:** Direct prompts to specific categories (coding, math, etc.) or personas for targeted responses.
*   **Customizable Workflows:** Design intricate sequences of LLM calls with full control over the process.
*   **Multi-LLM Orchestration:** Leverage the power of multiple LLMs simultaneously within a single prompt, enhancing output quality.
*   **Contextual Memory:** Utilize integrated memory systems to maintain consistent and context-aware conversations.
*   **RAG with Offline Wikipedia:** Integration with the [Offline Wikipedia API](https://github.com/SomeOddCodeGuy/OfflineWikipediaTextApi) to provide factual RAG.
*   **Vision Multi-Modal Support:** (Experimental) Process images via Ollama integration.
*   **Workflow Conditional Branching:** Dynamically adjust the workflow path based on LLM responses.
*   **Tool Integration:** Experimental support for tool-calling through MCPO integration with [iSevenDays](https://github.com/iSevenDays).
*   **Hot-swapping Models:** Maximize VRAM usage with Ollama's hotswapping capabilities.
*   **Preset Customization**: Configure and tune your LLM interactions with customizable JSON presets.

## What is WilmerAI?

WilmerAI is a sophisticated middleware application that acts as an intermediary between your front-end LLM-calling program (e.g., a chatbot interface) and the various LLM APIs. It offers a flexible and powerful way to manage and orchestrate prompts across multiple LLMs, enabling complex workflows and enhancing the quality of AI-generated responses.

WilmerAI uses custom workflows which allow you to determine what tools and when they are used. The system also allows for iterative LLM calls to improve performance and distributed LLMs.

Wilmer stands for **"What If Language Models Expertly Routed All Inference?"**

## Getting Started

### Requirements

*   Python (3.10 or 3.12 recommended)

### Installation

1.  **Using Provided Scripts (Recommended)**
    *   **Windows:** Run the `.bat` file.
    *   **macOS:** Run the `.sh` file.
2.  **Manual Installation**
    *   Install dependencies: `pip install -r requirements.txt`
    *   Run the server: `python server.py`

### Configuration

WilmerAI is configured primarily through JSON files located in the `Public` folder.

1.  **Endpoints:** Configure your LLM API connections in the `Public/Configs/Endpoints` folder.
2.  **Users:** Create user profiles with settings in `Public/Configs/Users`.
3.  **Workflows:** Design and customize your workflows in `Public/Workflows`.
4.  **Routing:** Define prompt routing rules in `Public/Configs/Routing`.
5.  **Presets:** Customize your LLM calls using JSON presets in the `Public/Configs/Presets` folders.

Detailed documentation on the configuration files can be found in the original [README](https://github.com/SomeOddCodeGuy/WilmerAI).

### Connecting to WilmerAI

WilmerAI exposes API endpoints compatible with:

*   OpenAI v1/completions
*   OpenAI chat/completions
*   Ollama api/generate
*   Ollama api/chat
*   KoboldCpp /api/v1/generate

Connect to WilmerAI from your chosen front-end application using these endpoints.

### Example Setups

*   **SillyTavern:** Connect as OpenAI or Ollama text/chat completions.
*   **Open WebUI:** Connect as an Ollama instance.

## Maintainer's Note

This project is under active development and a passion project supported in free time. Please be aware that fixes for any bugs or issues may take some time.

## Disclaimer

*   This is a personal project in active development. It may contain bugs and is provided "as is" without any warranty.
*   The views and methodologies presented do not reflect those of any employers.

## Contact

For questions, feedback, and collaboration opportunities, please reach out to: WilmerAI.Project@gmail.com

## Third-party Licenses

This project makes use of these third-party libraries:

*   Flask
*   requests
*   scikit-learn
*   urllib3
*   jinja2
*   pillow

Further information on their licensing and use can be found in the README in the ThirdParty-Licenses folder.