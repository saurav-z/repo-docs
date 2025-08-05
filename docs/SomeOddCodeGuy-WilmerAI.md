# WilmerAI: Expertly Routing Your Language Model Inferences

**Unlock the power of intelligent workflows and distributed LLMs with WilmerAI, an application that orchestrates complex interactions between your front-end and various LLM APIs.**  [Visit the original repository](https://github.com/SomeOddCodeGuy/WilmerAI).

## Key Features:

*   **Flexible Prompt Routing:** Direct prompts to custom categories (coding, math, personas) for tailored responses.
*   **Customizable Workflows:** Design unique sequences of LLM calls for optimal results, bypassing routing if needed.
*   **Multi-LLM Orchestration:** Leverage multiple LLMs simultaneously within a single workflow for enhanced capabilities.
*   **Offline Wikipedia Integration:** Improve factual responses by connecting to the Offline Wikipedia API.
*   **Contextual Chat Summaries:** Maintain long-term conversation consistency with continuously updated summaries and memories.
*   **Hot-Swappable Models:** Maximize VRAM usage with Ollama's model hotswapping to run complex workflows on limited hardware.
*   **Customizable Presets:** Tailor LLM parameters with easily modifiable JSON configuration files.
*   **Experimental Vision Support:** Enhance prompts with image processing capabilities via Ollama integration.
*   **Conditional Workflow Execution:** Trigger different workflows mid-conversation based on conditional evaluations.
*   **Advanced Tool Calling (MCP):** Leverage MCP server tool calling for powerful and flexible workflow integrations.
*   **Model Agnostic:** WilmerAI is designed to handle any LLM compatible with its API setup.

## What is WilmerAI?

WilmerAI acts as a powerful intermediary between your front-end application and the underlying LLM APIs. By exposing OpenAI and Ollama compatible endpoints, WilmerAI connects to various LLM APIs like OpenAI, KoboldCpp, and Ollama. Users can then construct intricate workflows. The user types a prompt into their frontend, which is then sent to WilmerAI. Here, it is routed through a sequence of customizable workflows, potentially involving multiple LLMs and tools. This allows for sophisticated processing behind the scenes, giving the appearance of a single, streamlined LLM call while in reality, many actions could be taking place.

## Why Use WilmerAI?

WilmerAI offers a flexible solution for managing LLM interactions, built around semi-autonomous workflows that give you granular control over the LLMs' paths. Instead of simply routing to a single LLM, Wilmer routes prompts to comprehensive workflows. This approach enables complex and adaptable categorization, where categorization is handled by user-defined workflows with numerous nodes and LLMs. This allows users to experiment with different prompting styles to optimize results. The goal is to empower users to easily refine and customize their interactions, and to maximize their domain knowledge and experience.

## Setup and Configuration

1.  **Installation:**
    *   Install Python 3.10 or 3.12.
    *   Use the provided `.bat` (Windows) or `.sh` (macOS) files, or manually install dependencies using `pip install -r requirements.txt` and then run `python server.py`.
2.  **Configuration:**
    *   All configurations are handled via JSON files in the `Public` folder.
    *   Essential configurations reside in the `Endpoints` and `Users` folders.
    *   For quick setup, copy a pre-made user configuration from `Public/Configs/Users/` and modify it to your needs, including adjusting the endpoints found under `Public/Configs/Endpoints`.
    *   Update `Public/Configs/Users/_current-user.json` to specify your active user profile, or use the `--User` argument when starting the server.
    *   Tailor your routing and workflows in the `Routing` and `Workflow` folders, creating custom categories and actions.

## Key Concepts:

*   **Endpoints:** Define the LLM API connection details (address, API type, model, and prompt template).
*   **Workflows:** Define the steps, in a JSON file, that will need to take place in order to send prompts to and receive responses from the LLMs.
*   **Users:** Configuration files that set the user-specific settings (port, workflows to run, memory and summary settings, etc.).
*   **Routing:** Choose a workflow based on a request's characteristics.
*   **Presets:** Customizable JSON configuration for settings like temperature, top\_p, and truncation length.
*   **Workflow Nodes**: The individual prompts that get sent to the LLM, from the system and user prompt to the response.
*   **Variables in Prompts:** Allows for many functions in the prompt to send out to the LLM, from the memories and summary to the categories.
*   **Workflow Lock:** Allows a workflow to pause or remain incomplete while a longer process, like memory generation, is taking place.

##  Additional Resources

*   [Setup Tutorial](https://www.youtube.com/watch?v=v2xYQCHZwJM): A walkthrough of downloading and setting up WilmerAI.
*   [WilmerAI Tutorial Playlist](https://www.youtube.com/playlist?list=PLjIfeYFu5Pl7J7KGJqVmHM4HU56nByb4X): In-depth tutorial.
*   [How to Connect to WilmerAI](Docs/README.md#connecting-in-sillytavern)

## Disclaimer

This is an active personal project, so it may contain bugs. Updates may occur later in the evenings or on weekends. WilmerAI is provided "as is" and does not represent the views of any employer.

## Contact

Reach out to WilmerAI.Project@gmail.com for inquiries.