# WilmerAI: Expertly Routing Language Models for Enhanced AI Experiences

**Unleash the power of multiple LLMs by routing and orchestrating prompts to create sophisticated AI assistants.** [View the original repo](https://github.com/SomeOddCodeGuy/WilmerAI)

WilmerAI is a versatile middleware designed to connect your favorite applications and frontends, like Open WebUI and SillyTavern, with your LLM APIs, enabling you to create complex workflows and tap into the power of distributed AI.

## Key Features:

*   **Prompt Routing:** Direct prompts to specific categories (coding, math, etc.) or personas for tailored responses.
*   **Custom Workflows:** Design workflows that cater to specific requirements.
*   **Multi-LLM Integration:** Leverage multiple LLMs within a single prompt, utilizing their strengths in tandem.
*   **RAG Support:** Integrate RAG setups with the Offline Wikipedia API and other retrieval methods.
*   **Continuous Memory:** Maintain consistent conversations with continually updated chat summaries, simulating a long-term memory.
*   **Hotswap Models:** Optimize VRAM usage and run diverse model setups leveraging Ollama's hotswapping.
*   **Customizable Presets:** Tailor configurations to your specific LLM API requirements via readily customizable JSON files.
*   **Vision Multi-Modal Support:** Enhance your workflows with image processing when using Ollama as the front-end API.
*   **Mid-Workflow Control:** Dynamic workflow execution based on conditions using the Conditional Custom Workflow Node.
*   **MCP Tool Calling Integration:** Experimental tool calling capabilities using MCPO.

## What Makes WilmerAI Powerful?

*   **Workflows for RAG:** Enhance your LLM responses by connecting it to sources like the Offline Wikipedia API and other retrieval methods.
*   **Iterative LLM Calls:** Fine-tune performance with workflows that incorporate iterative LLM calls.
*   **Distributed LLMs:** Combine the power of multiple LLMs, either through APIs or your own hardware, for complex tasks.

## Quick Start:

1.  **Install:**  Follow the installation instructions in the original README.
2.  **Configure Endpoints:** Set up the endpoints to connect to your chosen LLM APIs.
3.  **Define Users:** Create or modify user configurations with your desired routing and workflow settings.
4.  **Set Routing:** Define the appropriate routing logic to categorize your prompts.
5.  **Build Workflows:** Design, edit, and customize powerful, multi-step workflows for various tasks.
6.  **Connect & Use:** Connect to Wilmer via the supported API endpoints, and start crafting complex AI experiences.

## Advanced Features:

*   **Workflows:**  Define and customize a variety of workflows.
*   **Image Processor:**  Process images with LLMs.
*   **Custom Workflow Node:** Execute other workflows within a workflow.
*   **Conditional Custom Workflow Node:** Selectively execute sub-workflows based on runtime conditions.
*   **Workflow Lock:** Lock workflows for better performance
*   **Memory:** Enhance the LLM by summarizing the entire conversation up to the current point.
*   **Parallel Processing:** Distribute the workload across multiple LLMs.

## Important Notes & Disclaimer:

This is a personal project under active development. The software is provided "as-is" without warranty. The project and its content reflect contributions made in free time and do not represent the maintainer's employers. **Token usage is not tracked by WilmerAI; please monitor your LLM API usage for cost management.**

## Reach Out

Have questions, feedback, or just want to say hello?  Contact me at:
WilmerAI.Project@gmail.com

[See Original Repo](https://github.com/SomeOddCodeGuy/WilmerAI)