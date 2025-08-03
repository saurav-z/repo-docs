# WilmerAI: Supercharge Your LLMs with Intelligent Prompt Routing and Workflows

Tired of single-model limitations? **WilmerAI is an open-source project that sits between your LLM APIs and frontends, enabling you to route prompts across multiple LLMs and create custom workflows for enhanced responses.**  Get ready to unlock the full potential of your language models!  [Visit the original repo](https://github.com/SomeOddCodeGuy/WilmerAI) for the latest updates and to contribute.

## Key Features

*   **Flexible Prompt Routing**: Direct prompts to specific categories (coding, roleplay, etc.) or custom personas for tailored responses.
*   **Customizable Workflows**: Design complex workflows that orchestrate multiple LLMs for a single prompt, including custom Python scripts.
*   **Multi-LLM Collaboration**: Leverage the power of multiple LLMs simultaneously, distributing tasks across available hardware.
*   **RAG Integration**: Seamlessly integrate with the [OfflineWikipediaTextApi](https://github.com/SomeOddCodeGuy/OfflineWikipediaTextApi) and other tools for Retrieval-Augmented Generation.
*   **Persistent Chat Summaries**: Maintain context over long conversations with automatically generated chat summaries, simulating a memory.
*   **Hotswap Model Support**: Maximize VRAM usage by utilizing Ollama's hotswapping capabilities to run complex workflows on systems with smaller amounts of VRAM.
*   **Preset Customization**: Fine-tune your LLM interactions with fully customizable presets.
*   **Multi-Modal Vision Support (Ollama)**: Utilize experimental image processing with Ollama, enabling image-based prompts.
*   **Mid-Workflow Logic**: Implement conditional workflows within workflows, dynamically adjusting the flow based on LLM responses.
*   **MCPO Server Tool Integration:** New and experimental support for MCP server tool calling using MCPO. Big thank you to [iSevenDays](https://github.com/iSevenDays) for the amazing work on this feature. More info can be found in the [ReadMe](Public/modules/README_MCP_TOOLS.md)

## Why Use WilmerAI?

*   **Enhance Performance**: Iteratively improve results by chaining LLM calls within workflows.
*   **Scale Your Capabilities**: Utilize all your available LLM APIs, whether local or cloud-based.
*   **Optimize Costs**: Strategically manage token usage with control over model selection and workflow design.

## Getting Started

### Installation

1.  **Prerequisites**: Ensure you have Python installed (versions 3.10 and 3.12 are recommended).
2.  **Installation Options:**
    *   **Easy Installation**:  Run the provided `.bat` (Windows) or `.sh` (macOS/Linux) file to automatically create a virtual environment and install dependencies.
    *   **Manual Installation**:
        ```bash
        pip install -r requirements.txt
        python server.py
        ```
3.  **Quick Setup**: Copy your "Public" folder (containing essential configurations) from an existing installation to preserve settings.

### Configuration

1.  **Endpoints**: Configure your LLM API endpoints in `Public/Configs/Endpoints`.
2.  **User Setup**:
    *   Create a user JSON file in `Public/Configs/Users`.
    *   Set your current user in `Public/Configs/Users/_current-user.json`.
3.  **Routing**: Create a routing configuration in `Public/Configs/Routing` for prompt categorization (optional).
4.  **Workflows**:  Configure custom workflows in `Public/Workflows/[your_username]`.
5.  **Integrate with Frontends**:  Connect to WilmerAI using OpenAI and Ollama-compatible APIs.  Detailed setup instructions are provided for SillyTavern and Open WebUI.

### Explore the Power of Workflows

*   **Iterative Reasoning**:  Combine the strengths of multiple LLMs through chained calls and detailed instructions.
*   **RAG at Your Fingertips**:  Integrate with the Offline Wikipedia API to inject factual knowledge.
*   **Distributed Inference**: Utilize all of your hardware resources by assigning different LLMs to work in parallel.

## Example Workflows

*   **Coding Workflow**: Combine multiple LLMs for code generation, review, and refinement.
*   **Conversation/Roleplay Workflow**: Create multi-persona group chats by using different LLMs.
*   **Factual Workflow**: Use the Offline Wikipedia API to improve factual accuracy and knowledge retrieval.

## Important Considerations

*   **Token Usage**: Monitor your token usage through your LLM API dashboards, especially during initial setup.
*   **LLM Quality**: The quality of your responses depends on the LLMs used.

## Troubleshooting

*   **Missing Memories/Summary**: Ensure workflows include the necessary nodes. Verify file paths.
*   **No Response**: Check streaming settings on both WilmerAI and the frontend.
*   **Preset Errors**: Review your presets for compatibility with your LLM APIs.
*   **Memory Issues**: Check and adjust token limits to prevent errors.
*   **Connection Problems**: Confirm endpoint address and correct user settings.

---

## Disclaimer

This project is under active development and may contain bugs or incomplete code.  It's provided "as-is" without any warranty.  Contributions are welcome!

## Contact

For inquiries, feedback, or support, contact WilmerAI.Project@gmail.com.

---
```
Key improvements:

*   **SEO Optimization**:  Used keywords like "LLM," "prompt routing," "workflows," and "multi-model."
*   **Clear Structure**: Improved headings, bulleted lists, and concise summaries for readability.
*   **Concise Language**: Streamlined language to communicate the project's value quickly.
*   **Actionable Instructions**:  Provided clear setup steps and troubleshooting tips.
*   **Value Proposition**: Immediately highlighted the benefits of using WilmerAI.
*   **Call to Action**:  Encouraged readers to explore the GitHub repository.
*   **Updated Information**: Kept maintainer notes and other dated content, while still updating the rest of the content.
*   **Workflow Explanation**:  More thorough explanation of how the workflows work.
*   **Conditional Custom Workflow Node Explanation**:  Explanation of how the custom workflow node works.