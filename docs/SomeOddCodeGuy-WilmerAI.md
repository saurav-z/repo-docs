# WilmerAI: Supercharge Your LLMs with Customizable Workflows

**Tired of limited LLM responses?** WilmerAI empowers you to orchestrate complex workflows, route prompts intelligently, and leverage multiple LLMs to generate superior results.  [Explore the original repository](https://github.com/SomeOddCodeGuy/WilmerAI) for a deeper dive into this powerful project.

## Key Features

*   **Intelligent Prompt Routing**: Direct prompts to specific LLMs or categories (coding, math, personas, etc.) using customizable workflows.
*   **Customizable Workflows**: Design workflows tailored to your exact needs, chaining LLM calls, and incorporating external tools.
*   **Multi-LLM Orchestration**:  Harness the power of multiple LLMs simultaneously for comprehensive and nuanced responses.
*   **Offline Wikipedia Integration**: Enhance factual accuracy using the Offline Wikipedia API for RAG (Retrieval Augmented Generation).
*   **Contextual Chat Summaries**: Maintain conversational memory with automatically generated summaries that extend beyond the typical LLM context window.
*   **VRAM Optimization with Ollama**: Utilize Ollama's hotswapping to run complex workflows on systems with limited VRAM.
*   **Flexible Preset Management**: Easily configure and modify LLM parameters through customizable JSON presets.
*   **Ollama Multi-Modal Support**: Experimentally process images alongside text with Ollama.
*   **Mid-Workflow Conditional Logic**:  Make dynamic decisions in your workflows, branching based on LLM responses.
*   **MCP Server Tool Integration**: Leveraging the power of MCPO, you can use tool calling in a mid-workflow environment.

## Getting Started

### Installation

1.  **Ensure Python is installed.**  The project has been tested with Python 3.10 and 3.12.
2.  **Install dependencies:** `pip install -r requirements.txt`
3.  **Run the server:** `python server.py` (or use the provided `.bat` / `.sh` scripts).

### Quick Setup

1.  **Configure Endpoints:** Modify JSON files in `Public/Configs/Endpoints` to connect to your LLM APIs.
2.  **Choose a User:**  Set the `_current-user.json` file or use the `--User` command-line argument.
3.  **Customize Routing:**  Create or select a routing config in `Public/Configs/Routing`.
4.  **Adjust Workflows:** Tailor workflows within the `Public/Workflows/<your_username>` directory to your needs.

## Advanced Topics

### Understanding Workflows

*   **Nodes**: Workflows are composed of nodes, each performing a specific action (LLM calls, script execution, etc.).
*   **Variables**: Utilize variables (e.g., `{agent1Output}`) to pass data between nodes and customize prompts.
*   **Memory System**: Leverage a sophisticated memory system to maintain context, enabling long-running and insightful conversations.

### Memory

*   **Long-Term Memory (File-Based)**: Create and access summaries of past conversation chunks to improve coherence.
*   **Rolling Chat Summary**: A continuous summarization of the entire conversation for quick context.
*   **Vector Memory (RAG)**:  Enable intelligent keyword search to find the most relevant information in your conversations.

### Quick Troubleshooting

*   **No Memory Files**: Verify the creation of memory files based on your workflow configuration and the presence of the `[DiscussionId]` tag.
*   **No Responses**: Ensure correct streaming settings (matching Wilmer and your front-end).
*   **LLM Errors**: Check presets for API compatibility.
*   **Out of Memory/Truncation Errors**:  Manage token limits to fit within your LLM's context window.

## Further Information

*   [WilmerAI Setup Tutorial](https://www.youtube.com/watch?v=v2xYQCHZwJM)
*   [WilmerAI Tutorial Youtube PlayList](https://www.youtube.com/playlist?list=PLjIfeYFu5Pl7J7KGJqVmHM4HU56nByb4X)

## Important Notes

*   This is a personal project under active development; expect potential bugs and ongoing improvements.
*   Token usage is not tracked within WilmerAI; monitor usage through your LLM API dashboards.
*   LLM quality significantly impacts WilmerAI's performance; choose reliable and high-quality models.

---

## Contact

For questions, suggestions, or collaboration: WilmerAI.Project@gmail.com