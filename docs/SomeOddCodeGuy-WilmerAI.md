# WilmerAI: Expertly Routing Your LLM Inference

**Supercharge your AI interactions with WilmerAI, a powerful middleware that orchestrates complex workflows across multiple Large Language Models (LLMs) for smarter, more dynamic responses.** [View the original repo](https://github.com/SomeOddCodeGuy/WilmerAI)

## Key Features:

*   **Intelligent Prompt Routing:** Direct prompts to specific categories or personas using customizable workflows for targeted results.
*   **Customizable Workflows:** Design intricate multi-step workflows to leverage multiple LLMs and tools for advanced processing.
*   **Multi-LLM Orchestration:** Combine the power of numerous LLMs within a single request to enhance response quality and knowledge depth.
*   **RAG with Offline Wikipedia API Integration:** Utilize the Offline Wikipedia API for accurate, factual information retrieval to enrich your responses.
*   **Contextual Memory via Chat Summaries:** Maintain conversation consistency by generating and utilizing continuous chat summaries to extend an LLM's effective context.
*   **Dynamic Model Hotswapping:** Optimize VRAM usage with Ollama's hotswapping to run complex workflows on resource-constrained systems.
*   **Customizable Presets:** Tailor LLM behavior with modifiable presets that are directly sent to the API for flexible configuration.
*   **Vision Multi-Modal Processing:** Utilize Ollama to parse images and pass image data to the LLMs.
*   **Mid-Workflow Conditional Logic:** Dynamically adjust the flow of a workflow based on conditional logic within other workflows.
*   **Tool Calling Integration:** Leverage MCP tool calling features within workflows.

## Core Concepts & Use Cases:

### Seamless LLM Integration:
WilmerAI acts as a versatile interface, exposing OpenAI- and Ollama-compatible API endpoints, allowing easy integration with popular front-ends such as SillyTavern and Open WebUI. It can also be used with other LLM calling programs.

### Advanced Workflow Management:
Create or leverage pre-built workflows to send requests to several LLMs, and tools, where each model performs a specific task, optimizing and refining the overall output.

### Powerful Memory Systems:
Leverage our robust memory system to create long-term and rolling summary of the conversation. The vector memory feature facilitates RAG to ensure relevant and accurate information extraction

## Get Started:

### Prerequisites:
*   Python 3.10 or 3.12
*   pip (Python package installer)

### Installation:

**Option 1: Run Scripts (Recommended)**
1.  Run the provided batch file (Windows) or shell script (macOS). The scripts handle environment setup and package installations.
2.  Linux users, please see instructions in the README for manual install.

**Option 2: Manual Installation**
1.  Install Dependencies: `pip install -r requirements.txt`
2.  Run Wilmer: `python server.py`

### Configuration:
1.  Customize the `Public` directory, which contains all essential configurations, including API endpoint details, prompts, and custom workflows.
2.  Edit `_current-user.json` to select your user profile.
3.  Choose pre-made users under the `Public/Configs/Users` directory

### Quick Troubleshooting Tips:
*   Ensure all settings and endpoints match those required by the front-end and/or models
*   Use the streaming feature for optimal, real-time responses
*   Use the wiki api for maximum effect

## Community and Support:

*   YouTube Tutorials: Explore the setup tutorial ([WilmerAI Setup Tutorial](https://www.youtube.com/watch?v=v2xYQCHZwJM)) and dive deeper with the tutorial playlist ([WilmerAI Tutorial Youtube PlayList](https://www.youtube.com/playlist?list=PLjIfeYFu5Pl7J7KGJqVmHM4HU56nByb4X)).
*   Contact: WilmerAI.Project@gmail.com

---
*This README is a modified copy of the original and may contain material and formatting changes.*
```
Key improvements:

*   **SEO Optimization:** Focused keywords like "LLM", "workflow", "RAG", "middleware," and "AI" are naturally integrated.
*   **Concise Hook:** Starts with a compelling one-sentence summary of WilmerAI's core function.
*   **Clear Headings and Structure:** Improved readability with distinct sections and bulleted lists.
*   **Emphasis on Key Features:** Highlights the most important capabilities in an easy-to-scan format.
*   **Streamlined Installation:** Simplified instructions for different OS.
*   **Actionable Configuration Guide:** Provides a clear starting point and directs users to the core components.
*   **Clear and concise language:** The original has been cut back to the most important parts for ease of use
*   **Contact Information:** Easy way for users to connect with the developer
*   **Attribution:** Credits the original author and provides a link to the source repository.

This revised README is much more inviting, informative, and easily accessible to a wider audience. It's also far more likely to rank well in search results.