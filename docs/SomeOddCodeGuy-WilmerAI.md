# WilmerAI: Expertly Routing Your Language Model Inferences 

**Unlock the power of multi-LLM workflows and customizable prompt routing with WilmerAI, your gateway to advanced AI applications.** ([Original Repository](https://github.com/SomeOddCodeGuy/WilmerAI))

## Key Features:

*   **Intelligent Prompt Routing:** Categorize and direct prompts to custom domains (coding, math, etc.) or personas for dynamic interactions.
*   **Custom Workflows:** Design bespoke workflows that orchestrate multiple LLMs and tools for complex tasks.
*   **Multi-LLM Orchestration:** Leverage multiple LLMs in a single request, maximizing resource utilization and response quality.
*   **Offline Wikipedia Integration:** Seamlessly incorporate the [OfflineWikipediaTextApi](https://github.com/SomeOddCodeGuy/OfflineWikipediaTextApi) for enhanced factual accuracy.
*   **Persistent Conversation Memory:** Generate continual chat summaries to maintain context across long conversations, facilitating consistency and recall.
*   **VRAM Optimization with Ollama:** Utilize Ollama's hotswapping capabilities to run complex workflows, even on systems with limited VRAM.
*   **Flexible Presets:** Customize LLM configurations through easily editable JSON files, supporting a wide range of models and samplers.
*   **Multi-Modal Support via Ollama:** Process images by sending images to a Ollama backend, even if the LLM itself does not support images, enabling complex vision-language tasks.
*   **Mid-Workflow Conditional Routing:** Dynamically switch to different workflows based on intermediate LLM responses, enabling complex task flows.
*   **MCP Server Tool Integration:** Incorporate tool use using MCPO to your workflow.

## Why Choose WilmerAI?

WilmerAI transcends simple prompt routing. It's a powerful framework designed to optimize LLM interactions through semi-autonomous workflows, empowering users with granular control over LLM pathways, resulting in higher-quality responses and streamlined processing.

## Architecture and Functionality:

WilmerAI acts as a middleware, connecting your frontend (or any LLM-calling application) to various LLM APIs. It exposes standard OpenAI and Ollama compatible API endpoints, while supporting backend connections to OpenAI, KoboldCpp, and Ollama.

### Workflow Overview:

1.  A prompt originates from your frontend.
2.  WilmerAI receives the prompt.
3.  The prompt is routed and processed through a series of modular workflows.
4.  Workflows may involve multiple LLM calls and utilize external tools.
5.  The final response is returned to your frontend.

### Example Workflow Diagram:
*(Insert a visual diagram of a WilmerAI workflow here, e.g., the "Prompt Routing Example" image from the original README, or a simplified representation.)*

## Getting Started:

### Installation:

1.  **Prerequisites:** Ensure Python is installed. Tested with Python 3.10 and 3.12.
2.  **Installation Options:**
    *   **Using Provided Scripts (Recommended):**
        *   **Windows:** Run the `.bat` file.
        *   **macOS:** Run the `.sh` file.
    *   **Manual Installation:**
        ```bash
        pip install -r requirements.txt
        python server.py
        ```

    >   **Note:** The scripts create a virtual environment for isolation.

### Configuration:

*   All configurations are done via JSON files located in the `Public` folder.
*   Copy your `Public` folder to retain settings when updating WilmerAI.

### Key Configuration Steps:

1.  **Endpoints:** Configure your LLM API endpoints in `Public/Configs/Endpoints/`. Refer to `_example-endpoints` for guidance.
2.  **User Setup:**
    *   Create a user configuration file in `Public/Configs/Users/` (e.g., copy and rename an existing one).
    *   Update `Public/Configs/Users/_current-user.json` with your new user's name.
3.  **Routing (Optional):** Define prompt routing rules in `Public/Configs/Routing/`.
4.  **Workflows:** Review and customize workflow configurations in `Public/Workflows/[username]/`.
5.  **Connect:** Set up your application to use WilmerAI via OpenAI or Ollama compatible endpoints.

### Detailed Configuration:
*(Include a brief overview of Endpoint, ApiTypes, PromptTemplates, Users, Routing, Workflow configurations. Refer to the original README for details, summarizing key points.)*

## Advanced Features:

*   **Workflow Locks:** Manage asynchronous operations in multi-LLM setups to prevent race conditions.
*   **Image Processing:** Integrate image understanding using Ollama and vision models.
*   **Conditional Custom Workflows:** Dynamically branch workflows based on intermediate LLM outputs.
*   **Python Module Integration:** Extend functionality with custom Python scripts.
*   **Understanding Memories and Chat Summary:** Learn more about how the functions work, including the nodes: Full Chat Summary Node, Recent/Quality Memory Node, Recent Memory Summarizer Tool, Get Current Chat Summary From File.

## Connecting with Frontends:

*(Provide specific setup instructions for SillyTavern and Open WebUI, including screenshots if possible. Refer to the original README for details.)*

### SillyTavern:

*   Text Completion setup
*   Chat Completion setup

### Open WebUI:
*   Ollama setup

## Troubleshooting:

*(Summarize the Quick Troubleshooting Tips from the original README.)*

## Community & Support:

For feedback, requests, or to connect with the developer, reach out via:  WilmerAI.Project@gmail.com

## License

WilmerAI is licensed under the GNU General Public License v3.

## Third-Party Libraries

*(List the third-party libraries used and their licenses, as provided in the original README.)*