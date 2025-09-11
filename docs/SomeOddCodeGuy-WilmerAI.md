# WilmerAI: Revolutionizing AI Interactions with Contextual Routing and Workflow Orchestration

**Tired of generic AI responses? Discover WilmerAI, a powerful application designed for intelligent prompt routing and complex task orchestration.** ([See the original repo](https://github.com/SomeOddCodeGuy/WilmerAI))

## Key Features

*   **Advanced Contextual Routing:** WilmerAI goes beyond simple keyword matching. It analyzes the **entire conversation history** to understand the true intent behind user requests. This allows it to select the most appropriate specialized workflow for each task.
    *   **Prompt Routing:** Directs user requests to the correct specialized workflow (e.g., "Coding," "Factual," "Creative").
    *   **In-Workflow Routing:** Provides conditional "if/then" logic within workflows to dynamically choose the next step based on previous node outputs.

*   **Core: Node-Based Workflow Engine:** WilmerAI uses workflows, which are JSON files defining a sequence of steps (nodes). Each node performs a specific task, with outputs feeding into the next for complex processes.

*   **Multi-LLM & Multi-Tool Orchestration:** Effortlessly integrate various LLMs and tools within a single workflow. Each node can call a different LLM endpoint or execute a tool, enabling you to optimize task completion by utilizing the best model for each step.

*   **Modular & Reusable Workflows:** Build self-contained workflows for common tasks and execute them as a single node in larger workflows. This streamlines the design of complex agents and simplifies your workflow architecture.

*   **Stateful Conversation Memory:** Leverages a three-part memory system—chronological summary, rolling summary, and vector database—to maintain context over extended conversations and enhance routing accuracy.

*   **Adaptable API Gateway:** Compatible with both OpenAI- and Ollama-style API endpoints, enabling seamless integration with your existing front-end applications and tools.

*   **Flexible Backend Connectors:** Connect to a variety of LLM backends, including OpenAI, Ollama, and KoboldCpp, with a simple configuration system to easily set up endpoints, API types, and generation presets.

*   **MCP Server Tool Integration:** Experimental support for MCPO, offering tool use mid-workflow thanks to [iSevenDays](https://github.com/iSevenDays).

## Why Choose WilmerAI?

WilmerAI offers a superior alternative to traditional semantic routers. It delivers enhanced categorization with customizable user-defined workflows. Instead of relying on unreliable autonomous agents, Wilmer focuses on **semi-autonomous workflows**, giving you the granular control needed to orchestrate your LLMs with ease. Whether it's routing to a single LLM or many, WilmerAI ensures that every interaction is efficient and effective.

## Getting Started

### 1. Installation

WilmerAI is easy to set up:

**Option 1:  Using Provided Scripts**

*   **Windows**: Run the `.bat` file.
*   **macOS**: Run the `.sh` file.

**Option 2: Manual Installation**

1.  Install dependencies: `pip install -r requirements.txt`
2.  Run the server: `python server.py`

### 2. Configuration

To get started:

*   Update the endpoints for your user under Public/Configs/Endpoints. You can find some example characters under the `_example-endpoints` folder. Fill in every endpoint appropriately for the LLMs you are using.
*   Set your current user in Public/Configs/Users/\_current-user.json.

### 3. Connect and start building!

WilmerAI can be connected to applications, such as SillyTavern and Open WebUI. Connect as either OpenAI or Ollama and get started.

## Important Notes

*   This is a project under heavy development. Expect bugs and ongoing updates.
*   **Token Usage:** WilmerAI does not track or report token usage. Monitor your LLM API dashboards.

## Contact

For support, requests, or just to say hello, reach out to:
WilmerAI.Project@gmail.com

---