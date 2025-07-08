# Agent Development Kit (ADK): Build, Evaluate, and Deploy AI Agents with Ease

**The Agent Development Kit (ADK) is a versatile Python toolkit enabling developers to create and manage sophisticated AI agents with flexibility and control.** ([Original Repository](https://github.com/google/adk-python))

ADK is designed to streamline the development lifecycle, making it easy to build, test, and deploy AI agents, whether you're working with simple tasks or complex, multi-agent workflows.

## Key Features of ADK

*   ✅ **Rich Tool Ecosystem:** Integrate pre-built tools, custom functions, and OpenAPI specifications to equip your agents with diverse capabilities, optimized for the Google ecosystem.
*   🐍 **Code-First Development:** Define agent logic, tools, and orchestration directly in Python for maximum flexibility, testability, and version control.
*   🧱 **Modular Multi-Agent Systems:** Design scalable applications by composing multiple specialized agents into flexible hierarchies.
*   ☁️ **Deploy Anywhere:** Easily containerize and deploy your agents on platforms like Cloud Run or scale seamlessly with Vertex AI Agent Engine.

## 🚀 Installation

### Stable Release (Recommended)

Install the latest stable version using pip:

```bash
pip install google-adk
```

This version is updated weekly and is recommended for most users.

### Development Version

Install the latest changes directly from the main branch on GitHub if you need access to the newest bug fixes and features:

```bash
pip install git+https://github.com/google/adk-python.git@main
```

**Note:** The development version may contain experimental features and potentially unstable code. Use with caution.

## 📚 Documentation

Explore the complete documentation for detailed guidance on building, evaluating, and deploying agents:

*   **[Documentation](https://google.github.io/adk-docs)**

## 🏁 Feature Highlights

### Define a Single Agent

```python
from google.adk.agents import Agent
from google.adk.tools import google_search

root_agent = Agent(
    name="search_assistant",
    model="gemini-2.0-flash", # Or your preferred Gemini model
    instruction="You are a helpful assistant. Answer user questions using Google Search when needed.",
    description="An assistant that can search the web.",
    tools=[google_search]
)
```

### Define a Multi-Agent System

Create complex workflows by composing agents:

```python
from google.adk.agents import LlmAgent, BaseAgent

# Define individual agents
greeter = LlmAgent(name="greeter", model="gemini-2.0-flash", ...)
task_executor = LlmAgent(name="task_executor", model="gemini-2.0-flash", ...)

# Create parent agent and assign children via sub_agents
coordinator = LlmAgent(
    name="Coordinator",
    model="gemini-2.0-flash",
    description="I coordinate greetings and tasks.",
    sub_agents=[ # Assign sub_agents here
        greeter,
        task_executor
    ]
)
```

### Development UI

Utilize the built-in development UI for testing, evaluation, debugging, and showcasing your agent(s).

<img src="https://raw.githubusercontent.com/google/adk-python/main/assets/adk-web-dev-ui-function-call.png"/>

### Evaluate Agents

```bash
adk eval \
    samples_for_testing/hello_world \
    samples_for_testing/hello_world/hello_world_eval_set_001.evalset.json
```

## 🤝 Contributing

We welcome community contributions! Learn how to contribute via these resources:

*   [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/)
*   [Code Contributing Guidelines](./CONTRIBUTING.md)

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

*Happy Agent Building!*