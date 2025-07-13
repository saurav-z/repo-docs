# Agent Development Kit (ADK): Build, Evaluate, and Deploy AI Agents with Ease

**[Explore the Agent Development Kit on GitHub](https://github.com/google/adk-python)** to unlock the power of AI agents with a flexible, code-first toolkit.

<div align="center">
  <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256" alt="Agent Development Kit Logo">
</div>

The Agent Development Kit (ADK) is an open-source Python toolkit designed to empower developers in building, evaluating, and deploying sophisticated AI agents. While optimized for Google's ecosystem, ADK is built to be model and deployment agnostic, allowing developers to create and orchestrate agentic architectures from simple tasks to complex workflows.

## 🌟 Key Features

*   **Rich Tool Ecosystem:** Integrate a variety of tools, including pre-built options, custom functions, and OpenAPI specs, to expand your agent's capabilities, with seamless Google ecosystem integration.
*   **Code-First Development:** Define agent logic, tools, and orchestration directly in Python for maximum flexibility, testability, and version control.
*   **Modular Multi-Agent Systems:** Design scalable applications by composing multiple specialized agents into flexible hierarchies.
*   **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale seamlessly with Vertex AI Agent Engine.
*   **Built-in Development UI:** A built-in UI to test, evaluate, debug, and showcase agents.
*   **Evaluate Agents:** Easily assess agent performance using built-in evaluation tools.

## 🚀 Installation

### Stable Release (Recommended)

Install the latest stable version using pip:

```bash
pip install google-adk
```

This is the recommended approach for most users, providing the most recent official release with weekly updates.

### Development Version

To access the latest bug fixes and features before they are officially released, install directly from the `main` branch:

```bash
pip install git+https://github.com/google/adk-python.git@main
```

**Note:** The development version may include experimental features or bugs. Use it for testing or accessing critical fixes not yet available in the stable release.

## 📚 Documentation

Access comprehensive documentation for detailed guides and examples:

*   **[ADK Documentation](https://google.github.io/adk-docs)**

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

<img src="https://raw.githubusercontent.com/google/adk-python/main/assets/adk-web-dev-ui-function-call.png" alt="ADK Development UI Example">

### Evaluate Agents

```bash
adk eval \
    samples_for_testing/hello_world \
    samples_for_testing/hello_world/hello_world_eval_set_001.evalset.json
```

## 🤖 Agent2Agent (A2A) Protocol Integration

ADK seamlessly integrates with the [A2A protocol](https://github.com/google-a2a/A2A/) for remote agent-to-agent communication.  See this [example](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents) to see how it works.

## 🤝 Contributing

We welcome contributions! Please review our:

*   [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/)
*   [Code Contributing Guidelines](./CONTRIBUTING.md)

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.