# Agent Development Kit (ADK): Build, Evaluate, and Deploy AI Agents with Ease

**Unleash your creativity and build sophisticated AI agents with the flexible and code-first [Agent Development Kit (ADK)](https://github.com/google/adk-python) from Google!**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python Unit Tests](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml/badge.svg)](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/google/adk-python)

<div align="center">
  <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256" alt="Agent Development Kit Logo"/>
</div>

ADK is a powerful, open-source Python toolkit designed to streamline the entire AI agent lifecycle, from development and evaluation to deployment. Whether you're building simple task-oriented agents or complex multi-agent systems, ADK offers the flexibility and control you need. While optimized for Gemini and the Google ecosystem, ADK is model-agnostic and deployment-agnostic.

## ✨ Key Features

*   **Code-First Development:** Define agent logic, tools, and orchestration directly in Python, enabling version control, testing, and seamless integration with your existing development workflows.
*   **Rich Tool Ecosystem:** Leverage a broad range of pre-built tools, easily integrate custom functions and OpenAPI specifications, and seamlessly connect to the Google ecosystem.
*   **Modular Multi-Agent Systems:** Design and build scalable applications by composing multiple specialized agents into flexible, interconnected hierarchies.
*   **Deploy Anywhere:** Effortlessly containerize and deploy your agents on Cloud Run or scale them using Vertex AI Agent Engine.
*   **Built-in Development UI:** Test, evaluate, and debug your agents with a user-friendly UI.
*   **Agent Evaluation:** Evaluate your agent using the ADK evaluation tool to track metrics.

## 🤖 Agent2Agent (A2A) Protocol and ADK Integration

ADK seamlessly integrates with the [A2A protocol](https://github.com/google-a2a/A2A/) for secure and robust remote agent-to-agent communication.

See this [example](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents) for how they can work together.

## 🚀 Installation

### Stable Release (Recommended)

Install the latest stable version of ADK using `pip`:

```bash
pip install google-adk
```

This is the recommended method for most users, as it provides the most recent official release.

### Development Version

To access the latest bug fixes and features not yet released, install directly from the `main` branch:

```bash
pip install git+https://github.com/google/adk-python.git@main
```

**Note:** The development version may contain experimental changes or bugs. Use it for testing or accessing critical fixes before official releases.

## 📚 Documentation

Comprehensive documentation is available to guide you through the process of building, evaluating, and deploying agents:

*   **[Documentation](https://google.github.io/adk-docs)**

## 🏁 Feature Highlights

### Define a Single Agent:

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

### Define a Multi-Agent System:

Create sophisticated systems by composing multiple agents:

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

<img src="https://raw.githubusercontent.com/google/adk-python/main/assets/adk-web-dev-ui-function-call.png" alt="ADK Development UI"/>

A built-in development UI to help you test, evaluate, debug, and showcase your agent(s).

### Evaluate Agents

```bash
adk eval \
    samples_for_testing/hello_world \
    samples_for_testing/hello_world/hello_world_eval_set_001.evalset.json
```

## 🤝 Contributing

We welcome contributions! Review the following guidelines:

*   [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/)
*   [Code Contributing Guidelines](./CONTRIBUTING.md)

## Vibe Coding

Use [llms.txt](./llms.txt) or [llms-full.txt](./llms-full.txt) as context for your LLM during Vibe coding.

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

*Happy Agent Building!*