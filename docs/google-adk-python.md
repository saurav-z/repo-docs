# Agent Development Kit (ADK): Build Powerful AI Agents with Ease

**Empower your AI agent development with the Agent Development Kit (ADK), a versatile, open-source Python toolkit from Google for crafting, evaluating, and deploying sophisticated AI agents.** ([View the original repo](https://github.com/google/adk-python))

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python Unit Tests](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml/badge.svg)](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/google/adk-python)

<div align="center">
  <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256"/>
</div>

The Agent Development Kit (ADK) is a flexible and modular framework designed to streamline the development and deployment of AI agents. Built with a code-first approach, it provides developers with unparalleled control and flexibility for building agentic architectures, from simple tasks to complex, multi-agent systems.

### Key Features:

*   **Rich Tool Ecosystem:** Integrate a wide array of tools, including pre-built tools, custom functions, and OpenAPI specifications, offering agents diverse capabilities.
*   **Code-First Development:** Define agent logic, tools, and orchestration directly in Python for ultimate flexibility, testability, and version control.
*   **Modular Multi-Agent Systems:** Design scalable applications by composing multiple specialized agents into flexible hierarchies.
*   **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale seamlessly with Vertex AI Agent Engine.
*   **Built-in Development UI:** Test, evaluate, debug, and showcase your agent(s) with the built-in development UI.
*   **Evaluation Tools:** Evaluate your agent's performance with built-in evaluation tools.

---

## 🤖 Agent2Agent (A2A) Protocol Integration

ADK seamlessly integrates with the [A2A protocol](https://github.com/google-a2a/A2A/) for remote agent-to-agent communication.  See the [A2A example](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents) to learn more.

---

## 🚀 Getting Started

### Installation

Install the Agent Development Kit using `pip`:

```bash
pip install google-adk
```

### Development Version

For the latest features and bug fixes, install directly from the `main` branch:

```bash
pip install git+https://github.com/google/adk-python.git@main
```

## 📚 Documentation

Access comprehensive guides and resources:

*   [ADK Documentation](https://google.github.io/adk-docs)

## 🏁 Example: Building Agents with ADK

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

### Development UI Example:
<img src="https://raw.githubusercontent.com/google/adk-python/main/assets/adk-web-dev-ui-function-call.png"/>

###  Evaluate Agents

```bash
adk eval \
    samples_for_testing/hello_world \
    samples_for_testing/hello_world/hello_world_eval_set_001.evalset.json
```

---

## 🤝 Contributing

Contribute to the ADK project:

*   [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/)
*   [Code Contributing Guidelines](./CONTRIBUTING.md)

---

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.