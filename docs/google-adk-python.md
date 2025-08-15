# Agent Development Kit (ADK): Build, Evaluate, and Deploy AI Agents with Ease

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/google-adk)](https://pypi.org/project/google-adk/)
[![Python Unit Tests](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml/badge.svg)](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/google/adk-python)

<div align="center">
  <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256" alt="Agent Development Kit Logo"/>
</div>

**Unleash the power of AI with the Agent Development Kit (ADK), a Python-first toolkit designed for crafting, evaluating, and deploying sophisticated AI agents.**

ADK empowers developers to build, test, and deploy cutting-edge AI agents with flexibility and control.  Designed for the Google ecosystem, ADK is also model-agnostic and deployment-agnostic, ensuring compatibility with diverse frameworks. Dive in and make agent development feel like software development, simplifying the creation, deployment, and orchestration of agents from simple tasks to complex workflows. [Explore the original repository on GitHub](https://github.com/google/adk-python).

## Key Features

*   **Rich Tool Ecosystem:** Integrate pre-built tools, custom functions, OpenAPI specs, and existing tools to give agents diverse capabilities, ensuring tight integration with the Google ecosystem.
*   **Code-First Development:** Define agent logic, tools, and orchestration directly in Python for ultimate flexibility, testability, and versioning.
*   **Modular Multi-Agent Systems:** Design scalable applications by composing multiple specialized agents into flexible hierarchies.
*   **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale seamlessly with Vertex AI Agent Engine.
*   **Built-in Development UI:** Test, evaluate, debug, and showcase your agent(s) with a built-in development UI.
*   **Evaluation Tools:**  Evaluate agent performance using included evaluation tools.

## 🤖 Agent2Agent (A2A) Protocol and ADK Integration

ADK seamlessly integrates with the [A2A protocol](https://github.com/google-a2a/A2A/) for remote agent-to-agent communication. Explore this [example](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents) to see A2A and ADK in action.

## 🚀 Installation

### Stable Release (Recommended)

Install the latest stable version of ADK using `pip`:

```bash
pip install google-adk
```

Official releases are published weekly.

### Development Version

To access the latest bug fixes and features from the `main` branch:

```bash
pip install git+https://github.com/google/adk-python.git@main
```

**Note:** The development version may include experimental features or bugs.  Use this version for testing or critical fixes before the official release.

## 📚 Documentation

Comprehensive documentation is available to guide you through building, evaluating, and deploying agents:

*   [Documentation](https://google.github.io/adk-docs)

## 🏁 Quickstart

### Define a single agent:

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

### Define a multi-agent system:

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

### Evaluate Agents

```bash
adk eval \
    samples_for_testing/hello_world \
    samples_for_testing/hello_world/hello_world_eval_set_001.evalset.json
```

### Development UI
<img src="https://raw.githubusercontent.com/google/adk-python/main/assets/adk-web-dev-ui-function-call.png" alt="ADK Web Dev UI Function Call"/>

## 🤝 Contributing

We welcome contributions! Review the following resources:

*   [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/)
*   [Code Contributing Guidelines](./CONTRIBUTING.md)

## Vibe Coding

For agent development via vibe coding, reference the [llms.txt](./llms.txt) and [llms-full.txt](./llms-full.txt) files as context for your LLM. The former is a summary, while the latter contains full information.

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

*Happy Agent Building!*