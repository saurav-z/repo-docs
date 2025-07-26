# Agent Development Kit (ADK): Build, Evaluate, and Deploy AI Agents with Ease

**ADK empowers developers to build sophisticated AI agents with code-first development, offering unparalleled flexibility and control. ([See the original repository](https://github.com/google/adk-python))**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python Unit Tests](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml/badge.svg)](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/google/adk-python)

<h2 align="center">
  <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256" alt="Agent Development Kit Logo"/>
</h2>
<h3 align="center">
  An open-source Python toolkit for building, evaluating, and deploying AI agents.
</h3>
<h3 align="center">
  Important Links:
  <a href="https://google.github.io/adk-docs/">Docs</a>,
  <a href="https://github.com/google/adk-samples">Samples</a>,
  <a href="https://github.com/google/adk-java">Java ADK</a> &
  <a href="https://github.com/google/adk-web">ADK Web</a>.
</h3>

Agent Development Kit (ADK) is a flexible and modular framework designed for developing and deploying AI agents. While optimized for Gemini and the Google ecosystem, ADK is model-agnostic, deployment-agnostic, and built for compatibility with other frameworks. ADK makes agent development feel more like software development, simplifying the creation, deployment, and orchestration of agentic architectures.

---

## Key Features

*   **Rich Tool Ecosystem:** Integrate pre-built tools, custom functions, and OpenAPI specs to equip agents with diverse capabilities, with seamless integration with the Google ecosystem.
*   **Code-First Development:** Define agent logic, tools, and orchestration directly in Python for maximum flexibility, testability, and version control.
*   **Modular Multi-Agent Systems:** Design scalable applications by composing multiple specialized agents into flexible hierarchies.
*   **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale seamlessly with Vertex AI Agent Engine.
*   **Agent2Agent (A2A) Integration:** Seamlessly integrates with the A2A protocol for remote agent communication.

## 🚀 Installation

### Stable Release (Recommended)

Install the latest stable version using `pip`:

```bash
pip install google-adk
```

The release cadence is weekly.

### Development Version

For access to the latest bug fixes and features, install directly from the `main` branch:

```bash
pip install git+https://github.com/google/adk-python.git@main
```

*Note: The development version may include experimental changes or bugs.*

## 📚 Documentation

Explore the full documentation for detailed guides on building, evaluating, and deploying agents:

*   **[Documentation](https://google.github.io/adk-docs)**

## 🏁 Feature Highlights

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

Define a multi-agent system with coordinator agent, greeter agent, and task execution agent. Then ADK engine and the model will guide the agents works together to accomplish the task.

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

A built-in development UI to help you test, evaluate, debug, and showcase your agent(s).

<img src="https://raw.githubusercontent.com/google/adk-python/main/assets/adk-web-dev-ui-function-call.png" alt="ADK Development UI"/>

### Evaluate Agents

```bash
adk eval \
    samples_for_testing/hello_world \
    samples_for_testing/hello_world/hello_world_eval_set_001.evalset.json
```

## 🤝 Contributing

We welcome contributions from the community! Please see our:

*   [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/).
*   [Code Contributing Guidelines](./CONTRIBUTING.md)

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

*Happy Agent Building!*