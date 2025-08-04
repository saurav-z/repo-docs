# Agent Development Kit (ADK): Build, Evaluate, and Deploy AI Agents with Ease

**[Agent Development Kit (ADK)](https://github.com/google/adk-python) empowers developers to build, test, and deploy sophisticated AI agents using a code-first approach.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python Unit Tests](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml/badge.svg)](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/google/adk-python)
<p align="center">
  <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256" alt="Agent Development Kit Logo">
</p>

ADK is a flexible and modular framework designed to simplify the development and deployment of AI agents. While optimized for the Google ecosystem, it's model-agnostic and deployment-agnostic, integrating seamlessly with other frameworks, allowing developers to create everything from simple task-oriented agents to complex, multi-agent systems.

**Key Features:**

*   **Rich Tool Ecosystem:** Leverage pre-built tools, custom functions, and OpenAPI specifications for diverse agent capabilities, with tight Google ecosystem integration.
*   **Code-First Development:** Define agent logic, tools, and orchestration directly in Python for maximum flexibility, testability, and version control.
*   **Modular Multi-Agent Systems:** Design scalable applications by composing specialized agents into flexible hierarchies.
*   **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale seamlessly with Vertex AI Agent Engine.
*   **Built-in Development UI:** Test, evaluate, debug, and showcase your agents using a convenient development UI.
*   **Agent2Agent (A2A) Protocol Integration:** Utilize the A2A protocol for remote agent-to-agent communication.

---

## 🚀 Getting Started

### Installation

#### Stable Release (Recommended)

Install the latest stable version using `pip`:

```bash
pip install google-adk
```

Weekly releases ensure you get the most recent, officially supported features.

#### Development Version

To access the latest features and bug fixes before they are officially released, install directly from the `main` branch:

```bash
pip install git+https://github.com/google/adk-python.git@main
```

*Note: The development version may include experimental features or bugs and is primarily for testing and early access.*

---

## 📚 Documentation

Explore the comprehensive documentation for detailed guides and examples:

*   [Documentation](https://google.github.io/adk-docs)

---

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

<img src="https://raw.githubusercontent.com/google/adk-python/main/assets/adk-web-dev-ui-function-call.png" alt="ADK Development UI">

### Evaluate Agents

```bash
adk eval \
    samples_for_testing/hello_world \
    samples_for_testing/hello_world/hello_world_eval_set_001.evalset.json
```

---

## 🤝 Contributing

We welcome community contributions! Review our:

*   [General contribution guidelines and flow](https://google.github.io/adk-docs/contributing-guide/)
*   [Code Contributing Guidelines](./CONTRIBUTING.md)

---

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
---

*Happy Agent Building!*