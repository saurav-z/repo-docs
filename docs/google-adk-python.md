# Agent Development Kit (ADK): Build Powerful AI Agents with Python

**Empower your AI development: ADK is an open-source, code-first Python toolkit for building, evaluating, and deploying sophisticated AI agents.** ([Original Repository](https://github.com/google/adk-python))

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python Unit Tests](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml/badge.svg)](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/google/adk-python)

<h2 align="center">
  <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256"/>
</h2>

ADK provides a flexible, modular, and code-first approach to agent development, enabling developers to create, deploy, and orchestrate AI agents for a wide range of applications. While optimized for the Google ecosystem, ADK is designed to be model-agnostic and deployment-agnostic, ensuring compatibility with various frameworks.

**Key Features:**

*   **Code-First Development:** Define agent logic, tools, and orchestration directly in Python, promoting flexibility, testability, and versioning.
*   **Rich Tool Ecosystem:** Integrate pre-built tools, custom functions, and OpenAPI specifications to equip agents with diverse capabilities.
*   **Modular Multi-Agent Systems:** Design scalable applications by composing multiple specialized agents into flexible hierarchies.
*   **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale seamlessly with Vertex AI Agent Engine.
*   **Built-in Development UI:** Test, evaluate, debug, and showcase your agents using the built-in development UI.
*   **Agent2Agent (A2A) Protocol Integration:** Seamlessly integrate with the A2A protocol for remote agent communication.

---

## 🚀 Getting Started

### Installation

Choose your preferred installation method:

#### Stable Release (Recommended)

Install the latest stable version of ADK using `pip`:

```bash
pip install google-adk
```

This is the recommended option for most users, providing the most recent official release.

#### Development Version

For access to the latest features and bug fixes (potentially unstable), install directly from the main branch:

```bash
pip install git+https://github.com/google/adk-python.git@main
```

**Note:** Use the development version for testing and accessing critical fixes before official releases.

---

## 📚 Documentation

Explore the comprehensive documentation for detailed guides and examples:

*   **[Documentation](https://google.github.io/adk-docs)**

---

## 💻 Example Usage

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

###  Evaluate Agents

```bash
adk eval \
    samples_for_testing/hello_world \
    samples_for_testing/hello_world/hello_world_eval_set_001.evalset.json
```

---

## 🤝 Contributing

We welcome contributions! Refer to the following resources:

*   [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/)
*   [Code Contributing Guidelines](./CONTRIBUTING.md)

---

## 📄 License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

---

*Happy Agent Building!*