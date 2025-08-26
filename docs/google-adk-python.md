# Agent Development Kit (ADK): Build, Evaluate, and Deploy AI Agents

**Unlock the power of code-first AI agent development with Google's Agent Development Kit (ADK).** [See the original repository](https://github.com/google/adk-python)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/google-adk)](https://pypi.org/project/google-adk/)
[![Python Unit Tests](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml/badge.svg)](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/google/adk-python)

<div align="center">
  <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256" alt="Agent Development Kit Logo"/>
</div>

Agent Development Kit (ADK) is a powerful, open-source Python toolkit designed for building, evaluating, and deploying sophisticated AI agents.  It provides the flexibility and control you need to create agents ranging from simple task automation to complex multi-agent systems. ADK is optimized for the Google ecosystem, but it is model-agnostic and deployment-agnostic, ensuring compatibility with a variety of frameworks.

## Key Features

*   **Code-First Development:** Define agents, tools, and orchestration logic directly in Python, enabling version control, testing, and ultimate flexibility.
*   **Rich Tool Ecosystem:** Easily integrate with pre-built tools, custom functions, and OpenAPI specifications, empowering agents with diverse capabilities.  Tight Google ecosystem integration.
*   **Modular Multi-Agent Systems:** Design scalable applications by composing multiple specialized agents into flexible hierarchies for complex workflows.
*   **Deploy Anywhere:** Containerize and deploy agents on Cloud Run or scale seamlessly with Vertex AI Agent Engine.
*   **Agent Configuration:** Build agents without code using the [Agent Config](https://google.github.io/adk-docs/agents/config/) feature.
*   **Built-in Development UI:** Test, evaluate, debug, and showcase your agent(s) with the built-in development UI.
*   **Agent Evaluation:** Evaluate agents using a command line tool.

## 🤖 Agent2Agent (A2A) Protocol Integration

ADK seamlessly integrates with the [A2A protocol](https://github.com/google-a2a/A2A/) for remote agent-to-agent communication.  See this [example](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents) for how they can work together.

## 🚀 Installation

### Stable Release (Recommended)

```bash
pip install google-adk
```

This is the recommended installation method for most users, providing the latest officially released version.  Releases occur weekly.

### Development Version

To access the latest bug fixes and features before they are officially released, install directly from the main branch:

```bash
pip install git+https://github.com/google/adk-python.git@main
```

**Important:** The development version may contain experimental features or bugs. Use it for testing or to access critical fixes before the stable release.

## 📚 Documentation

Comprehensive documentation is available to guide you through building, evaluating, and deploying agents:

*   [Documentation](https://google.github.io/adk-docs)

## 🏁 Feature Highlights

### Defining a Single Agent:

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

### Defining a Multi-Agent System:

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
<img src="https://raw.githubusercontent.com/google/adk-python/main/assets/adk-web-dev-ui-function-call.png"/>

### Evaluate Agents

```bash
adk eval \
    samples_for_testing/hello_world \
    samples_for_testing/hello_world/hello_world_eval_set_001.evalset.json
```

## 🤝 Contributing

We welcome contributions from the community!

*   [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/)
*   [Code Contributing Guidelines](./CONTRIBUTING.md)

## Vibe Coding

Use [llms.txt](./llms.txt) and [llms-full.txt](./llms-full.txt) as context for LLMs in vibe coding.

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

*Happy Agent Building!*