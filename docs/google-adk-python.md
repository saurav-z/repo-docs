# Agent Development Kit (ADK): Build, Evaluate, and Deploy AI Agents with Ease

**Unleash the power of AI agents with the Agent Development Kit (ADK), a code-first Python toolkit for building sophisticated agentic applications.** ([Original Repo](https://github.com/google/adk-python))

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/google-adk)](https://pypi.org/project/google-adk/)
[![Python Unit Tests](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml/badge.svg)](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/google/adk-python)

<div align="center">
  <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256" alt="Agent Development Kit Logo">
</div>

ADK is a flexible, modular, and model-agnostic framework designed for developers to create, deploy, and orchestrate AI agents, going from simple tasks to complex workflows. It's optimized for the Google ecosystem but compatible with other frameworks, making agent development feel like software development.

## Key Features of the Agent Development Kit

*   **Code-First Development:** Define agent logic, tools, and orchestration directly in Python, promoting flexibility, testability, and version control.
*   **Rich Tool Ecosystem:** Integrate pre-built tools, custom functions, and OpenAPI specifications to equip agents with diverse capabilities, with tight integration with the Google ecosystem.
*   **Modular Multi-Agent Systems:** Design scalable applications by composing multiple specialized agents into flexible hierarchies.
*   **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale seamlessly with Vertex AI Agent Engine.
*   **Agent Config**: Build agents without code using the [Agent Config](https://google.github.io/adk-docs/agents/config/) feature.
*   **Development UI**: A built-in development UI to help you test, evaluate, debug, and showcase your agent(s).
*   **Agent Evaluation**: Evaluate agent performance with built-in evaluation tools.

## 🤖 Agent2Agent (A2A) Protocol Integration

ADK seamlessly integrates with the [A2A protocol](https://github.com/google-a2a/A2A/) for remote agent-to-agent communication. See an [example](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents) of how to use it.

## 🚀 Getting Started

### Installation

Install the ADK using pip:

*   **Stable Release (Recommended):** `pip install google-adk` (Bi-weekly release cadence)
*   **Development Version:** `pip install git+https://github.com/google/adk-python.git@main` (For the latest features and fixes; may contain experimental changes or bugs.)

## 📚 Documentation

*   Explore the full documentation:  [Documentation](https://google.github.io/adk-docs)

## 🏁 Example: Defining Agents

**Define a single agent:**

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

**Define a multi-agent system:**

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

###  Evaluate Agents
```bash
adk eval \
    samples_for_testing/hello_world \
    samples_for_testing/hello_world/hello_world_eval_set_001.evalset.json
```

## 🤝 Contributing

We welcome your contributions! See our:

*   [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/).
*   [Code Contributing Guidelines](./CONTRIBUTING.md)

## Vibe Coding

If you are to develop agent via vibe coding the [llms.txt](./llms.txt) and the [llms-full.txt](./llms-full.txt) can be used as context to LLM. While the former one is a summarized one and the later one has the full information in case your LLM has big enough context window.

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

*Happy Agent Building!*