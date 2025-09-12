# Agent Development Kit (ADK): Build Powerful AI Agents with Python

**Unleash the power of AI agents with Google's open-source Agent Development Kit (ADK), a code-first toolkit for flexible and scalable agent development.** ([Original Repository](https://github.com/google/adk-python))

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/google-adk)](https://pypi.org/project/google-adk/)
[![Python Unit Tests](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml/badge.svg)](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/google/adk-python)

<div align="center">
  <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256" alt="Agent Development Kit Logo"/>
</div>

ADK is a flexible and modular framework designed for building, evaluating, and deploying sophisticated AI agents. While optimized for the Google ecosystem, ADK is model-agnostic and deployment-agnostic, offering developers unparalleled control and adaptability.

## Key Features:

*   **Code-First Development:** Define agents, tools, and orchestration logic directly in Python, enabling version control, testing, and ultimate flexibility.
*   **Rich Tool Ecosystem:** Integrate pre-built tools, custom functions, and OpenAPI specifications, or leverage existing tools for diverse agent capabilities.
*   **Modular Multi-Agent Systems:** Create scalable applications by composing multiple specialized agents into flexible hierarchies.
*   **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale seamlessly with Vertex AI Agent Engine.
*   **Agent Configuration:** Build agents without any code by utilizing the new [Agent Config](https://google.github.io/adk-docs/agents/config/) feature.

## 🚀 Getting Started

### Installation

#### Stable Release (Recommended)

Install the latest stable version using `pip`:

```bash
pip install google-adk
```

#### Development Version

Install the latest development version directly from GitHub (for the newest features and bug fixes):

```bash
pip install git+https://github.com/google/adk-python.git@main
```

## 📚 Documentation

Comprehensive documentation is available to guide you through every step of agent development:

*   **[Documentation](https://google.github.io/adk-docs)**

## 🤖 Key Integrations

*   **A2A Protocol:** ADK seamlessly integrates with the [A2A protocol](https://github.com/google-a2a/A2A/) for remote agent-to-agent communication.  See this [example](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents) to see how they work together.

## 💡 Feature Highlights

### Define a single agent:

```python
from google.adk.agents import Agent
from google.adk.tools import google_search

root_agent = Agent(
    name="search_assistant",
    model="gemini-2.5-flash", # Or your preferred Gemini model
    instruction="You are a helpful assistant. Answer user questions using Google Search when needed.",
    description="An assistant that can search the web.",
    tools=[google_search]
)
```

### Define a multi-agent system:

```python
from google.adk.agents import LlmAgent, BaseAgent

# Define individual agents
greeter = LlmAgent(name="greeter", model="gemini-2.5-flash", ...)
task_executor = LlmAgent(name="task_executor", model="gemini-2.5-flash", ...)

# Create parent agent and assign children via sub_agents
coordinator = LlmAgent(
    name="Coordinator",
    model="gemini-2.5-flash",
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

We encourage community contributions! Review the [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/).  If you want to contribute code, read the [Code Contributing Guidelines](./CONTRIBUTING.md).

## Vibe Coding

If you are to develop agent via vibe coding the [llms.txt](./llms.txt) and the [llms-full.txt](./llms-full.txt) can be used as context to LLM. While the former one is a summarized one and the later one has the full information in case your LLM has big enough context window.

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.