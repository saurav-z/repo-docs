# Agent Development Kit (ADK): Build, Evaluate, and Deploy AI Agents with Ease

**ADK empowers developers to create sophisticated AI agents with a flexible, code-first approach.** ([Original Repo](https://github.com/google/adk-python))

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python Unit Tests](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml/badge.svg)](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/google/adk-python)

<div align="center">
  <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256" alt="Agent Development Kit Logo"/>
</div>

ADK is an open-source Python toolkit designed to streamline the development, evaluation, and deployment of advanced AI agents. It offers unparalleled flexibility and control, making agent creation feel like traditional software development. While optimized for the Google ecosystem, ADK is model-agnostic and deployment-agnostic, enabling seamless integration with diverse frameworks. Build everything from simple assistants to complex multi-agent systems with ADK.

**Key Features:**

*   ✅ **Code-First Development:** Define agent logic, tools, and orchestration directly in Python, enabling robust testing and version control.
*   🛠️ **Rich Tool Ecosystem:** Leverage pre-built tools, custom functions, and OpenAPI integrations for enhanced agent capabilities, with tight Google ecosystem integration.
*   👥 **Modular Multi-Agent Systems:** Design scalable applications by composing specialized agents into flexible hierarchies.
*   ☁️ **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale seamlessly with Vertex AI Agent Engine.
*   💻 **Development UI:** Built-in UI to help you test, evaluate, debug, and showcase your agents.
*   ✅ **Agent Evaluation:** Evaluate agents easily using the built in evaluation features
*   🤖 **A2A Protocol Integration:** Integrates seamlessly with the [A2A protocol](https://github.com/google-a2a/A2A/) for remote agent-to-agent communication.

---

## 🚀 Installation

### Stable Release (Recommended)

```bash
pip install google-adk
```

The stable release is updated weekly.

### Development Version

```bash
pip install git+https://github.com/google/adk-python.git@main
```

Note: The development version includes the newest features and fixes but may contain experimental changes or bugs.

## 📚 Documentation

*   **[Comprehensive Documentation](https://google.github.io/adk-docs):** Access detailed guides on building, evaluating, and deploying agents.

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

### Development UI

<img src="https://raw.githubusercontent.com/google/adk-python/main/assets/adk-web-dev-ui-function-call.png" alt="Development UI"/>

### Evaluate Agents

```bash
adk eval \
    samples_for_testing/hello_world \
    samples_for_testing/hello_world/hello_world_eval_set_001.evalset.json
```

## 🤝 Contributing

We welcome contributions! Review the following resources:

*   [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/).
*   [Code Contributing Guidelines](./CONTRIBUTING.md)

## Vibe Coding

Files [llms.txt](./llms.txt) and [llms-full.txt](./llms-full.txt) can be used as context to LLM.

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.