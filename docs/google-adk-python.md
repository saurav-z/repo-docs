# Agent Development Kit (ADK): Build Powerful AI Agents with Python

**Unlock the potential of AI agents with the Agent Development Kit (ADK), a flexible Python toolkit for building, evaluating, and deploying sophisticated AI agents.** [Learn more on the original repository](https://github.com/google/adk-python).

<div align="center">
  <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256"/>
</div>

ADK is an open-source, code-first framework, optimized for Gemini and the Google ecosystem, while maintaining model and deployment agnosticism, offering seamless integration with various frameworks. Designed for developers, it simplifies creating, deploying, and orchestrating AI agents, from simple tasks to complex workflows.

**Key Features of the Agent Development Kit:**

*   **Rich Tool Ecosystem:** Integrate pre-built tools, custom functions, and OpenAPI specs for diverse agent capabilities, with tight Google ecosystem integration.
*   **Code-First Development:** Define agent logic, tools, and orchestration directly in Python, ensuring flexibility, testability, and version control.
*   **Modular Multi-Agent Systems:** Design scalable applications by composing multiple specialized agents into flexible hierarchies.
*   **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale seamlessly with Vertex AI Agent Engine.

## 🤖 Agent-to-Agent (A2A) Protocol Integration

ADK integrates with the [A2A protocol](https://github.com/google-a2a/A2A/) for remote agent-to-agent communication. See this [example](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents) for how they can work together.

## 🚀 Installation

### Stable Release (Recommended)

Install the latest stable version using pip:

```bash
pip install google-adk
```

*   The release cadence is weekly.

### Development Version

Install directly from the main branch for the latest features and bug fixes:

```bash
pip install git+https://github.com/google/adk-python.git@main
```

*   Note: The development version may contain experimental changes or bugs.

## 📚 Documentation

Explore the comprehensive documentation for in-depth guides:

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

<img src="https://raw.githubusercontent.com/google/adk-python/main/assets/adk-web-dev-ui-function-call.png"/>

### Evaluate Agents

```bash
adk eval \
    samples_for_testing/hello_world \
    samples_for_testing/hello_world/hello_world_eval_set_001.evalset.json
```

## 🤝 Contributing

We welcome community contributions!  Find details on the [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/) and [Code Contributing Guidelines](./CONTRIBUTING.md).

## Vibe Coding

Use the [llms.txt](./llms.txt) and the [llms-full.txt](./llms-full.txt) files as context for LLMs in vibe coding.

## 📄 License

Licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file.