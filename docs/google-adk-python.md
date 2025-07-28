# Agent Development Kit (ADK) for Python: Build, Evaluate, and Deploy AI Agents

**Unleash the power of AI with the Agent Development Kit (ADK), a flexible and code-first Python toolkit for building sophisticated AI agents. [Learn more at the original repository](https://github.com/google/adk-python).**

<div align="center">
    <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256" alt="Agent Development Kit Logo"/>
</div>

The Agent Development Kit (ADK) empowers developers to create, evaluate, and deploy AI agents with unparalleled flexibility and control. While optimized for Gemini and the Google ecosystem, ADK is model-agnostic and deployment-agnostic, allowing for seamless integration with other frameworks.

**Key Features:**

*   **Rich Tool Ecosystem:** Utilize pre-built tools, custom functions, OpenAPI specs, or integrate existing tools for diverse agent capabilities.
*   **Code-First Development:** Define agent logic, tools, and orchestration directly in Python for ultimate flexibility, testability, and versioning.
*   **Modular Multi-Agent Systems:** Design scalable applications by composing multiple specialized agents into flexible hierarchies.
*   **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale seamlessly with Vertex AI Agent Engine.
*   **Built-in Development UI:** Test, evaluate, debug, and showcase your agent(s) with a user-friendly interface.
*   **Evaluation Capabilities:** Easily evaluate your agents.

## 🤖 Agent2Agent (A2A) Protocol and ADK Integration

ADK integrates with the [A2A protocol](https://github.com/google-a2a/A2A/) for remote agent-to-agent communication.  See this [example](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents) for how they can work together.

## 🚀 Installation

### Stable Release (Recommended)

```bash
pip install google-adk
```

### Development Version

```bash
pip install git+https://github.com/google/adk-python.git@main
```

**Note:** The development version may contain experimental features or bugs.

## 📚 Documentation

Explore detailed guides for building, evaluating, and deploying agents:

*   [Documentation](https://google.github.io/adk-docs)

## 🏁 Feature Highlight

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

## 🤝 Contributing

We welcome contributions! See our:

*   [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/)
*   [Code Contributing Guidelines](./CONTRIBUTING.md)

## 📄 License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.