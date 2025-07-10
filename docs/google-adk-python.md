# Agent Development Kit (ADK): Build and Deploy Powerful AI Agents

**Empower your AI agent development with Google's Agent Development Kit (ADK), a code-first toolkit for flexible and scalable agentic systems.** ([View on GitHub](https://github.com/google/adk-python))

<div align="center">
  <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256"/>
</div>

ADK is an open-source Python framework designed for building, evaluating, and deploying sophisticated AI agents. It's optimized for the Google ecosystem, including Gemini, while remaining model-agnostic and deployment-agnostic. ADK allows developers to create, deploy, and orchestrate agentic architectures, from simple tasks to complex workflows with ease.

## Key Features of ADK:

*   **Rich Tool Ecosystem:** Utilize pre-built tools, custom functions, and integrate with existing APIs for diverse agent capabilities.
*   **Code-First Development:** Define agent logic, tools, and orchestration directly in Python for enhanced flexibility, testability, and version control.
*   **Modular Multi-Agent Systems:** Design scalable applications by composing multiple specialized agents into flexible hierarchies.
*   **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale seamlessly with Vertex AI Agent Engine.

## Agent2Agent (A2A) Protocol Integration

ADK integrates with the [A2A protocol](https://github.com/google-a2a/A2A/) for remote agent-to-agent communication. See this [example](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents) for a demonstration of how to use the protocol with ADK.

## Installation

### Stable Release (Recommended)

```bash
pip install google-adk
```

This is the recommended version for most users, representing the latest official release. The release cadence is weekly.

### Development Version

```bash
pip install git+https://github.com/google/adk-python.git@main
```

Install the latest features and bug fixes from the main branch.  Note that this version may contain experimental features and is suitable for testing or accessing the newest updates.

## Documentation

Comprehensive documentation is available for detailed guides:

*   [Documentation](https://google.github.io/adk-docs)

## Feature Highlights

### Define a Single Agent

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

### Define a Multi-Agent System

ADK helps to create multi-agent systems:

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

ADK includes a built-in development UI to help you test, evaluate, debug, and showcase your agent(s).

<img src="https://raw.githubusercontent.com/google/adk-python/main/assets/adk-web-dev-ui-function-call.png"/>

### Evaluate Agents

```bash
adk eval \
    samples_for_testing/hello_world \
    samples_for_testing/hello_world/hello_world_eval_set_001.evalset.json
```

## Contributing

We welcome community contributions! Please review our:

*   [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/).
*   [Code Contributing Guidelines](./CONTRIBUTING.md)

## License

This project is licensed under the Apache 2.0 License.  See the [LICENSE](LICENSE) file for details.