# Agent Development Kit (ADK): Build Powerful AI Agents with Python

**Unleash the power of AI agents with the Agent Development Kit (ADK), a flexible, code-first Python toolkit for building, evaluating, and deploying sophisticated agents.** ([Original Repository](https://github.com/google/adk-python))

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/google-adk)](https://pypi.org/project/google-adk/)
[![Python Unit Tests](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml/badge.svg)](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/google/adk-python)

<div align="center">
  <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256" alt="Agent Development Kit Logo">
</div>

<div align="center">
  An open-source, code-first Python toolkit for building, evaluating, and deploying sophisticated AI agents with flexibility and control.
</div>

<div align="center">
  <br>
  <b>Important Links:</b>
  <a href="https://google.github.io/adk-docs/">Docs</a>,
  <a href="https://github.com/google/adk-samples">Samples</a>,
  <a href="https://github.com/google/adk-java">Java ADK</a> &
  <a href="https://github.com/google/adk-web">ADK Web</a>.
</div>

## Key Features

*   **Code-First Development:** Define agent logic, tools, and orchestration directly in Python, enabling version control, testing, and ultimate flexibility.
*   **Rich Tool Ecosystem:** Integrate pre-built tools, custom functions, and OpenAPI specs to empower your agents. Seamless Google ecosystem integration.
*   **Modular Multi-Agent Systems:** Design scalable applications by composing multiple specialized agents into flexible hierarchies.
*   **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale seamlessly with Vertex AI Agent Engine.
*   **Agent Config**: Build agents without code. Check out the [Agent Config](https://google.github.io/adk-docs/agents/config/) feature.

## What's New

- **Agent Config**: Build agents without code. Check out the
  [Agent Config](https://google.github.io/adk-docs/agents/config/) feature.

## Integration with Agent2Agent (A2A) Protocol

ADK seamlessly integrates with the [A2A protocol](https://github.com/google-a2a/A2A/) for secure and efficient remote agent-to-agent communication.  See the [example](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents) to see how they work together.

## Installation

### Stable Release (Recommended)

```bash
pip install google-adk
```

Get the latest official release for stability and reliability.  Releases occur weekly.

### Development Version

```bash
pip install git+https://github.com/google/adk-python.git@main
```

Access the latest features and bug fixes by installing directly from the `main` branch.  Note that this may include experimental features or bugs.

## Documentation

Comprehensive documentation is available to guide you through building, evaluating, and deploying agents:

*   [Documentation](https://google.github.io/adk-docs)

## Feature Highlight: Code Examples

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

<img src="https://raw.githubusercontent.com/google/adk-python/main/assets/adk-web-dev-ui-function-call.png" alt="ADK Development UI">

### Evaluate Agents

```bash
adk eval \
    samples_for_testing/hello_world \
    samples_for_testing/hello_world/hello_world_eval_set_001.evalset.json
```

## Contributing

We welcome community contributions!  Please refer to our:

-   [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/).
-   [Code Contributing Guidelines](./CONTRIBUTING.md)

## Vibe Coding

Use the [llms.txt](./llms.txt) and the [llms-full.txt](./llms-full.txt) files as context for your LLMs during Vibe coding.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

*Happy Agent Building!*