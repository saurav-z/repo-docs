<!-- Improved README - SEO Optimized -->

# Agent Development Kit (ADK) - Build Powerful AI Agents with Python

**Unleash the power of AI with the Agent Development Kit (ADK), a Python-first toolkit designed for creating, evaluating, and deploying sophisticated AI agents.** ([Original Repo](https://github.com/google/adk-python))

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/google-adk)](https://pypi.org/project/google-adk/)
[![Python Unit Tests](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml/badge.svg)](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/google/adk-python)

<div align="center">
  <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256" alt="Agent Development Kit Logo"/>
</div>

## Key Features

*   **Code-First Development:** Define agents and their logic directly in Python, offering full control, testability, and versioning.
*   **Rich Tool Ecosystem:** Integrate pre-built tools, custom functions, and OpenAPI specifications to empower agents with diverse capabilities.
*   **Modular Multi-Agent Systems:** Design scalable applications by composing specialized agents into flexible hierarchies.
*   **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale seamlessly with Vertex AI Agent Engine.
*   **Agent Config**: Build agents without code using the [Agent Config](https://google.github.io/adk-docs/agents/config/) feature.
*   **Built-in Development UI:** Test, evaluate, debug, and showcase your agents with a user-friendly development UI.
*   **Evaluation Tools**: Evaluate the performance of your agents with the built-in evaluation tools.

## Core Benefits

*   **Flexibility and Control:** Tailor agents to your specific needs with a code-first approach.
*   **Scalability:** Build multi-agent systems that can handle complex tasks.
*   **Ease of Deployment:** Deploy your agents quickly and easily on a variety of platforms.
*   **Integration:** Works well with Gemini models and the Google ecosystem.

## Getting Started

### Installation

#### Stable Release (Recommended)

Install the latest stable version using pip:

```bash
pip install google-adk
```

Weekly release cadence.

#### Development Version

Install directly from the `main` branch for the latest features and bug fixes:

```bash
pip install git+https://github.com/google/adk-python.git@main
```

*Note:* The development version may contain experimental features or bugs.

### Documentation

Explore the full documentation for in-depth guides:

*   [Documentation](https://google.github.io/adk-docs)

## Examples

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

## Integration with A2A Protocol

ADK integrates with the [A2A protocol](https://github.com/google-a2a/A2A/) for remote agent-to-agent communication.
See this [example](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents) for how they can work together.

## Contributing

We welcome contributions! See our:

*   [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/).
*   [Code Contributing Guidelines](./CONTRIBUTING.md).

## Vibe Coding

Use [llms.txt](./llms.txt) and [llms-full.txt](./llms-full.txt) as context for LLM during vibe coding.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

*Happy Agent Building!*