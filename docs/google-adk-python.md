# Agent Development Kit (ADK): Build Powerful AI Agents with Python

**Unlock the potential of AI agents with the Agent Development Kit (ADK), a Python-first toolkit designed for flexible, code-driven agent creation, evaluation, and deployment.** ([Original Repo](https://github.com/google/adk-python))

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/google-adk)](https://pypi.org/project/google-adk/)
[![Python Unit Tests](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml/badge.svg)](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/google/adk-python)

<div align="center">
  <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256" alt="Agent Development Kit Logo">
</div>

## Key Features

*   **Code-First Development:** Define agent logic, tools, and orchestration directly in Python for unparalleled flexibility, testability, and version control.
*   **Rich Tool Ecosystem:** Seamlessly integrate with pre-built tools, custom functions, and OpenAPI specifications, with a focus on the Google ecosystem.
*   **Modular Multi-Agent Systems:** Design scalable applications by composing multiple specialized agents into flexible hierarchies for complex workflows.
*   **Deploy Anywhere:** Effortlessly containerize and deploy agents on Cloud Run or scale with Vertex AI Agent Engine.
*   **Agent Config:** Build agents without code. Check out the [Agent Config](https://google.github.io/adk-docs/agents/config/) feature.

## What's New

*   **Agent Config**: Build agents without code. Check out the
  [Agent Config](https://google.github.io/adk-docs/agents/config/) feature.

## Agent2Agent (A2A) Protocol and ADK Integration

ADK integrates with the [A2A protocol](https://github.com/google-a2a/A2A/) for remote agent-to-agent communication.  See an [example](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents) of how they can work together.

## Installation

### Stable Release (Recommended)

```bash
pip install google-adk
```

This version is recommended for most users as it represents the most recent official release, with weekly release cadence.

### Development Version

```bash
pip install git+https://github.com/google/adk-python.git@main
```

Use the development version for testing or accessing critical fixes before official releases.  It may contain experimental changes or bugs.

## Documentation

Explore comprehensive guides for building, evaluating, and deploying agents:

*   [Documentation](https://google.github.io/adk-docs)

## Example Code

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

A built-in development UI to help you test, evaluate, debug, and showcase your agent(s).

<img src="https://raw.githubusercontent.com/google/adk-python/main/assets/adk-web-dev-ui-function-call.png" alt="ADK Development UI">

### Evaluate Agents

```bash
adk eval \
    samples_for_testing/hello_world \
    samples_for_testing/hello_world/hello_world_eval_set_001.evalset.json
```

## Contributing

We welcome community contributions! See our:

*   [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/).
*   [Code Contributing Guidelines](./CONTRIBUTING.md)

## Vibe Coding

Use [llms.txt](./llms.txt) and [llms-full.txt](./llms-full.txt) as context for LLM-based agent development.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

*Happy Agent Building!*
```
Key improvements and SEO considerations:

*   **Strong Hook:** The initial sentence directly highlights the value proposition, attracting developers.
*   **Clear Headings:** Uses `h2` and `h3` for better readability and SEO structure.
*   **Bulleted Key Features:**  Emphasizes the core benefits and advantages of using ADK.
*   **Keywords:** Naturally incorporates relevant keywords like "AI agents," "Python," "toolkit," "agent development," "Gemini," "Google," "code-first," "modular," and "deployment."
*   **Concise Language:** Removes redundant phrases and streamlines the text.
*   **Call to Action:** Encourages engagement with "Happy Agent Building!"
*   **Improved Formatting:**  Uses bolding and formatting to highlight key information.
*   **Alt Text for Image:** Added `alt` text to the image for accessibility and SEO.
*   **Internal Linking:**  Links to documentation, samples, and contributing guidelines.
*   **Removed HTML:** Converted the HTML-based section to Markdown.
*   **Development UI Description:** Added a description to the Development UI example for added context.
*   **Clearer installation instructions**:  Combined similar information, while retaining the details.