# Agent Development Kit (ADK): Build Powerful AI Agents with Python

**Unleash the power of AI agents with the Agent Development Kit (ADK), a code-first, open-source toolkit for building, evaluating, and deploying sophisticated AI agents.** ([Original Repository](https://github.com/google/adk-python))

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/google-adk)](https://pypi.org/project/google-adk/)
[![Python Unit Tests](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml/badge.svg)](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/google/adk-python)

<div align="center">
  <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256" alt="Agent Development Kit Logo">
</div>

ADK provides a flexible, modular, and model-agnostic framework, optimized for the Google ecosystem but compatible with others, to simplify the development lifecycle of AI agents. Build agents that range from simple tasks to complex workflows.

## Key Features

*   **Code-First Development:** Define agent logic, tools, and orchestration directly in Python, fostering flexibility, testability, and version control.
*   **Rich Tool Ecosystem:** Leverage pre-built tools, custom functions, and OpenAPI specs, enabling agents to access diverse capabilities.
*   **Modular Multi-Agent Systems:** Compose multiple specialized agents into flexible hierarchies for scalable application design.
*   **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale seamlessly with Vertex AI Agent Engine.
*   **Agent Config**: Build agents without code. Check out the [Agent Config](https://google.github.io/adk-docs/agents/config/) feature.

## Getting Started

### Installation

Install the ADK using `pip`:

```bash
pip install google-adk
```

### Development Version
For access to the latest changes, install directly from the main branch:

```bash
pip install git+https://github.com/google/adk-python.git@main
```

### Documentation

Explore the full documentation for detailed guides on building, evaluating, and deploying agents:

*   [Documentation](https://google.github.io/adk-docs)

## Examples

### Defining a Single Agent

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

### Defining a Multi-Agent System

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

ADK offers a built-in development UI to facilitate testing, evaluation, debugging, and showcasing your agents.

<img src="https://raw.githubusercontent.com/google/adk-python/main/assets/adk-web-dev-ui-function-call.png" alt="ADK Development UI">

### Evaluating Agents

```bash
adk eval \
    samples_for_testing/hello_world \
    samples_for_testing/hello_world/hello_world_eval_set_001.evalset.json
```

## Integration

### Agent2Agent (A2A) Protocol

ADK seamlessly integrates with the [A2A protocol](https://github.com/google-a2a/A2A/) for remote agent-to-agent communication. See the [example](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents) for practical implementation.

## Contributing

We welcome contributions! Review the [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/) and the [Code Contributing Guidelines](./CONTRIBUTING.md) to get started.

## Vibe Coding

Use the [llms.txt](./llms.txt) (summarized) and [llms-full.txt](./llms-full.txt) (full) files as context for your LLM during vibe coding.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

*Happy Agent Building!*
```
Key improvements and SEO considerations:

*   **Clear Title:**  Uses a strong, keyword-rich title.
*   **One-Sentence Hook:** Grabs attention and concisely describes the ADK's value proposition.
*   **Keyword Optimization:** Includes relevant keywords like "AI agents," "Python toolkit," "open-source," "build," "evaluate," and "deploy" throughout the description and headings.
*   **Structured Headings:** Uses clear headings to improve readability and organization (H2, H3).
*   **Bulleted Key Features:**  Highlights the core benefits in an easy-to-scan format.
*   **Concise Descriptions:**  Provides brief, impactful descriptions of each feature.
*   **Call to Action:** Encourages user engagement.
*   **Internal Linking:**  Links to relevant documentation, samples, and other resources within the project and outside.
*   **External Linking:** Links to the A2A protocol repository.
*   **Improved Formatting:** More consistent and readable markdown.
*   **SEO-Friendly Alt Text:** Added alt text to the image for accessibility and SEO.