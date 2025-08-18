# Agent Development Kit (ADK): Build Powerful AI Agents with Ease

**Unleash the power of AI agents with Google's Agent Development Kit (ADK), a flexible and open-source Python toolkit for building, evaluating, and deploying sophisticated AI applications.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/google-adk)](https://pypi.org/project/google-adk/)
[![Python Unit Tests](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml/badge.svg)](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/google/adk-python)

[<img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256" alt="Agent Development Kit Logo">](https://github.com/google/adk-python)

ADK provides a code-first approach to agent development, allowing developers to create, deploy, and orchestrate agentic architectures ranging from simple tasks to complex workflows. It is optimized for Gemini and the Google ecosystem but is model-agnostic, deployment-agnostic, and compatible with other frameworks.

**Key Features:**

*   **Rich Tool Ecosystem:** Integrate pre-built tools, custom functions, OpenAPI specifications, and existing tools for diverse agent capabilities.
*   **Code-First Development:** Define agent logic, tools, and orchestration directly in Python for ultimate flexibility, testability, and versioning.
*   **Modular Multi-Agent Systems:** Design scalable applications by composing multiple specialized agents into flexible hierarchies.
*   **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale seamlessly with Vertex AI Agent Engine.

## Agent2Agent (A2A) Protocol and ADK Integration

ADK seamlessly integrates with the [A2A protocol](https://github.com/google-a2a/A2A/) for remote agent-to-agent communication. See this [example](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents) for how they can work together.

## Installation

### Stable Release (Recommended)

Install the latest stable version using pip:

```bash
pip install google-adk
```

*   The release cadence is weekly.
*   Recommended for most users for the latest official release.

### Development Version

Install directly from the main branch for the latest bug fixes and features:

```bash
pip install git+https://github.com/google/adk-python.git@main
```

*   Use for testing upcoming changes or accessing critical fixes before official releases.
*   May contain experimental changes or bugs.

## Documentation

Explore comprehensive guides for building, evaluating, and deploying agents:

*   **[Documentation](https://google.github.io/adk-docs)**

## Quickstart: Example Code

### Define a Single Agent:

```python
from google.adk.agents import Agent
from google.adk.tools import google_search

root_agent = Agent(
    name="search_assistant",
    model="gemini-2.0-flash",  # Or your preferred Gemini model
    instruction="You are a helpful assistant. Answer user questions using Google Search when needed.",
    description="An assistant that can search the web.",
    tools=[google_search]
)
```

### Define a Multi-Agent System:

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
    sub_agents=[  # Assign sub_agents here
        greeter,
        task_executor
    ]
)
```

### Development UI

A built-in development UI to help you test, evaluate, debug, and showcase your agent(s).

<img src="https://raw.githubusercontent.com/google/adk-python/main/assets/adk-web-dev-ui-function-call.png" alt="ADK Development UI Screenshot" />

### Evaluate Agents

```bash
adk eval \
    samples_for_testing/hello_world \
    samples_for_testing/hello_world/hello_world_eval_set_001.evalset.json
```

## Contributing

We welcome community contributions!  Please see our:

*   [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/).
*   [Code Contributing Guidelines](./CONTRIBUTING.md)

## Vibe Coding

Use [llms.txt](./llms.txt) and [llms-full.txt](./llms-full.txt) as context for LLMs when developing agents via vibe coding. The former provides a summarized context, and the latter offers comprehensive information.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

*Ready to build? Dive into agent development with ADK!*

---
[Back to the Original Repository](https://github.com/google/adk-python)
```

Key improvements and SEO considerations:

*   **Clear Headline:**  Strong, keyword-rich headline for improved searchability ("Agent Development Kit," "AI Agents," "Python Toolkit").
*   **One-Sentence Hook:**  Concise and engaging opening that immediately explains the kit's purpose.
*   **Bulleted Key Features:** Highlights the most important aspects of the toolkit, making it easy to scan.
*   **Clear Structure with Headings:**  Organized content for readability and SEO (using `h2` and `h3` where appropriate).
*   **Keyword Optimization:** Includes relevant keywords like "AI agents," "Python toolkit," "agent development," "Gemini," "multi-agent systems," and "Cloud Run," etc.
*   **Action-Oriented Language:** Uses phrases like "Unleash the power," "Build," and "Explore" to encourage engagement.
*   **Alt Text for Images:** Added `alt` text to the image tag for accessibility and SEO.
*   **Link to Original Repo:** Includes a prominent link back to the original repository.
*   **Concise Explanations:** Keeps descriptions brief but informative.
*   **Direct Code Examples:**  Shows users how to get started quickly.
*   **Call to Action:**  Encourages users to "Dive into agent development."
*   **Emphasis on Benefits:** Highlights what users can *do* with the ADK.
*   **Simplified Installation Instructions:**  Improved clarity for both stable and development installations.
*   **Enhanced Readability:** Improved formatting with bullet points and spacing.