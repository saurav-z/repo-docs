<p align="center">
  <a href="https://github.com/crewAIInc/crewAI">
    <img src="docs/images/crewai_logo.png" width="600px" alt="Open source Multi-AI Agent orchestration framework">
  </a>
</p>

<p align="center" style="display: flex; justify-content: center; gap: 20px; align-items: center;">
  <a href="https://trendshift.io/repositories/11239" target="_blank">
    <img src="https://trendshift.io/api/badge/repositories/11239" alt="crewAIInc%2FcrewAI | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/>
  </a>
</p>

<p align="center">
  <a href="https://crewai.com">Homepage</a>
  ·
  <a href="https://docs.crewai.com">Docs</a>
  ·
  <a href="https://app.crewai.com">Start Cloud Trial</a>
  ·
  <a href="https://blog.crewai.com">Blog</a>
  ·
  <a href="https://community.crewai.com">Forum</a>
</p>

<p align="center">
  <a href="https://github.com/crewAIInc/crewAI">
    <img src="https://img.shields.io/github/stars/crewAIInc/crewAI" alt="GitHub Repo stars">
  </a>
  <a href="https://github.com/crewAIInc/crewAI/network/members">
    <img src="https://img.shields.io/github/forks/crewAIInc/crewAI" alt="GitHub forks">
  </a>
  <a href="https://github.com/crewAIInc/crewAI/issues">
    <img src="https://img.shields.io/github/issues/crewAIInc/crewAI" alt="GitHub issues">
  </a>
  <a href="https://github.com/crewAIInc/crewAI/pulls">
    <img src="https://img.shields.io/github/issues-pr/crewAIInc/crewAI" alt="GitHub pull requests">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
  </a>
</p>

<p align="center">
  <a href="https://pypi.org/project/crewai/">
    <img src="https://img.shields.io/pypi/v/crewai" alt="PyPI version">
  </a>
  <a href="https://pypi.org/project/crewai/">
    <img src="https://img.shields.io/pypi/dm/crewai" alt="PyPI downloads">
  </a>
  <a href="https://twitter.com/crewAIInc">
    <img src="https://img.shields.io/twitter/follow/crewAIInc?style=social" alt="Twitter Follow">
  </a>
</p>

## 🚀 Unleash the Power of Autonomous AI: Build Intelligent Automations with CrewAI!

CrewAI is a cutting-edge, open-source Python framework designed to streamline multi-agent AI orchestration, offering unparalleled speed, flexibility, and control. Dive into the future of automation and revolutionize your workflows.

**[Visit the Original Repository on GitHub](https://github.com/crewAIInc/crewAI)**

**Key Features:**

*   ✨ **Standalone & Lean:**  Built from the ground up, independent of LangChain or other agent frameworks, for faster execution and optimized resource usage.
*   🧠 **Intuitive Crews & Precise Flows:** Effortlessly orchestrate autonomous AI agents with Crews for collaborative intelligence, and Flows for granular, event-driven control.
*   🛠️ **Deep Customization:** Tailor every aspect from high-level workflows down to low-level agent behaviors and prompts.
*   🔄 **Seamless Integration:** Easily combine Crews (autonomy) and Flows (precision) to create complex, real-world automations.
*   🚀 **High Performance:** Experience consistently fast results and minimal resource usage.
*   🤝 **Vibrant Community:** Benefit from robust documentation, comprehensive resources, and a thriving community of over 100,000 certified developers.

## Why Choose CrewAI?

<div align="center" style="margin-bottom: 30px;">
  <img src="docs/images/asset.png" alt="CrewAI Logo" width="100%">
</div>

CrewAI empowers you to build intelligent automations with a unique combination of simplicity, flexibility, and performance.

*   **Standalone Framework**: No dependence on LangChain or other agent frameworks.
*   **High Performance**: Optimized for speed and minimal resource usage.
*   **Flexible Low-Level Customization**: Complete freedom in customizing workflows, agent behaviors, prompts, and execution logic.
*   **Ideal for Every Use Case**: Proven effective for simple and highly complex scenarios.
*   **Robust Community**: Backed by a growing community of over 100,000 certified developers.

## Getting Started

Get up and running with CrewAI quickly with this tutorial:

[![CrewAI Getting Started Tutorial](https://img.youtube.com/vi/-kSOTtYzgEw/hqdefault.jpg)](https://www.youtube.com/watch?v=-kSOTtYzgEw "CrewAI Getting Started Tutorial")

### Learning Resources

*   [Multi AI Agent Systems with CrewAI](https://www.deeplearning.ai/short-courses/multi-ai-agent-systems-with-crewai/)
*   [Practical Multi AI Agents and Advanced Use Cases with CrewAI](https://www.deeplearning.ai/short-courses/practical-multi-ai-agents-and-advanced-use-cases-with-crewai/)

## Understanding Flows and Crews

CrewAI offers two powerful approaches:

1.  **Crews:** Autonomous teams of AI agents for collaborative problem-solving.
    *   Autonomous decision-making between agents.
    *   Dynamic task delegation and collaboration.
    *   Specialized roles.
    *   Flexible problem-solving.

2.  **Flows:** Event-driven workflows for precise control.
    *   Fine-grained control over execution paths.
    *   Secure, consistent state management.
    *   Clean integration with production code.
    *   Conditional branching.

Combine Crews and Flows for the best results!

## Getting Started with Installation

### 1. Installation

Ensure you have Python >=3.10 <3.14 installed.

Install CrewAI:

```shell
pip install crewai
```

For optional features:

```shell
pip install 'crewai[tools]'
```

### Troubleshooting Dependencies

*   **ModuleNotFoundError: No module named 'tiktoken'**: `pip install 'crewai[embeddings]'` or `pip install 'crewai[tools]'`
*   **Failed building wheel for tiktoken**: Install Rust, verify Visual C++ Build Tools, try upgrading pip, or use a pre-built wheel.

### 2. Project Setup

Create a project:

```shell
crewai create crew <project_name>
```

Project structure:

```
my_project/
├── .gitignore
├── pyproject.toml
├── README.md
├── .env
└── src/
    └── my_project/
        ├── __init__.py
        ├── main.py
        ├── crew.py
        ├── tools/
        │   ├── custom_tool.py
        │   └── __init__.py
        └── config/
            ├── agents.yaml
            └── tasks.yaml
```

Customize agents in `agents.yaml`, tasks in `tasks.yaml`, crew logic in `crew.py`, and the entry point in `main.py`.

### 3. Example:

**agents.yaml**

```yaml
researcher:
  role: >
    {topic} Senior Data Researcher
  goal: >
    Uncover cutting-edge developments in {topic}
  backstory: >
    You're a seasoned researcher...
reporting_analyst:
  role: >
    {topic} Reporting Analyst
  goal: >
    Create detailed reports...
  backstory: >
    You're a meticulous analyst...
```

**tasks.yaml**

```yaml
research_task:
  description: >
    Conduct a thorough research about {topic}
    Make sure you find any interesting and relevant information given
    the current year is 2025.
  expected_output: >
    A list with 10 bullet points of the most relevant information about {topic}
  agent: researcher
reporting_task:
  description: >
    Review the context you got and expand each topic into a full section for a report.
    Make sure the report is detailed and contains any and all relevant information.
  expected_output: >
    A fully fledge reports with the mains topics, each with a full section of information.
    Formatted as markdown without '```'
  agent: reporting_analyst
  output_file: report.md
```

**crew.py**

```python
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

@CrewBase
class LatestAiDevelopmentCrew():
    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def researcher(self) -> Agent:
        return Agent(...)

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(...)

    @task
    def research_task(self) -> Task:
        return Task(...)

    @task
    def reporting_task(self) -> Task:
        return Task(...)

    @crew
    def crew(self) -> Crew:
        return Crew(...)
```

**main.py**

```python
#!/usr/bin/env python
import sys
from latest_ai_development.crew import LatestAiDevelopmentCrew

def run():
    inputs = {
        'topic': 'AI Agents'
    }
    LatestAiDevelopmentCrew().crew().kickoff(inputs=inputs)
```

### 4. Running Your Crew

Set environment variables:

*   `OPENAI_API_KEY`
*   `SERPER_API_KEY`

Lock and install dependencies:

```shell
cd my_project
crewai install (Optional)
```

Run the crew:

```bash
crewai run
```
or
```bash
python src/my_project/main.py
```

## Key Features

*   **Standalone & Lean**: Completely independent framework.
*   **Flexible & Precise**: Crews and Flows for optimal control.
*   **Seamless Integration**: Combine Crews and Flows.
*   **Deep Customization**: Tailor every aspect.
*   **Reliable Performance**: Consistent results.
*   **Thriving Community**: Extensive resources and support.

## Examples

*   [Landing Page Generator](https://github.com/crewAIInc/crewAI-examples/tree/main/crews/landing_page_generator)
*   [Having Human input on the execution](https://docs.crewai.com/how-to/Human-Input-on-Execution)
*   [Trip Planner](https://github.com/crewAIInc/crewAI-examples/tree/main/crews/trip_planner)
*   [Stock Analysis](https://github.com/crewAIInc/crewAI-examples/tree/main/crews/stock_analysis)

### Quick Tutorial

[![CrewAI Tutorial](https://img.youtube.com/vi/tnejrr-0a94/maxresdefault.jpg)](https://www.youtube.com/watch?v=tnejrr-0a94 "CrewAI Tutorial")

### Write Job Descriptions

[Check out code for this example](https://github.com/crewAIInc/crewAI-examples/tree/main/crews/job-posting) or watch a video below:

[![Jobs postings](https://img.youtube.com/vi/u98wEMz-9to/maxresdefault.jpg)](https://www.youtube.com/watch?v=u98wEMz-9to "Jobs postings")

### Trip Planner

[Check out code for this example](https://github.com/crewAIInc/crewAI-examples/tree/main/crews/trip_planner) or watch a video below:

[![Trip Planner](https://img.youtube.com/vi/xis7rWp-hjs/maxresdefault.jpg)](https://www.youtube.com/watch?v=xis7rWp-hjs "Trip Planner")

### Stock Analysis

[Check out code for this example](https://github.com/crewAIInc/crewAI-examples/tree/main/crews/stock_analysis) or watch a video below:

[![Stock Analysis](https://img.youtube.com/vi/e0Uj4yWdaAg/maxresdefault.jpg)](https://www.youtube.com/watch?v=e0Uj4yWdaAg "Stock Analysis")

### Using Crews and Flows Together

```python
from crewai.flow.flow import Flow, listen, start, router, or_
from crewai import Crew, Agent, Task, Process
from pydantic import BaseModel

class MarketState(BaseModel):
    sentiment: str = "neutral"
    confidence: float = 0.0
    recommendations: list = []

class AdvancedAnalysisFlow(Flow[MarketState]):
    @start()
    def fetch_market_data(self):
        self.state.sentiment = "analyzing"
        return {"sector": "tech", "timeframe": "1W"}

    @listen(fetch_market_data)
    def analyze_with_crew(self, market_data):
        analyst = Agent(...)
        researcher = Agent(...)

        analysis_task = Task(...)
        research_task = Task(...)

        analysis_crew = Crew(...)
        return analysis_crew.kickoff(inputs=market_data)

    @router(analyze_with_crew)
    def determine_next_steps(self):
        if self.state.confidence > 0.8:
            return "high_confidence"
        elif self.state.confidence > 0.5:
            return "medium_confidence"
        return "low_confidence"

    @listen("high_confidence")
    def execute_strategy(self):
        strategy_crew = Crew(...)
        return strategy_crew.kickoff()

    @listen(or_("medium_confidence", "low_confidence"))
    def request_additional_analysis(self):
        self.state.recommendations.append("Gather more data")
        return "Additional analysis required"
```

## Connecting Your Crew to a Model

*   [Connect CrewAI to LLMs](https://docs.crewai.com/how-to/LLM-Connections/)

## How CrewAI Compares

**CrewAI's Advantage**: Autonomous agent intelligence with precise workflow control through its unique Crews and Flows architecture.

-   **LangGraph**:
    *P.S. CrewAI demonstrates significant performance advantages over LangGraph...([see comparison](https://github.com/crewAIInc/crewAI-examples/tree/main/Notebooks/CrewAI%20Flows%20%26%20Langgraph/QA%20Agent)).*
    *...faster completion times in certain coding tasks, like in this example ([detailed analysis](https://github.com/crewAIInc/crewAI-examples/blob/main/Notebooks/CrewAI%20Flows%20%26%20Langgraph/Coding%20Assistant/coding_assistant_eval.ipynb)).*

-   **Autogen**
-   **ChatDev**

## Contribution

*   Fork the repository.
*   Create a new branch for your feature.
*   Add your feature or improvement.
*   Send a pull request.
*   We appreciate your input!

### Installing Dependencies

```bash
uv lock
uv sync
```

### Virtual Env

```bash
uv venv
```

### Pre-commit hooks

```bash
pre-commit install
```

### Running Tests

```bash
uv run pytest .
```

### Running static type checks

```bash
uvx mypy src
```

### Packaging

```bash
uv build
```

### Installing Locally

```bash
pip install dist/*.tar.gz
```

## Telemetry

CrewAI uses anonymous telemetry to collect usage data to improve the library.

-   **NO data is collected** concerning prompts, task descriptions, agents' backstories or goals, usage of tools, API calls, responses, any data processed by the agents, or secrets and environment variables, with the exception of the conditions mentioned.
-   Telemetry can be disabled by setting the environment variable `OTEL_SDK_DISABLED` to true.
-   When `share_crew` is enabled detailed data including task descriptions, agents' backstories or goals, and other specific attributes are collected to provide deeper insights while respecting user privacy.
-   Data collected includes:
    -   Version of CrewAI
    -   Version of Python
    -   General OS
    -   Number of agents and tasks
    -   Crew Process being used
    -   If Agents are using memory or allowing delegation
    -   If Tasks are being executed in parallel or sequentially
    -   Language model being used
    -   Roles of agents in a crew
    -   Tools names available

*   Users can opt-in to Further Telemetry, sharing the complete telemetry data by setting the `share_crew` attribute to `True` on their Crews.

## License

CrewAI is released under the [MIT License](https://github.com/crewAIInc/crewAI/blob/main/LICENSE).

## Frequently Asked Questions (FAQ)

### General

*   [What exactly is CrewAI?](#q-what-exactly-is-crewai)
*   [How do I install CrewAI?](#q-how-do-i-install-crewai)
*   [Does CrewAI depend on LangChain?](#q-does-crewai-depend-on-langchain)
*   [Is CrewAI open-source?](#q-is-crewai-open-source)
*   [Does CrewAI collect data from users?](#q-does-crewai-collect-data-from-users)

### Features and Capabilities

*   [Can CrewAI handle complex use cases?](#q-can-crewai-handle-complex-use-cases)
*   [Can I use CrewAI with local AI models?](#q-can-i-use-crewai-with-local-ai-models)
*   [What makes Crews different from Flows?](#q-what-makes-crews-different-from-flows)
*   [How is CrewAI better than LangChain?](#q-how-is-crewai-better-than-langchain)
*   [Does CrewAI support fine-tuning or training custom models?](#q-does-crewai-support-fine-tuning-or-training-custom-models)

### Resources and Community

*   [Where can I find real-world CrewAI examples?](#q-where-can-i-find-real-world-crewai-examples)
*   [How can I contribute to CrewAI?](#q-how-can-i-contribute-to-crewai)

### Enterprise Features

*   [What additional features does CrewAI Enterprise offer?](#q-what-additional-features-does-crewai-enterprise-offer)
*   [Is CrewAI Enterprise available for cloud and on-premise deployments?](#q-is-crewai-enterprise-available-for-cloud-and-on-premise-deployments)
*   [Can I try CrewAI Enterprise for free?](#q-can-i-try-crewai-enterprise-for-free)

### Q: What exactly is CrewAI?

A: CrewAI is a standalone, lean, and fast Python framework for orchestrating autonomous AI agents. It's built from scratch, independent of LangChain.

### Q: How do I install CrewAI?

A: `pip install crewai`.  For extra tools, `pip install 'crewai[tools]'`.

### Q: Does CrewAI depend on LangChain?

A: No, CrewAI is entirely independent.

### Q: Can CrewAI handle complex use cases?

A: Yes, from simple to enterprise-grade scenarios.

### Q: Can I use CrewAI with local AI models?

A: Yes, using tools like Ollama. See the [LLM Connections documentation](https://docs.crewai.com/how-to/LLM-Connections/).

### Q: What makes Crews different from Flows?

A: Crews are for agent collaboration. Flows offer precise, event-driven control. You can combine them.

### Q: How is CrewAI better than LangChain?

A: CrewAI offers simpler APIs, faster execution, more reliable results, and a more active community.

### Q: Is CrewAI open-source?

A: Yes.

### Q: Does CrewAI collect data from users?

A: Anonymous telemetry for improvements, but never your sensitive data unless sharing `share_crew` is enabled.

### Q: Where can I find real-world CrewAI examples?

A:  In the [CrewAI-examples repository](https://github.com/crewAIInc/crewAI-examples).

### Q: How can I contribute to CrewAI?

A: Fork, branch, create a PR.

### Q: What additional features does CrewAI Enterprise offer?

A:  Unified control plane, real-time observability, integrations, security, insights, and support.

### Q: Is CrewAI Enterprise available for cloud and on-premise deployments?

A: Yes.

### Q: Can I try CrewAI Enterprise for free?

A: Yes, access the [Crew Control Plane](https://app.crewai.com) for free.

### Q: Does CrewAI support fine-tuning or training custom models?

A: Yes.

### Q: Can CrewAI agents interact with external tools and APIs?

A: Absolutely!

### Q: Is CrewAI suitable for production environments?

A: Yes.

### Q: How scalable is CrewAI?

A: Highly scalable.

### Q: Does CrewAI offer debugging and monitoring tools?

A: Yes, in CrewAI Enterprise.

### Q: What programming languages does CrewAI support?

A: Primarily Python, but integrates with others.

### Q: Does CrewAI offer educational resources for beginners?

A: Yes, through learn.crewai.com.

### Q: Can CrewAI automate human-in-the-loop workflows?

A: Yes.