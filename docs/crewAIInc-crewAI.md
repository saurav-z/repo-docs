<p align="center">
  <a href="https://github.com/crewAIInc/crewAI">
    <img src="docs/images/crewai_logo.png" width="600px" alt="Open source Multi-AI Agent orchestration framework">
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

## 🚀 Unleash the Power of Multi-Agent AI with CrewAI!

**CrewAI is a lightning-fast, standalone Python framework that empowers developers to build, orchestrate, and deploy autonomous AI agents with unparalleled flexibility and control.**  Explore the future of AI automation with a solution designed for both simplicity and enterprise-grade performance.  [Visit the Original Repo](https://github.com/crewAIInc/crewAI) to get started today!

### Key Features

*   **Standalone & Lean:**  Completely independent of frameworks like LangChain, ensuring faster execution and minimal resource usage.
*   **Crews for Autonomy & Flows for Precision:** Easily orchestrate agents through [Crews](https://docs.crewai.com/concepts/crews) (autonomous teamwork) or [Flows](https://docs.crewai.com/concepts/flows) (event-driven control) for a perfect balance of flexibility and control.
*   **Seamless Integration:**  Effortlessly combine Crews and Flows to create complex, real-world automations with production-ready code.
*   **Deep Customization:**  Tailor every aspect of your AI agents, from high-level workflows to low-level agent behaviors and prompts.
*   **High Performance:** Optimized for speed and minimal resource consumption.
*   **Vibrant Community:** Backed by a rapidly growing community of over **100,000 certified** developers and extensive resources, providing exceptional support and guidance.

## Table of Contents

*   [Why CrewAI?](#why-crewai)
*   [Getting Started](#getting-started)
    *   [Installation](#1-installation)
    *   [Example: Simple Crew with YAML Configuration](#2-setting-up-your-crew-with-the-yaml-configuration)
    *   [Running Your Crew](#3-running-your-crew)
*   [Key Features](#key-features)
*   [Understanding Flows and Crews](#understanding-flows-and-crews)
*   [Examples](#examples)
    *   [Quick Tutorial](#quick-tutorial)
    *   [Write Job Descriptions](#write-job-descriptions)
    *   [Trip Planner](#trip-planner)
    *   [Stock Analysis](#stock-analysis)
    *   [Using Crews and Flows Together](#using-crews-and-flows-together)
*   [Connecting Your Crew to a Model](#connecting-your-crew-to-a-model)
*   [How CrewAI Compares](#how-crewai-compares)
*   [Contribution](#contribution)
    *   [Installing Dependencies](#installing-dependencies)
    *   [Virtual Env](#virtual-env)
    *   [Pre-commit hooks](#pre-commit-hooks)
    *   [Running Tests](#running-tests)
    *   [Running static type checks](#running-static-type-checks)
    *   [Packaging](#packaging)
    *   [Installing Locally](#installing-locally)
*   [Telemetry](#telemetry)
*   [License](#license)
*   [Frequently Asked Questions (FAQ)](#frequently-asked-questions-faq)

## Why CrewAI?

<div align="center" style="margin-bottom: 30px;">
  <img src="docs/images/asset.png" alt="CrewAI Logo" width="100%">
</div>

CrewAI empowers you to build intelligent automations with the best combination of speed, flexibility, and control.  It's ideal for any use case and backed by a thriving community.

*   **Standalone Framework:**  Built from scratch, independent of LangChain or any other agent framework.
*   **High Performance:** Optimized for speed and minimal resource usage, enabling faster execution.
*   **Flexible Low-Level Customization:** Complete freedom to customize both high-level workflows and low-level agent behaviors.
*   **Ideal for Every Use Case:** Proven effective for simple tasks and highly complex, real-world, enterprise-grade scenarios.
*   **Robust Community:** Backed by a rapidly growing community of over **100,000 certified** developers offering comprehensive support and resources.

## Getting Started

Get up and running with CrewAI in minutes!

### 1. Installation

Ensure you have Python >=3.10 <3.14 installed.  CrewAI uses [UV](https://docs.astral.sh/uv/) for dependency management.

```shell
pip install crewai
```

For optional features and tools:

```shell
pip install 'crewai[tools]'
```

#### Troubleshooting Dependencies

1.  **ModuleNotFoundError: No module named 'tiktoken'**:

    ```shell
    pip install 'crewai[embeddings]'
    ```
    or
    ```shell
    pip install 'crewai[tools]'
    ```
2.  **Failed building wheel for tiktoken**:

    *   Ensure Rust compiler is installed.
    *   For Windows: Verify Visual C++ Build Tools are installed.
    *   Try upgrading pip: `pip install --upgrade pip`
    *   If issues persist, use a pre-built wheel: `pip install tiktoken --prefer-binary`

### 2. Setting Up Your Crew with the YAML Configuration

Create a new project:

```shell
crewai create crew <project_name>
```

This generates a project structure:

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

Customize your project by editing these files:

*   `src/my_project/config/agents.yaml`: Define your AI agents.
*   `src/my_project/config/tasks.yaml`: Define tasks for your agents.
*   `src/my_project/crew.py`:  Add custom logic, tools, and arguments.
*   `src/my_project/main.py`:  Add inputs for your agents and tasks.
*   `.env`: Add your environment variables.

#### Example of a Simple Crew:

**agents.yaml**

```yaml
# src/my_project/config/agents.yaml
researcher:
  role: >
    {topic} Senior Data Researcher
  goal: >
    Uncover cutting-edge developments in {topic}
  backstory: >
    You're a seasoned researcher with a knack for uncovering the latest
    developments in {topic}. Known for your ability to find the most relevant
    information and present it in a clear and concise manner.

reporting_analyst:
  role: >
    {topic} Reporting Analyst
  goal: >
    Create detailed reports based on {topic} data analysis and research findings
  backstory: >
    You're a meticulous analyst with a keen eye for detail. You're known for
    your ability to turn complex data into clear and concise reports, making
    it easy for others to understand and act on the information you provide.
```

**tasks.yaml**

```yaml
# src/my_project/config/tasks.yaml
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
# src/my_project/crew.py
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

@CrewBase
class LatestAiDevelopmentCrew():
	"""LatestAiDevelopment crew"""
	agents: List[BaseAgent]
	tasks: List[Task]

	@agent
	def researcher(self) -> Agent:
		return Agent(
			config=self.agents_config['researcher'],
			verbose=True,
			tools=[SerperDevTool()]
		)

	@agent
	def reporting_analyst(self) -> Agent:
		return Agent(
			config=self.agents_config['reporting_analyst'],
			verbose=True
		)

	@task
	def research_task(self) -> Task:
		return Task(
			config=self.tasks_config['research_task'],
		)

	@task
	def reporting_task(self) -> Task:
		return Task(
			config=self.tasks_config['reporting_task'],
			output_file='report.md'
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the LatestAiDevelopment crew"""
		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
		)
```

**main.py**

```python
#!/usr/bin/env python
# src/my_project/main.py
import sys
from latest_ai_development.crew import LatestAiDevelopmentCrew

def run():
    """
    Run the crew.
    """
    inputs = {
        'topic': 'AI Agents'
    }
    LatestAiDevelopmentCrew().crew().kickoff(inputs=inputs)
```

### 3. Running Your Crew

Set these environment variables in your `.env` file:

*   `OPENAI_API_KEY=sk-...`  (or other LLM API key)
*   `SERPER_API_KEY=YOUR_KEY_HERE` (for search functionality)

Navigate to your project directory:

```shell
cd my_project
```

Install dependencies (Optional):

```shell
crewai install (Optional)
```

Run your crew:

```bash
crewai run
```

or

```bash
python src/my_project/main.py
```

If you encounter errors with poetry, update the `crewai` package:

```bash
crewai update
```

The output will appear in your console, and `report.md` will be created with the final report.

## Understanding Flows and Crews

CrewAI offers two powerful approaches:

1.  **Crews:** AI agents with true autonomy, collaborating through role-based interaction.
    *   Autonomous decision-making.
    *   Dynamic task delegation.
    *   Specialized roles.
    *   Flexible problem-solving.
2.  **Flows:** Event-driven workflows for precise control.
    *   Fine-grained control over execution paths.
    *   Secure state management.
    *   Seamless integration with Python code.
    *   Conditional branching.

Combine Crews and Flows to build sophisticated AI applications that balance autonomy with control.

## Examples

Explore real-world examples in the [CrewAI-examples repo](https://github.com/crewAIInc/crewAI-examples?tab=readme-ov-file):

*   [Landing Page Generator](https://github.com/crewAIInc/crewAI-examples/tree/main/landing_page_generator)
*   [Human Input on Execution](https://docs.crewai.com/how-to/Human-Input-on-Execution)
*   [Trip Planner](https://github.com/crewAIInc/crewAI-examples/tree/main/trip_planner)
*   [Stock Analysis](https://github.com/crewAIInc/crewAI-examples/tree/main/stock_analysis)

### Quick Tutorial

[![CrewAI Tutorial](https://img.youtube.com/vi/tnejrr-0a94/maxresdefault.jpg)](https://www.youtube.com/watch?v=tnejrr-0a94 "CrewAI Tutorial")

### Write Job Descriptions

[Check out code for this example](https://github.com/crewAIInc/crewAI-examples/tree/main/job-posting) or watch a video below:

[![Jobs postings](https://img.youtube.com/vi/u98wEMz-9to/maxresdefault.jpg)](https://www.youtube.com/watch?v=u98wEMz-9to "Jobs postings")

### Trip Planner

[Check out code for this example](https://github.com/crewAIInc/crewAI-examples/tree/main/trip_planner) or watch a video below:

[![Trip Planner](https://img.youtube.com/vi/xis7rWp-hjs/maxresdefault.jpg)](https://www.youtube.com/watch?v=xis7rWp-hjs "Trip Planner")

### Stock Analysis

[Check out code for this example](https://github.com/crewAIInc/crewAI-examples/tree/main/stock_analysis) or watch a video below:

[![Stock Analysis](https://img.youtube.com/vi/e0Uj4yWdaAg/maxresdefault.jpg)](https://www.youtube.com/watch?v=e0Uj4yWdaAg "Stock Analysis")

### Using Crews and Flows Together

Combine Crews and Flows to create sophisticated automation pipelines. CrewAI flows support logical operators like `or_` and `and_` to combine multiple conditions, usable with `@start`, `@listen`, or `@router`.

-   `or_`: Triggers when any of the specified conditions are met.
-   `and_`: Triggers when all of the specified conditions are met.

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
        analyst = Agent(
            role="Senior Market Analyst",
            goal="Conduct deep market analysis with expert insight",
            backstory="You're a veteran analyst known for identifying subtle market patterns"
        )
        researcher = Agent(
            role="Data Researcher",
            goal="Gather and validate supporting market data",
            backstory="You excel at finding and correlating multiple data sources"
        )

        analysis_task = Task(
            description="Analyze {sector} sector data for the past {timeframe}",
            expected_output="Detailed market analysis with confidence score",
            agent=analyst
        )
        research_task = Task(
            description="Find supporting data to validate the analysis",
            expected_output="Corroborating evidence and potential contradictions",
            agent=researcher
        )

        analysis_crew = Crew(
            agents=[analyst, researcher],
            tasks=[analysis_task, research_task],
            process=Process.sequential,
            verbose=True
        )
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
        strategy_crew = Crew(
            agents=[
                Agent(role="Strategy Expert",
                      goal="Develop optimal market strategy")
            ],
            tasks=[
                Task(description="Create detailed strategy based on analysis",
                     expected_output="Step-by-step action plan")
            ]
        )
        return strategy_crew.kickoff()

    @listen(or_("medium_confidence", "low_confidence"))
    def request_additional_analysis(self):
        self.state.recommendations.append("Gather more data")
        return "Additional analysis required"
```

## Connecting Your Crew to a Model

CrewAI offers flexible LLM connections.  By default, agents use the OpenAI API.  For details on alternative connections, consult the [Connect CrewAI to LLMs](https://docs.crewai.com/how-to/LLM-Connections/) documentation.

## How CrewAI Compares

**CrewAI's Advantage**: CrewAI provides both autonomous agent intelligence with its unique Crews and precise workflow control through Flows. It excels at high-level orchestration and low-level customization.

*   **LangGraph**: Requires more complex state management and code than CrewAI.  Tight coupling with LangChain can limit flexibility.

*P.S. CrewAI demonstrates significant performance advantages over LangGraph, executing 5.76x faster in certain cases like this QA task example ([see comparison](https://github.com/crewAIInc/crewAI-examples/tree/main/Notebooks/CrewAI%20Flows%20%26%20Langgraph/QA%20Agent)) while achieving higher evaluation scores with faster completion times in certain coding tasks, like in this example ([detailed analysis](https://github.com/crewAIInc/crewAI-examples/blob/main/Notebooks/CrewAI%20Flows%20%26%20Langgraph/Coding%20Assistant/coding_assistant_eval.ipynb)).*

*   **Autogen**:  Autogen lacks an inherent concept of process, which can make it difficult to orchestrate agents' interactions.
*   **ChatDev**:  ChatDev's implementation is rigid, which hinders scalability and flexibility in real-world applications.

## Contribution

We welcome your contributions!

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

CrewAI uses anonymous telemetry to improve the library by focusing on the most-used features.

**NO data is collected** regarding prompts, task descriptions, agents' backstories, tool usage, API calls, responses, or secrets.

Data collected includes:

*   Version of CrewAI
*   Version of Python
*   General OS (e.g., macOS/Windows/Linux)
*   Number of agents and tasks
*   Crew Process being used
*   If Agents are using memory or allowing delegation
*   If Tasks are being executed in parallel or sequentially
*   Language model being used
*   Roles of agents
*   Tool names available

Users can enable `share_crew` to share detailed crew and task execution data.

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

A: CrewAI is a lean and fast Python framework that helps you build, orchestrate, and deploy autonomous AI agents. CrewAI is a standalone framework, making it faster, simpler, and easier to use than alternative agent frameworks.

### Q: How do I install CrewAI?

A: Install CrewAI using pip:

```shell
pip install crewai
```

For additional tools, use:

```shell
pip install 'crewai[tools]'
```

### Q: Does CrewAI depend on LangChain?

A: No. CrewAI is built from scratch without relying on LangChain or other agent frameworks, ensuring a lean, fast experience.

### Q: Can CrewAI handle complex use cases?

A: Yes. CrewAI excels in scenarios of all complexities, offering deep customization options, from internal prompts to sophisticated workflow orchestration.

### Q: Can I use CrewAI with local AI models?

A: Yes, CrewAI supports various language models, including local ones. Consult the [LLM Connections documentation](https://docs.crewai.com/how-to/LLM-Connections/) for more details.

### Q: What makes Crews different from Flows?

A: Crews provide autonomous agent collaboration for flexible decision-making. Flows offer precise control for detailed execution and state management.

### Q: How is CrewAI better than LangChain?

A: CrewAI offers simpler APIs, faster execution speeds, more reliable results, robust documentation, and an active community, making the overall user experience better.

### Q: Is CrewAI open-source?

A: Yes, CrewAI is open-source and welcomes community contributions.

### Q: Does CrewAI collect data from users?

A: CrewAI collects anonymous telemetry data to improve the library without collecting any sensitive data.

### Q: Where can I find real-world CrewAI examples?

A: Check out the [CrewAI-examples repository](https://github.com/crewAIInc/crewAI-examples) for various real-world examples.

### Q: How can I contribute to CrewAI?

A: Contribute by forking the repository, creating a branch, implementing your changes, and submitting a pull request. See the Contribution section.

### Q: What additional features does CrewAI Enterprise offer?

A: CrewAI Enterprise includes a unified control plane, real-time observability, secure integrations, advanced security, actionable insights, and dedicated support.

### Q: Is CrewAI Enterprise available for cloud and on-premise deployments?

A: Yes, CrewAI Enterprise supports both cloud and on-premise deployments.

### Q: Can I try CrewAI Enterprise for free?

A: Yes, explore the [Crew Control Plane](https://app.crewai.com) for free.

### Q: Does CrewAI support fine-tuning or training custom models?

A: Yes, CrewAI can integrate with custom-trained or fine-tuned models.

### Q: Can CrewAI agents interact with external tools and APIs?

A: Yes, CrewAI agents can easily integrate with external tools, APIs, and databases.

### Q: Is CrewAI suitable for production environments?

A: Yes, CrewAI is designed for production environments.

### Q: How scalable is CrewAI?

A: CrewAI is highly scalable.

### Q: Does CrewAI offer debugging and monitoring tools?

A: Yes, CrewAI Enterprise includes advanced debugging, tracing, and real-time observability features.

### Q: What programming languages does CrewAI support?

A: CrewAI is primarily Python-based, but supports APIs written in any language.

### Q: Does CrewAI offer educational resources for beginners?

A: Yes, CrewAI provides tutorials, courses, and documentation through learn.crewai.com.

### Q: Can CrewAI automate human-in-the-loop workflows?

A: Yes, CrewAI supports human-in-the-loop workflows.