<p align="center">
  <a href="https://github.com/crewAIInc/crewAI">
    <img src="docs/images/crewai_logo.png" width="600px" alt="Open source Multi-AI Agent orchestration framework">
  </a>
</p>

<p align="center">
  **Unleash the Power of Multi-Agent AI: Build Autonomous, Intelligent Systems with CrewAI**
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

## CrewAI: The Ultimate Framework for Multi-Agent AI Automation

CrewAI is a cutting-edge, **standalone Python framework** designed to revolutionize how you build and deploy autonomous AI agents, offering unparalleled speed, flexibility, and control.  Dive into the [original repo](https://github.com/crewAIInc/crewAI) to get started!

**Key Features:**

*   🚀 **Standalone & Lean:** Built from scratch, independent of LangChain, for blazing-fast execution and minimal overhead.
*   🧠 **Flexible Orchestration:** Easily create autonomous agents with Crews, or control agent interactions using precise Flows.
*   🔗 **Seamless Integration:** Combine Crews and Flows for complex, real-world automation pipelines.
*   🛠️ **Deep Customization:** Tailor every detail, from high-level workflows to low-level agent behavior and internal prompts.
*   💯 **Reliable Performance:** Consistent results across simple tasks and enterprise-level automations.
*   🤝 **Thriving Community:** Backed by comprehensive documentation and a rapidly growing community with over 100,000 certified developers offering exceptional support.

## Table of Contents

-   [Why CrewAI?](#why-crewai)
-   [Getting Started](#getting-started)
    -   [Installation](#getting-started-installation)
    -   [Creating a New Project](#creating-a-new-project)
    -   [Running Your Crew](#running-your-crew)
-   [Key Features](#key-features)
-   [Examples](#examples)
    -   [Quick Tutorial](#quick-tutorial)
    -   [Write Job Descriptions](#write-job-descriptions)
    -   [Trip Planner](#trip-planner)
    -   [Stock Analysis](#stock-analysis)
    -   [Using Crews and Flows Together](#using-crews-and-flows-together)
-   [Connecting Your Crew to a Model](#connecting-your-crew-to-a-model)
-   [How CrewAI Compares](#how-crewai-compares)
-   [Contribution](#contribution)
    -   [Installing Dependencies](#installing-dependencies)
    -   [Virtual Env](#virtual-env)
    -   [Pre-commit hooks](#pre-commit-hooks)
    -   [Running Tests](#running-tests)
    -   [Running static type checks](#running-static-type-checks)
    -   [Packaging](#packaging)
    -   [Installing Locally](#installing-locally)
-   [Telemetry](#telemetry)
-   [License](#license)
-   [Frequently Asked Questions (FAQ)](#frequently-asked-questions-faq)
    -   [General](#general)
    -   [Features and Capabilities](#features-and-capabilities)
    -   [Resources and Community](#resources-and-community)
    -   [Enterprise Features](#enterprise-features)
    -   [Q&A](#q-a)
       -   [What exactly is CrewAI?](#q-what-exactly-is-crewai)
       -   [How do I install CrewAI?](#q-how-do-i-install-crewai)
       -   [Does CrewAI depend on LangChain?](#q-does-crewai-depend-on-langchain)
       -   [Can CrewAI handle complex use cases?](#q-can-crewai-handle-complex-use-cases)
       -   [Can I use CrewAI with local AI models?](#q-can-i-use-crewai-with-local-ai-models)
       -   [What makes Crews different from Flows?](#q-what-makes-crews-different-from-flows)
       -   [How is CrewAI better than LangChain?](#q-how-is-crewai-better-than-langchain)
       -   [Is CrewAI open-source?](#q-is-crewai-open-source)
       -   [Does CrewAI collect data from users?](#q-does-crewai-collect-data-from-users)
       -   [Where can I find real-world CrewAI examples?](#q-where-can-i-find-real-world-crewai-examples)
       -   [How can I contribute to CrewAI?](#q-how-can-i-contribute-to-crewai)
       -   [What additional features does CrewAI Enterprise offer?](#q-what-additional-features-does-crewai-enterprise-offer)
       -   [Is CrewAI Enterprise available for cloud and on-premise deployments?](#q-is-crewai-enterprise-available-for-cloud-and-on-premise-deployments)
       -   [Can I try CrewAI Enterprise for free?](#q-can-i-try-crewai-enterprise-for-free)
       -   [Does CrewAI support fine-tuning or training custom models?](#q-does-crewai-support-fine-tuning-or-training-custom-models)
       -   [Can CrewAI agents interact with external tools and APIs?](#q-can-crewai-agents-interact-with-external-tools-and-apis)
       -   [Is CrewAI suitable for production environments?](#q-is-crewai-suitable-for-production-environments)
       -   [How scalable is CrewAI?](#q-how-scalable-is-crewai)
       -   [Does CrewAI offer debugging and monitoring tools?](#q-does-crewai-offer-debugging-and-monitoring-tools)
       -   [What programming languages does CrewAI support?](#q-what-programming-languages-does-crewai-support)
       -   [Does CrewAI offer educational resources for beginners?](#q-does-crewai-offer-educational-resources-for-beginners)
       -   [Can CrewAI automate human-in-the-loop workflows?](#q-can-crewai-automate-human-in-the-loop-workflows)

## Why CrewAI?

<div align="center" style="margin-bottom: 30px;">
  <img src="docs/images/asset.png" alt="CrewAI Logo" width="100%">
</div>

CrewAI provides the best-in-class combination of speed, flexibility, and control.

-   **Standalone Framework**: Built from scratch, independent of LangChain or any other agent framework.
-   **High Performance**: Optimized for speed and minimal resource usage, enabling faster execution.
-   **Flexible Low Level Customization**: Complete freedom to customize at both high and low levels - from overall workflows and system architecture to granular agent behaviors, internal prompts, and execution logic.
-   **Ideal for Every Use Case**: Proven effective for both simple tasks and highly complex, real-world, enterprise-grade scenarios.
-   **Robust Community**: Backed by a rapidly growing community of over **100,000 certified** developers offering comprehensive support and resources.

CrewAI empowers developers and enterprises to confidently build intelligent automations, bridging the gap between simplicity, flexibility, and performance.

## Getting Started

Follow these steps to set up and run your first CrewAI agents:

<a id="getting-started-installation"></a>
### Installation

Ensure you have Python >=3.10 <3.14 installed.  CrewAI uses [UV](https://docs.astral.sh/uv/) for dependency management and package handling.

1.  Install CrewAI:

    ```bash
    pip install crewai
    ```

2.  For optional features including additional tools:

    ```bash
    pip install 'crewai[tools]'
    ```

#### Troubleshooting Dependencies

1.  **ModuleNotFoundError: No module named 'tiktoken'**

    *   Install tiktoken explicitly: `pip install 'crewai[embeddings]'`
    *   If using embedchain or other tools: `pip install 'crewai[tools]'`
2.  **Failed building wheel for tiktoken**

    *   Ensure Rust compiler is installed (see installation steps above)
    *   For Windows: Verify Visual C++ Build Tools are installed
    *   Try upgrading pip: `pip install --upgrade pip`
    *   If issues persist, use a pre-built wheel: `pip install tiktoken --prefer-binary`

<a id="creating-a-new-project"></a>
### Creating a New Project

Run the following CLI command:

```bash
crewai create crew <project_name>
```

This creates a new project folder with the following structure:

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

Customize your project by modifying files within the `src/my_project` folder:

*   `src/my_project/config/agents.yaml`: Define your agents.
*   `src/my_project/config/tasks.yaml`: Define your tasks.
*   `src/my_project/crew.py`: Add logic, tools, and specific arguments.
*   `src/my_project/main.py`: Add custom inputs for agents and tasks.
*   `.env`: Add environment variables.

#### Example: Simple Crew with Sequential Process

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

<a id="running-your-crew"></a>
### Running Your Crew

1.  Set the following environment variables in your `.env` file:

    *   `OPENAI_API_KEY`: Your OpenAI API key (or other LLM API key).
    *   `SERPER_API_KEY`: Your Serper.dev API key.

2.  Navigate to your project directory and lock dependencies:

    ```bash
    cd my_project
    crewai install (Optional)
    ```

3.  Run your crew:

    ```bash
    crewai run
    ```

    or

    ```bash
    python src/my_project/main.py
    ```

    *If you encounter poetry-related errors, update your CrewAI package:*

    ```bash
    crewai update
    ```

You should see output in the console, and `report.md` will be created in your project's root.

## Key Features

CrewAI differentiates itself as a lean, standalone, high-performance multi-AI Agent framework.

-   **Standalone & Lean**: Completely independent from other frameworks like LangChain, offering faster execution and lighter resource demands.
-   **Flexible & Precise**: Easily orchestrate autonomous agents through intuitive [Crews](https://docs.crewai.com/concepts/crews) or precise [Flows](https://docs.crewai.com/concepts/flows), achieving perfect balance for your needs.
-   **Seamless Integration**: Effortlessly combine Crews (autonomy) and Flows (precision) to create complex, real-world automations.
-   **Deep Customization**: Tailor every aspect—from high-level workflows down to low-level internal prompts and agent behaviors.
-   **Reliable Performance**: Consistent results across simple tasks and complex, enterprise-level automations.
-   **Thriving Community**: Backed by robust documentation and over 100,000 certified developers, providing exceptional support and guidance.

## Examples

Explore various real-life AI crew examples:

-   [CrewAI-examples repo](https://github.com/crewAIInc/crewAI-examples?tab=readme-ov-file):

    -   [Landing Page Generator](https://github.com/crewAIInc/crewAI-examples/tree/main/landing_page_generator)
    -   [Having Human input on the execution](https://docs.crewai.com/how-to/Human-Input-on-Execution)
    -   [Trip Planner](https://github.com/crewAIInc/crewAI-examples/tree/main/trip_planner)
    -   [Stock Analysis](https://github.com/crewAIInc/crewAI-examples/tree/main/stock_analysis)

<a id="quick-tutorial"></a>
### Quick Tutorial

[![CrewAI Tutorial](https://img.youtube.com/vi/tnejrr-0a94/maxresdefault.jpg)](https://www.youtube.com/watch?v=tnejrr-0a94 "CrewAI Tutorial")

<a id="write-job-descriptions"></a>
### Write Job Descriptions

[Check out code for this example](https://github.com/crewAIInc/crewAI-examples/tree/main/job-posting) or watch a video below:

[![Jobs postings](https://img.youtube.com/vi/u98wEMz-9to/maxresdefault.jpg)](https://www.youtube.com/watch?v=u98wEMz-9to "Jobs postings")

<a id="trip-planner"></a>
### Trip Planner

[Check out code for this example](https://github.com/crewAIInc/crewAI-examples/tree/main/trip_planner) or watch a video below:

[![Trip Planner](https://img.youtube.com/vi/xis7rWp-hjs/maxresdefault.jpg)](https://www.youtube.com/watch?v=xis7rWp-hjs "Trip Planner")

<a id="stock-analysis"></a>
### Stock Analysis

[Check out code for this example](https://github.com/crewAIInc/crewAI-examples/tree/main/stock_analysis) or watch a video below:

[![Stock Analysis](https://img.youtube.com/vi/e0Uj4yWdaAg/maxresdefault.jpg)](https://www.youtube.com/watch?v=e0Uj4yWdaAg "Stock Analysis")

<a id="using-crews-and-flows-together"></a>
### Using Crews and Flows Together

CrewAI's power truly shines when combining Crews with Flows to create sophisticated automation pipelines.
CrewAI flows support logical operators like `or_` and `and_` to combine multiple conditions. This can be used with `@start`, `@listen`, or `@router` decorators to create complex triggering conditions.

-   `or_`: Triggers when any of the specified conditions are met.
-   `and_`: Triggers when all of the specified conditions are met.

Here's how you can orchestrate multiple Crews within a Flow:

```python
from crewai.flow.flow import Flow, listen, start, router, or_
from crewai import Crew, Agent, Task, Process
from pydantic import BaseModel

# Define structured state for precise control
class MarketState(BaseModel):
    sentiment: str = "neutral"
    confidence: float = 0.0
    recommendations: list = []

class AdvancedAnalysisFlow(Flow[MarketState]):
    @start()
    def fetch_market_data(self):
        # Demonstrate low-level control with structured state
        self.state.sentiment = "analyzing"
        return {"sector": "tech", "timeframe": "1W"}  # These parameters match the task description template

    @listen(fetch_market_data)
    def analyze_with_crew(self, market_data):
        # Show crew agency through specialized roles
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

        # Demonstrate crew autonomy
        analysis_crew = Crew(
            agents=[analyst, researcher],
            tasks=[analysis_task, research_task],
            process=Process.sequential,
            verbose=True
        )
        return analysis_crew.kickoff(inputs=market_data)  # Pass market_data as named inputs

    @router(analyze_with_crew)
    def determine_next_steps(self):
        # Show flow control with conditional routing
        if self.state.confidence > 0.8:
            return "high_confidence"
        elif self.state.confidence > 0.5:
            return "medium_confidence"
        return "low_confidence"

    @listen("high_confidence")
    def execute_strategy(self):
        # Demonstrate complex decision making
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

This example demonstrates how to:

1.  Use Python code for basic data operations
2.  Create and execute Crews as steps in your workflow
3.  Use Flow decorators to manage the sequence of operations
4.  Implement conditional branching based on Crew results

## Connecting Your Crew to a Model

CrewAI supports using various LLMs through a variety of connection options. By default your agents will use the OpenAI API when querying the model. However, there are several other ways to allow your agents to connect to models. For example, you can configure your agents to use a local model via the Ollama tool.

Please refer to the [Connect CrewAI to LLMs](https://docs.crewai.com/how-to/LLM-Connections/) page for details on configuring your agents' connections to models.

## How CrewAI Compares

**CrewAI's Advantage**: CrewAI combines autonomous agent intelligence with precise workflow control through its unique Crews and Flows architecture. The framework excels at both high-level orchestration and low-level customization, enabling complex, production-grade systems with granular control.

-   **LangGraph**: While LangGraph provides a foundation for building agent workflows, its approach requires significant boilerplate code and complex state management patterns. The framework's tight coupling with LangChain can limit flexibility when implementing custom agent behaviors or integrating with external systems.

    *P.S. CrewAI demonstrates significant performance advantages over LangGraph, executing 5.76x faster in certain cases like this QA task example ([see comparison](https://github.com/crewAIInc/crewAI-examples/tree/main/Notebooks/CrewAI%20Flows%20%26%20Langgraph/QA%20Agent)) while achieving higher evaluation scores with faster completion times in certain coding tasks, like in this example ([detailed analysis](https://github.com/crewAIInc/crewAI-examples/blob/main/Notebooks/CrewAI%20Flows%20%26%20Langgraph/Coding%20Assistant/coding_assistant_eval.ipynb)).*

-   **Autogen**: While Autogen excels at creating conversational agents capable of working together, it lacks an inherent concept of process. In Autogen, orchestrating agents' interactions requires additional programming, which can become complex and cumbersome as the scale of tasks grows.
-   **ChatDev**: ChatDev introduced the idea of processes into the realm of AI agents, but its implementation is quite rigid. Customizations in ChatDev are limited and not geared towards production environments, which can hinder scalability and flexibility in real-world applications.

## Contribution

CrewAI is open-source and we welcome contributions.

<a id="installing-dependencies"></a>
### Installing Dependencies

```bash
uv lock
uv sync
```

<a id="virtual-env"></a>
### Virtual Env

```bash
uv venv
```

<a id="pre-commit-hooks"></a>
### Pre-commit hooks

```bash
pre-commit install
```

<a id="running-tests"></a>
### Running Tests

```bash
uv run pytest .
```

<a id="running-static-type-checks"></a>
### Running static type checks

```bash
uvx mypy src
```

<a id="packaging"></a>
### Packaging

```bash
uv build
```

<a id="installing-locally"></a>
### Installing Locally

```bash
pip install dist/*.tar.gz
```

## Telemetry

CrewAI uses anonymous telemetry to collect usage data with the main purpose of helping us improve the library by focusing our efforts on the most used features, integrations and tools.

It's pivotal to understand that **NO data is collected** concerning prompts, task descriptions, agents' backstories or goals, usage of tools, API calls, responses, any data processed by the agents, or secrets and environment variables, with the exception of the conditions mentioned. When the `share_crew` feature is enabled, detailed data including task descriptions, agents' backstories or goals, and other specific attributes are collected to provide deeper insights while respecting user privacy. Users can disable telemetry by setting the environment variable OTEL_SDK_DISABLED to true.

Data collected includes:

-   Version of CrewAI
    -   So we can understand how many users are using the latest version
-   Version of Python
    -   So we can decide on what versions to better support
-   General OS (e.g. number of CPUs, macOS/Windows/Linux)
    -   So we know what OS we should focus on and if we could build specific OS related features
-   Number of agents and tasks in a crew
    -   So we make sure we are testing internally with similar use cases and educate people on the best practices
-   Crew Process being used
    -   Understand where we should focus our efforts
-   If Agents are using memory or allowing delegation
    -   Understand if we improved the features or maybe even drop them
-   If Tasks are being executed in parallel or sequentially
    -   Understand if we should focus more on parallel execution
-   Language model being used
    -   Improved support on most used languages
-   Roles of agents in a crew
    -   Understand high level use cases so we can build better tools, integrations and examples about it
-   Tools names available
    -   Understand out of the publicly available tools, which ones are being used the most so we can improve them

Users can opt-in to Further Telemetry, sharing the complete telemetry data by setting the `share_crew` attribute to `True` on their Crews. Enabling `share_crew` results in the collection of detailed crew and task execution data, including `goal`, `backstory`, `context`, and `output` of tasks. This enables a deeper insight into usage patterns while respecting the user's choice to share.

## License

CrewAI is released under the [MIT License](https://github.com/crewAIInc/crewAI/blob/main/LICENSE).

## Frequently Asked Questions (FAQ)

<a id="general"></a>
### General

-   [What exactly is CrewAI?](#q-what-exactly-is-crewai)
-   [How do I install CrewAI?](#q-how-do-i-install-crewai)
-   [Does CrewAI depend on LangChain?](#q-does-crewai-depend-on-langchain)
-   [Can CrewAI handle complex use cases?](#q-can-crewai-handle-complex-use-cases)
-   [Can I use CrewAI with local AI models?](#q-can-i-use-crewai-with-local-ai-models)
-   [What makes Crews different from Flows?](#q-what-makes-crews-different-from-flows)
-   [How is CrewAI better than LangChain?](#q-how-is-crewai-better-than-langchain)
-   [Is CrewAI open-source?](#q-is-crewai-open-source)
-   [Does CrewAI collect data from users?](#q-does-crewai-collect-data-from-users)

<a id="features-and-capabilities"></a>
### Features and Capabilities

-   [Where can I find real-world CrewAI examples?](#q-where-can-i-find-real-world-crewai-examples)
-   [How can I contribute to CrewAI?](#q-how-can-i-contribute-to-crewai)

<a id="resources-and-community"></a>
### Resources and Community

-   [What additional features does CrewAI Enterprise offer?](#q-what-additional-features-does-crewai-enterprise-offer)
-   [Is CrewAI Enterprise available for cloud and on-premise deployments?](#q-is-crewai-enterprise-available-for-cloud-and-on-premise-deployments)
-   [Can I try CrewAI Enterprise for free?](#q-can-i-try-crewai-enterprise-for-free)
-   [Does CrewAI support fine-tuning or training custom models?](#q-does-crewai-support-fine-tuning-or-training-custom-models)
-   [Can CrewAI agents interact with external tools and APIs?](#q-can-crewai-agents-interact-with-external-tools-and-apis)
-   [Is CrewAI suitable for production environments?](#q-is-crewai-suitable-for-production-environments)
-   [How scalable is CrewAI?](#q-how-scalable-is-crewai)
-   [Does CrewAI offer debugging and monitoring tools?](#q-does-crewai-offer-debugging-and-monitoring-tools)
-   [What programming languages does CrewAI support?](#q-what-programming-languages-does-crewai-support)
-   [Does CrewAI offer educational resources for beginners?](#q-does-crewai-offer-educational-resources-for-beginners)
-   [Can CrewAI automate human-in-the-loop workflows?](#q-can-crewai-automate-human-in-the-loop-workflows)

<a id="enterprise-features"></a>
### Enterprise Features

-   [What additional features does CrewAI Enterprise offer?](#q-what-additional-features-does-crewai-enterprise-offer)
-   [Is CrewAI Enterprise available for cloud and on-premise deployments?](#q-is-crewai-enterprise-available-for-cloud-and-on-premise-deployments)
-   [Can I try CrewAI Enterprise for free?](#q-can-i-try-crewai-enterprise-for-free)
-   [Does CrewAI support fine-tuning or training custom models?](#q-does-crewai-support-fine-tuning-or-training-custom-models)
-   [Can CrewAI agents interact with external tools and APIs?](#q-can-crewai-agents-interact-with-external-tools-and-apis)
-   [Is CrewAI suitable for production environments?](#q-is-crewai-suitable-for-production-environments)
-   [How scalable is CrewAI?](#q-how-scalable-is-crewai)
-   [Does CrewAI offer debugging and monitoring tools?](#q-does-crewai-offer-debugging-and-monitoring-tools)
-   [What programming languages does CrewAI support?](#q-what-programming-languages-does-crewai-support)
-   [Does CrewAI offer educational resources for beginners?](#q-does-crewai-offer-educational-resources-for-beginners)
-   [Can CrewAI automate human-in-the-loop workflows?](#q-can-crewai-automate-human-in-the-loop-workflows)
<a id="q-a"></a>
### Q&A

#### Q: What exactly is CrewAI?

A: CrewAI is a standalone, lean, and fast Python framework built specifically for orchestrating autonomous AI agents. Unlike frameworks like LangChain, CrewAI does not rely on external dependencies, making it leaner, faster, and simpler.

#### Q: How do I install CrewAI?

A: Install CrewAI using pip:

```shell
pip install crewai
```

For additional tools, use:

```shell
pip install 'crewai[tools]'
```

#### Q: Does CrewAI depend on LangChain?

A: No. CrewAI is built entirely from the ground up, with no dependencies on LangChain or other agent frameworks. This ensures a lean, fast, and flexible experience.

#### Q: Can CrewAI handle complex use cases?

A: Yes. CrewAI excels at both simple and highly complex real-world scenarios