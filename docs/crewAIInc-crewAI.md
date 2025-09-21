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

## Revolutionize AI Automation with CrewAI: The Lightning-Fast Framework for Multi-Agent Orchestration

[Explore the CrewAI GitHub Repository](https://github.com/crewAIInc/crewAI) to build advanced, autonomous AI solutions!

CrewAI is a cutting-edge, open-source Python framework designed to empower developers to create and manage multi-agent AI systems with unparalleled speed, flexibility, and control.  **Built from scratch and independent of LangChain**, CrewAI simplifies the complexities of AI orchestration, making it ideal for a wide range of applications—from rapid prototyping to enterprise-grade deployments.

**Key Features:**

*   **Standalone & Lean**:  Completely independent, ensuring faster execution and minimal resource usage.
*   **High Performance**: Optimized for speed and efficiency, allowing for quick development and deployment.
*   **Flexible Orchestration**:  Utilize Crews (agent collaboration) and Flows (event-driven workflows) for versatile automation.
*   **Deep Customization**: Tailor workflows and agent behavior with granular control.
*   **Seamless Integration**:  Easily combine Crews and Flows for sophisticated, real-world automations.
*   **Robust Community**: Backed by extensive documentation and a rapidly growing community of over 100,000 certified developers.
*   **Enterprise-Ready**:  Suitable for both simple projects and complex, enterprise-level applications.

## Table of Contents

*   [Why CrewAI?](#why-crewai)
*   [Getting Started](#getting-started)
*   [Key Features](#key-features)
*   [Understanding Flows and Crews](#understanding-flows-and-crews)
*   [CrewAI vs Other Frameworks](#how-crewai-compares)
*   [Examples](#examples)
    *   [Quick Tutorial](#quick-tutorial)
    *   [Write Job Descriptions](#write-job-descriptions)
    *   [Trip Planner](#trip-planner)
    *   [Stock Analysis](#stock-analysis)
    *   [Using Crews and Flows Together](#using-crews-and-flows-together)
*   [Connecting Your Crew to a Model](#connecting-your-crew-to-a-model)
*   [How CrewAI Compares](#how-crewai-compares)
*   [Frequently Asked Questions (FAQ)](#frequently-asked-questions-faq)
*   [Contribution](#contribution)
*   [Telemetry](#telemetry)
*   [License](#license)

## Why CrewAI?

CrewAI offers the best combination of speed, flexibility, and control, providing both simplicity and precision. It empowers developers and enterprises to confidently build intelligent automations, bridging the gap between simplicity, flexibility, and performance.

<div align="center" style="margin-bottom: 30px;">
  <img src="docs/images/asset.png" alt="CrewAI Logo" width="100%">
</div>

CrewAI is the ideal choice for:

*   **Standalone Framework**: Built from the ground up, independent of LangChain or other agent frameworks.
*   **High Performance**: Optimized for speed and minimal resource usage, enabling faster execution.
*   **Flexible Low-Level Customization**: Complete freedom to customize workflows, system architecture, agent behaviors, internal prompts, and execution logic.
*   **Ideal for Every Use Case**: Effective for both simple tasks and highly complex, real-world, enterprise-grade scenarios.
*   **Robust Community**: Backed by a rapidly growing community of over **100,000 certified** developers offering comprehensive support and resources.

## Getting Started

Follow this quick tutorial to set up and run your first CrewAI agents:

[![CrewAI Getting Started Tutorial](https://img.youtube.com/vi/-kSOTtYzgEw/hqdefault.jpg)](https://www.youtube.com/watch?v=-kSOTtYzgEw "CrewAI Getting Started Tutorial")

### Learning Resources

Master CrewAI with our comprehensive courses:

*   [Multi AI Agent Systems with CrewAI](https://www.deeplearning.ai/short-courses/multi-ai-agent-systems-with-crewai/)
*   [Practical Multi AI Agents and Advanced Use Cases](https://www.deeplearning.ai/short-courses/practical-multi-ai-agents-and-advanced-use-cases-with-crewai/)

### Understanding Flows and Crews

CrewAI offers two powerful, complementary approaches:

1.  **Crews**: Teams of AI agents with true autonomy and agency, working together to accomplish complex tasks through role-based collaboration. Crews enable:

    *   Natural, autonomous decision-making between agents
    *   Dynamic task delegation and collaboration
    *   Specialized roles with defined goals and expertise
    *   Flexible problem-solving approaches

2.  **Flows**: Production-ready, event-driven workflows that deliver precise control over complex automations. Flows provide:

    *   Fine-grained control over execution paths for real-world scenarios
    *   Secure, consistent state management between tasks
    *   Clean integration of AI agents with production Python code
    *   Conditional branching for complex business logic

The true power of CrewAI emerges when combining Crews and Flows. This synergy allows you to:

*   Build complex, production-grade applications
*   Balance autonomy with precise control
*   Handle sophisticated real-world scenarios
*   Maintain clean, maintainable code structure

### Getting Started with Installation

Follow these steps to install CrewAI:

### 1. Installation

Ensure you have Python >=3.10 <3.14 installed.  CrewAI uses [UV](https://docs.astral.sh/uv/) for dependency management.

```shell
pip install crewai
```

For optional tools, including extra components that require more dependencies:

```shell
pip install 'crewai[tools]'
```

### Troubleshooting Dependencies

Common issues and solutions:

#### Common Issues

1.  **ModuleNotFoundError: No module named 'tiktoken'**

    *   Install tiktoken explicitly: `pip install 'crewai[embeddings]'`
    *   If using embedchain or other tools: `pip install 'crewai[tools]'`
2.  **Failed building wheel for tiktoken**

    *   Ensure Rust compiler is installed (see installation steps above)
    *   For Windows: Verify Visual C++ Build Tools are installed
    *   Try upgrading pip: `pip install --upgrade pip`
    *   If issues persist, use a pre-built wheel: `pip install tiktoken --prefer-binary`

### 2. Setting Up Your Crew with the YAML Configuration

To create a new CrewAI project:

```shell
crewai create crew <project_name>
```

This creates a project folder with the structure:

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

Customize your project by editing the files in `src/my_project`:

*   `src/my_project/config/agents.yaml`: Define agents.
*   `src/my_project/config/tasks.yaml`: Define tasks.
*   `src/my_project/crew.py`: Add logic, tools, and arguments.
*   `src/my_project/main.py`: Add custom inputs for agents and tasks.
*   `.env`: Add environment variables.

#### Example of a simple crew:

Instantiate your crew:

```shell
crewai create crew latest-ai-development
```

Modify the files as needed to fit your use case:

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

Set the following environment variables in your `.env` file:

*   `OPENAI_API_KEY=sk-...` (or other LLM API key)
*   `SERPER_API_KEY=YOUR_KEY_HERE`

Lock dependencies:

```shell
cd my_project
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

If you encounter poetry issues, update the `crewai` package:

```bash
crewai update
```

The output will appear in the console, and `report.md` will be created.

The hierarchical process is also available to coordinate the planning and execution of tasks through delegation and validation of results. [See more about the processes here](https://docs.crewai.com/core-concepts/Processes/).

## Key Features

CrewAI stands out as a lean, standalone, high-performance multi-AI Agent framework, offering simplicity, flexibility, and precise control—free from the complexities of other agent frameworks.

*   **Standalone & Lean**: Completely independent, ensuring fast execution and minimal resource usage.
*   **Flexible & Precise**: Orchestrate agents through intuitive [Crews](https://docs.crewai.com/concepts/crews) or precise [Flows](https://docs.crewai.com/concepts/flows).
*   **Seamless Integration**: Combine Crews (autonomy) and Flows (precision) for complex automations.
*   **Deep Customization**: Tailor workflows with full control over every aspect.
*   **Reliable Performance**: Consistent results across both simple and enterprise-grade automations.
*   **Thriving Community**:  Supported by comprehensive documentation and a rapidly growing community.

## Examples

Explore real-life examples of AI crews in the [CrewAI-examples repo](https://github.com/crewAIInc/crewAI-examples?tab=readme-ov-file):

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

Combine Crews with Flows to create sophisticated automation pipelines.
CrewAI flows support logical operators like `or_` and `and_` to combine multiple conditions. This can be used with `@start`, `@listen`, or `@router` decorators to create complex triggering conditions.

- `or_`: Triggers when any of the specified conditions are met.
- `and_`Triggers when all of the specified conditions are met.

Example of how to orchestrate multiple Crews within a Flow:

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

CrewAI supports various LLMs. Your agents will use the OpenAI API by default. You can also configure your agents to use a local model via the Ollama tool or other options.

Refer to the [Connect CrewAI to LLMs](https://docs.crewai.com/how-to/LLM-Connections/) page for details.

## How CrewAI Compares

**CrewAI's Advantage**: CrewAI combines autonomous agent intelligence with precise workflow control through its unique Crews and Flows architecture. The framework excels at both high-level orchestration and low-level customization, enabling complex, production-grade systems with granular control.

-   **LangGraph**: While LangGraph provides a foundation for building agent workflows, its approach requires significant boilerplate code and complex state management patterns. The framework's tight coupling with LangChain can limit flexibility when implementing custom agent behaviors or integrating with external systems.

*P.S. CrewAI demonstrates significant performance advantages over LangGraph, executing 5.76x faster in certain cases like this QA task example ([see comparison](https://github.com/crewAIInc/crewAI-examples/tree/main/Notebooks/CrewAI%20Flows%20%26%20Langgraph/QA%20Agent)) while achieving higher evaluation scores with faster completion times in certain coding tasks, like in this example ([detailed analysis](https://github.com/crewAIInc/crewAI-examples/blob/main/Notebooks/CrewAI%20Flows%20%26%20Langgraph/Coding%20Assistant/coding_assistant_eval.ipynb)).*

-   **Autogen**: While Autogen excels at creating conversational agents capable of working together, it lacks an inherent concept of process. In Autogen, orchestrating agents' interactions requires additional programming, which can become complex and cumbersome as the scale of tasks grows.
-   **ChatDev**: ChatDev introduced the idea of processes into the realm of AI agents, but its implementation is quite rigid. Customizations in ChatDev are limited and not geared towards production environments, which can hinder scalability and flexibility in real-world applications.

## Contribution

CrewAI is open-source and welcomes contributions. To contribute:

*   Fork the repository.
*   Create a new branch.
*   Add your feature or improvement.
*   Submit a pull request.

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

CrewAI uses anonymous telemetry to help improve the library by focusing on the most used features, integrations and tools.

**NO data is collected** concerning prompts, task descriptions, agents' backstories or goals, usage of tools, API calls, responses, or secrets and environment variables, with the exception of the conditions mentioned. When the `share_crew` feature is enabled, detailed data including task descriptions, agents' backstories or goals, and other specific attributes are collected to provide deeper insights while respecting user privacy. Users can disable telemetry by setting the environment variable OTEL_SDK_DISABLED to true.

Data collected includes:

*   Version of CrewAI
*   Version of Python
*   General OS (e.g. number of CPUs, macOS/Windows/Linux)
*   Number of agents and tasks in a crew
*   Crew Process being used
*   If Agents are using memory or allowing delegation
*   If Tasks are being executed in parallel or sequentially
*   Language model being used
*   Roles of agents in a crew
*   Tools names available

Users can opt-in to Further Telemetry, sharing the complete telemetry data by setting the `share_crew` attribute to `True` on their Crews. Enabling `share_crew` results in the collection of detailed crew and task execution data, including `goal`, `backstory`, `context`, and `output` of tasks. This enables a deeper insight into usage patterns while respecting the user's choice to share.

## License

CrewAI is released under the [MIT License](https://github.com/crewAIInc/crewAI/blob/main/LICENSE).

## Frequently Asked Questions (FAQ)

### General

*   [What is CrewAI?](#q-what-exactly-is-crewai)
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

### Q: What is CrewAI?

A: CrewAI is a standalone, lean, and fast Python framework built specifically for orchestrating autonomous AI agents, without dependencies on other frameworks.

### Q: How do I install CrewAI?

A:  Install CrewAI using pip:

```shell
pip install crewai
```

For additional tools, use:

```shell
pip install 'crewai[tools]'
```

### Q: Does CrewAI depend on LangChain?

A: No, CrewAI is built independently.

### Q: Can CrewAI handle complex use cases?

A: Yes, CrewAI is suitable for both simple and complex applications.

### Q: Can I use CrewAI with local AI models?

A: Yes, through tools like Ollama and LM Studio.  See the [LLM Connections documentation](https://docs.crewai.com/how-to/LLM-Connections/).

### Q: What makes Crews different from Flows?

A: Crews: agent collaboration; Flows: event-driven control.  Use both together.

### Q: How is CrewAI better than LangChain?

A: CrewAI offers simpler APIs, faster execution, and better results with an active community.

### Q: Is CrewAI open-source?

A: Yes.

### Q: Does CrewAI collect data from users?

A: Anonymous telemetry data is collected for improvement purposes. Sensitive data is not collected unless users choose to share more information.

### Q: Where can I find real-world CrewAI examples?

A: In the [CrewAI-examples repository](https://github.com/crewAIInc/crewAI-examples).

### Q: How can I contribute to CrewAI?

A: Fork the repository, create a branch, make changes, and submit a pull request.

### Q: What additional features does CrewAI Enterprise offer?

A: Control plane, real-time observability, secure integrations, advanced security, actionable insights, and dedicated support.

### Q: Is CrewAI Enterprise available for cloud and on-premise deployments?

A: Yes.

### Q: Can I try CrewAI Enterprise for free?

A: Yes, try the [Crew Control Plane](https://app.crewai.com).

### Q: Does CrewAI support fine-tuning or training custom models?

A: Yes, it integrates with custom models.

### Q: Can CrewAI agents interact with external tools and APIs?

A: Absolutely!

### Q: Is CrewAI suitable for production environments?

A: Yes, designed for production use.

### Q: How scalable is CrewAI?

A: Highly scalable for both simple and large-scale deployments.

### Q: Does CrewAI offer debugging and monitoring tools?

A: Yes, CrewAI Enterprise includes advanced debugging and monitoring features.

### Q: What programming languages does CrewAI support?

A: Primarily Python, but easily integrates with other languages.

### Q: Does CrewAI offer educational resources for beginners?

A: Yes, extensive tutorials and courses are provided.

### Q: Can CrewAI automate human-in-the-loop workflows?

A: Yes.