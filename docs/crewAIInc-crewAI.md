<p align="center">
  <a href="https://github.com/crewAIInc/crewAI">
    <img src="docs/images/crewai_logo.png" width="600px" alt="CrewAI: Open Source Multi-Agent Orchestration">
  </a>
</p>

<p align="center">
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

## **Supercharge Your AI Automation with CrewAI: The Standalone Multi-Agent Framework**

CrewAI is a **powerful, flexible, and fast Python framework** for orchestrating autonomous AI agents, empowering you to build sophisticated AI applications with ease.

**Key Features:**

*   **Standalone & Lean:** Built from scratch, independent of LangChain or other frameworks, for faster execution and minimal overhead.
*   **Autonomous Crews:** Design AI teams that collaborate naturally, making decisions together and achieving complex goals.
*   **Precise Flows:** Create event-driven workflows with granular control over execution paths, perfect for real-world scenarios.
*   **Deep Customization:** Tailor everything, from high-level workflows to low-level agent behaviors, for ultimate control.
*   **Seamless Integration:** Combine Crews (autonomy) and Flows (precision) to build sophisticated and robust AI applications.
*   **Thriving Community:** Supported by over 100,000 certified developers and comprehensive documentation.

**[Explore the CrewAI Repository on GitHub](https://github.com/crewAIInc/crewAI)**

## Table of Contents

*   [Why CrewAI?](#why-crewai)
*   [Getting Started](#getting-started)
*   [Key Features](#key-features)
*   [Understanding Flows and Crews](#understanding-flows-and-crews)
*   [CrewAI vs LangGraph](#how-crewai-compares)
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

<div align="center" style="margin-bottom: 30px;">
  <img src="docs/images/asset.png" alt="CrewAI Logo" width="100%">
</div>

CrewAI empowers developers and enterprises to build intelligent automations with:

*   **Standalone Framework:** Built from scratch, independent of LangChain.
*   **High Performance:** Optimized for speed and minimal resource usage.
*   **Flexible Customization:** Complete freedom to customize every aspect.
*   **Ideal for Every Use Case:** Proven effective for simple and complex scenarios.
*   **Robust Community:** Backed by a rapidly growing community of over **100,000 certified** developers.

## Getting Started

Jumpstart your CrewAI journey with our straightforward setup tutorial.

[![CrewAI Getting Started Tutorial](https://img.youtube.com/vi/-kSOTtYzgEw/hqdefault.jpg)](https://www.youtube.com/watch?v=-kSOTtYzgEw "CrewAI Getting Started Tutorial")

### Learning Resources

Expand your CrewAI expertise with our comprehensive courses:

*   [Multi AI Agent Systems with CrewAI](https://www.deeplearning.ai/short-courses/multi-ai-agent-systems-with-crewai/)
*   [Practical Multi AI Agents and Advanced Use Cases with CrewAI](https://www.deeplearning.ai/short-courses/practical-multi-ai-agents-and-advanced-use-cases-with-crewai/)

### Understanding Flows and Crews

CrewAI offers two primary approaches for building advanced AI applications:

1.  **Crews:** Autonomous teams of AI agents that collaborate on complex tasks through role-based delegation.
2.  **Flows:** Production-ready, event-driven workflows providing granular control over automation.

The true power of CrewAI comes from combining Crews and Flows to build sophisticated, production-ready applications.

### Getting Started with Installation

1.  **Installation:**

    Ensure you have Python >=3.10 <3.14 installed.

    ```shell
    pip install crewai
    ```

    For additional tools:

    ```shell
    pip install 'crewai[tools]'
    ```

### Troubleshooting Dependencies

If you encounter issues:

1.  **ModuleNotFoundError: No module named 'tiktoken'**
    *   `pip install 'crewai[embeddings]'` or `pip install 'crewai[tools]'`
2.  **Failed building wheel for tiktoken**
    *   Ensure Rust compiler is installed.
    *   Upgrade pip: `pip install --upgrade pip`
    *   Use a pre-built wheel: `pip install tiktoken --prefer-binary`

### 2. Setting Up Your Crew with the YAML Configuration

Create your CrewAI project with the CLI command:

```shell
crewai create crew <project_name>
```

This creates a project structure:

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

Customize your project by editing:

*   `src/my_project/config/agents.yaml` - Define your agents.
*   `src/my_project/config/tasks.yaml` - Define your tasks.
*   `src/my_project/crew.py` - Add logic, tools, and arguments.
*   `src/my_project/main.py` - Add custom inputs.
*   `.env` - Store your environment variables.

#### Example: Simple Crew

Instantiate the crew:

```shell
crewai create crew latest-ai-development
```

Modify the files:

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

Set your API keys in `.env`:

```
OPENAI_API_KEY=sk-...
SERPER_API_KEY=YOUR_KEY_HERE
```

Run the following commands:

```shell
cd my_project
crewai install (Optional)
crewai run
```

or

```bash
python src/my_project/main.py
```

If you encounter issues with poetry, update the crewai package:

```bash
crewai update
```

The `report.md` file will be created in your project root.

You can also use the hierarchical process for coordinated task planning and execution. [See more about the processes here](https://docs.crewai.com/core-concepts/Processes/).

## Key Features

CrewAI offers a powerful, independent framework to build, customize, and scale multi-agent automation solutions.

*   **Standalone & Lean:** No dependencies on LangChain, for faster and efficient execution.
*   **Flexible & Precise:** Orchestrate autonomous agents using [Crews](https://docs.crewai.com/concepts/crews) or [Flows](https://docs.crewai.com/concepts/flows).
*   **Seamless Integration:** Combine Crews (autonomy) and Flows (precision).
*   **Deep Customization:** Tailor all aspects of agent behavior and workflow.
*   **Reliable Performance:** Consistent results across simple and complex tasks.
*   **Thriving Community:** Backed by robust documentation and over 100,000 certified developers.

## Examples

Explore real-world CrewAI examples in the [CrewAI-examples repo](https://github.com/crewAIInc/crewAI-examples?tab=readme-ov-file):

*   [Landing Page Generator](https://github.com/crewAIInc/crewAI-examples/tree/main/landing_page_generator)
*   [Having Human input on the execution](https://docs.crewai.com/how-to/Human-Input-on-Execution)
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

Orchestrate multiple Crews within a Flow:

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

*   Use Python for data operations.
*   Create and execute Crews as steps in the workflow.
*   Manage operation sequences with Flow decorators.
*   Implement conditional branching based on Crew results.

## Connecting Your Crew to a Model

CrewAI supports using various LLMs through a variety of connection options. Refer to the [Connect CrewAI to LLMs](https://docs.crewai.com/how-to/LLM-Connections/) page for details on configuring your agents' connections to models.

## How CrewAI Compares

CrewAI provides a unique combination of autonomous agent intelligence and precise workflow control through its Crews and Flows architecture.

*   **CrewAI's Advantage**: Combines autonomous agent intelligence with precise workflow control through its unique Crews and Flows architecture, making it powerful and flexible.

-   **LangGraph**: Requires significant boilerplate code and complex state management.
*   **Autogen**: Lacks an inherent concept of process, making agent orchestration complex.
*   **ChatDev**: Limited customizations and not geared towards production environments.

*P.S. CrewAI executes 5.76x faster in certain cases ([see comparison](https://github.com/crewAIInc/crewAI-examples/tree/main/Notebooks/CrewAI%20Flows%20%26%20Langgraph/QA%20Agent)) and achieves higher evaluation scores ([detailed analysis](https://github.com/crewAIInc/crewAI-examples/blob/main/Notebooks/CrewAI%20Flows%20%26%20Langgraph/Coding%20Assistant/coding_assistant_eval.ipynb)) in certain coding tasks.*

## Contribution

We welcome contributions! To contribute:

*   Fork the repository.
*   Create a new branch.
*   Add your features.
*   Send a pull request.

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

CrewAI uses anonymous telemetry to improve the library.

*   **NO data is collected** concerning prompts, task descriptions, agents' backstories or goals, usage of tools, API calls, responses, or secrets and environment variables.
*   Users can disable telemetry by setting the environment variable OTEL_SDK_DISABLED to true.

Data collected includes:

*   Version of CrewAI
*   Version of Python
*   General OS (e.g. macOS/Windows/Linux)
*   Number of agents and tasks in a crew
*   Crew Process being used
*   If Agents are using memory or allowing delegation
*   If Tasks are being executed in parallel or sequentially
*   Language model being used
*   Roles of agents in a crew
*   Tools names available

Users can opt-in to Further Telemetry, sharing the complete telemetry data by setting the `share_crew` attribute to `True` on their Crews.

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

### Q: What exactly is CrewAI?

A: CrewAI is a standalone, lean, and fast Python framework for orchestrating autonomous AI agents, without relying on external dependencies.

### Q: How do I install CrewAI?

A: Install using pip:

```shell
pip install crewai
```

For additional tools:

```shell
pip install 'crewai[tools]'
```

### Q: Does CrewAI depend on LangChain?

A: No. CrewAI is independent of LangChain.

### Q: Can CrewAI handle complex use cases?

A: Yes, CrewAI excels at simple and complex real-world scenarios.

### Q: Can I use CrewAI with local AI models?

A: Yes, CrewAI supports local AI models via tools like Ollama.

### Q: What makes Crews different from Flows?

A: Crews offer autonomous agent collaboration, while Flows provide precise control.

### Q: How is CrewAI better than LangChain?

A: CrewAI provides simpler APIs, faster execution, and robust documentation.

### Q: Is CrewAI open-source?

A: Yes, CrewAI is open-source.

### Q: Does CrewAI collect data from users?

A: CrewAI collects anonymous telemetry data for improvement.

### Q: Where can I find real-world CrewAI examples?

A:  Explore examples in the [CrewAI-examples repository](https://github.com/crewAIInc/crewAI-examples).

### Q: How can I contribute to CrewAI?

A: Fork the repository, create a branch, add your features, and submit a pull request.

### Q: What additional features does CrewAI Enterprise offer?

A: Enterprise features include a unified control plane, real-time observability, and robust security.

### Q: Is CrewAI Enterprise available for cloud and on-premise deployments?

A: Yes, both cloud and on-premise options are available.

### Q: Can I try CrewAI Enterprise for free?

A: Yes, try the [Crew Control Plane](https://app.crewai.com) for free.

### Q: Does CrewAI support fine-tuning or training custom models?

A: Yes, CrewAI can integrate with custom-trained or fine-tuned models.

### Q: Can CrewAI agents interact with external tools and APIs?

A: Yes, CrewAI agents can easily integrate with external tools, APIs, and databases.

### Q: Is CrewAI suitable for production environments?

A: Yes, CrewAI is designed for production-grade environments.

### Q: How scalable is CrewAI?

A: CrewAI is highly scalable, supporting simple and complex workflows.

### Q: Does CrewAI offer debugging and monitoring tools?

A: Yes, CrewAI Enterprise offers debugging and tracing features.

### Q: What programming languages does CrewAI support?

A: CrewAI is primarily Python-based but can integrate with APIs in any language.

### Q: Does CrewAI offer educational resources for beginners?

A: Yes, through [learn.crewai.com](https://learn.crewai.com).

### Q: Can CrewAI automate human-in-the-loop workflows?

A: Yes.