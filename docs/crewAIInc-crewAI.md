<p align="center">
  <a href="https://github.com/crewAIInc/crewAI">
    <img src="docs/images/crewai_logo.png" width="600px" alt="Open source Multi-AI Agent orchestration framework">
  </a>
</p>

# CrewAI: Unleash the Power of Autonomous AI Agents (Your All-in-One Solution)

**Revolutionize your automation with CrewAI, a lean and lightning-fast framework that empowers you to build autonomous AI agents, free from the constraints of other frameworks.** This open-source project, found on [GitHub](https://github.com/crewAIInc/crewAI), provides the perfect blend of simplicity and control.

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

## Key Features:

*   🚀 **Standalone & Lean:** Built from scratch, independent of LangChain, ensuring speed and efficiency.
*   🧠 **Autonomous Crews:** Empower agents to collaborate and make decisions for complex tasks.
*   ⚙️ **Precise Flows:** Implement event-driven workflows for fine-grained control and state management.
*   🛠️ **Deep Customization:** Tailor every aspect of your AI agents, from workflows to agent behaviors.
*   ⚡️ **High Performance:** Experience fast execution times and minimal resource usage.
*   🤝 **Thriving Community:** Benefit from extensive documentation and a rapidly growing community of over 100,000 certified developers at [learn.crewai.com](https://learn.crewai.com).

## Table of Contents

*   [Why CrewAI?](#why-crewai)
*   [Getting Started](#getting-started)
    *   [Installation](#getting-started-installation)
    *   [Project Setup with the YAML Configuration](#getting-started-project-setup)
    *   [Running Your Crew](#getting-started-running-your-crew)
*   [Key Features](#key-features)
*   [Understanding Flows and Crews](#understanding-flows-and-crews)
*   [CrewAI vs. Other Frameworks](#crewai-vs-other-frameworks)
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

CrewAI offers the best combination of speed, flexibility, and control for multi-agent automation:

*   **Standalone Framework:** No dependencies on LangChain or other agent frameworks.
*   **High Performance:** Optimized for speed and minimal resource usage.
*   **Flexible Customization:** Complete control over workflows, agent behavior, and execution logic.
*   **Ideal for Every Use Case:** Suitable for simple tasks and complex, enterprise-grade scenarios.
*   **Robust Community:** Backed by a growing community, documentation, and resources.

## Getting Started

Get up and running with your first CrewAI agents!

[![CrewAI Getting Started Tutorial](https://img.youtube.com/vi/-kSOTtYzgEw/hqdefault.jpg)](https://www.youtube.com/watch?v=-kSOTtYzgEw "CrewAI Getting Started Tutorial")

### Learning Resources
Explore these detailed courses and tutorials to master CrewAI:

*   [Multi AI Agent Systems with CrewAI](https://www.deeplearning.ai/short-courses/multi-ai-agent-systems-with-crewai/) - Master the fundamentals of multi-agent systems
*   [Practical Multi AI Agents and Advanced Use Cases](https://www.deeplearning.ai/short-courses/practical-multi-ai-agents-and-advanced-use-cases-with-crewai/) - Deep dive into advanced implementations

### Getting Started with Installation

To get started with CrewAI, follow these simple steps:

### <a id="getting-started-installation"></a> 1. Installation

Ensure you have Python >=3.10 <3.14 installed on your system. CrewAI uses [UV](https://docs.astral.sh/uv/) for dependency management and package handling, offering a seamless setup and execution experience.

First, install CrewAI:

```shell
pip install crewai
```

If you want to install the 'crewai' package along with its optional features that include additional tools for agents, you can do so by using the following command:

```shell
pip install 'crewai[tools]'
```

The command above installs the basic package and also adds extra components which require more dependencies to function.

### Troubleshooting Dependencies

If you encounter issues during installation or usage, here are some common solutions:

#### Common Issues

1.  **ModuleNotFoundError: No module named 'tiktoken'**

    *   Install tiktoken explicitly: `pip install 'crewai[embeddings]'`
    *   If using embedchain or other tools: `pip install 'crewai[tools]'`
2.  **Failed building wheel for tiktoken**

    *   Ensure Rust compiler is installed (see installation steps above)
    *   For Windows: Verify Visual C++ Build Tools are installed
    *   Try upgrading pip: `pip install --upgrade pip`
    *   If issues persist, use a pre-built wheel: `pip install tiktoken --prefer-binary`

### <a id="getting-started-project-setup"></a> 2. Setting Up Your Crew with the YAML Configuration

To create a new CrewAI project, run the following CLI (Command Line Interface) command:

```shell
crewai create crew <project_name>
```

This command creates a new project folder with the following structure:

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

You can now start developing your crew by editing the files in the `src/my_project` folder. The `main.py` file is the entry point of the project, the `crew.py` file is where you define your crew, the `agents.yaml` file is where you define your agents, and the `tasks.yaml` file is where you define your tasks.

#### To customize your project, you can:

*   Modify `src/my_project/config/agents.yaml` to define your agents.
*   Modify `src/my_project/config/tasks.yaml` to define your tasks.
*   Modify `src/my_project/crew.py` to add your own logic, tools, and specific arguments.
*   Modify `src/my_project/main.py` to add custom inputs for your agents and tasks.
*   Add your environment variables into the `.env` file.

#### Example of a simple crew with a sequential process:

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

### <a id="getting-started-running-your-crew"></a> 3. Running Your Crew

Before running your crew, make sure you have the following keys set as environment variables in your `.env` file:

*   An [OpenAI API key](https://platform.openai.com/account/api-keys) (or other LLM API key): `OPENAI_API_KEY=sk-...`
*   A [Serper.dev](https://serper.dev/) API key: `SERPER_API_KEY=YOUR_KEY_HERE`

Lock the dependencies and install them by using the CLI command but first, navigate to your project directory:

```shell
cd my_project
crewai install (Optional)
```

To run your crew, execute the following command in the root of your project:

```bash
crewai run
```

or

```bash
python src/my_project/main.py
```

If an error happens due to the usage of poetry, please run the following command to update your crewai package:

```bash
crewai update
```

You should see the output in the console and the `report.md` file should be created in the root of your project with the full final report.

In addition to the sequential process, you can use the hierarchical process, which automatically assigns a manager to the defined crew to properly coordinate the planning and execution of tasks through delegation and validation of results. [See more about the processes here](https://docs.crewai.com/core-concepts/Processes/).

## Understanding Flows and Crews

CrewAI offers two powerful, complementary approaches for building sophisticated AI applications:

1.  **Crews:** Teams of AI agents with true autonomy and agency, working collaboratively through role-based collaboration.
    *   Natural, autonomous decision-making between agents
    *   Dynamic task delegation and collaboration
    *   Specialized roles with defined goals and expertise
    *   Flexible problem-solving approaches
2.  **Flows:** Production-ready, event-driven workflows that deliver precise control over complex automations.
    *   Fine-grained control over execution paths
    *   Secure, consistent state management
    *   Clean integration with Python code
    *   Conditional branching for complex business logic

The true power of CrewAI emerges when combining Crews and Flows. This synergy allows you to build complex, production-grade applications, balance autonomy with precise control, and handle sophisticated real-world scenarios with clean, maintainable code.

## <a id="crewai-vs-other-frameworks"></a> CrewAI vs. Other Frameworks

**CrewAI's Advantage:** CrewAI's architecture uniquely blends autonomous agent intelligence with precise workflow control through its intuitive Crews and Flows. This combination allows developers to effectively orchestrate complex tasks while retaining granular customization.

*   **LangGraph:** While LangGraph provides a solid foundation for agent workflows, it requires more code and can lead to complicated state management.
    *P.S. CrewAI executes 5.76x faster in certain cases ([see comparison](https://github.com/crewAIInc/crewAI-examples/tree/main/Notebooks/CrewAI%20Flows%20%26%20Langgraph/QA%20Agent)) while achieving higher evaluation scores with faster completion times in certain coding tasks, like in this example ([detailed analysis](https://github.com/crewAIInc/crewAI-examples/blob/main/Notebooks/CrewAI%20Flows%20%26%20Langgraph/Coding%20Assistant/coding_assistant_eval.ipynb)).*
*   **Autogen:** Autogen excels at conversation but orchestrating agents' interactions can be cumbersome as tasks grow.
*   **ChatDev:** ChatDev's process-oriented approach is limited in terms of customization and geared towards production environments, which can hinder scalability and flexibility.

## Examples

Explore practical CrewAI examples in the [CrewAI-examples repo](https://github.com/crewAIInc/crewAI-examples?tab=readme-ov-file):

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

CrewAI's power truly shines when combining Crews with Flows to create sophisticated automation pipelines.
CrewAI flows support logical operators like `or_` and `and_` to combine multiple conditions. This can be used with `@start`, `@listen`, or `@router` decorators to create complex triggering conditions.

*   `or_`: Triggers when any of the specified conditions are met.
*   `and_`Triggers when all of the specified conditions are met.

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

## Connecting Your Crew to a Model

CrewAI supports diverse LLMs, offering multiple connection options. By default, your agents use the OpenAI API for model queries. Additionally, you can configure agents to use local models via the Ollama tool.

Refer to the [Connect CrewAI to LLMs](https://docs.crewai.com/how-to/LLM-Connections/) for detailed model connection configurations.

## How CrewAI Compares

CrewAI distinguishes itself by offering a unique combination of autonomous agent intelligence and fine-tuned workflow control through its Crews and Flows architecture. This framework is designed for both high-level orchestration and granular customization, ideal for creating intricate, production-ready systems with detailed control.

## Contribution

We welcome contributions! Follow these steps:

*   Fork the repository.
*   Create a branch for your feature.
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

CrewAI uses anonymous telemetry to collect usage data to improve the library by focusing on the most used features, integrations, and tools.

*   **NO data is collected** regarding prompts, task descriptions, agents' backstories or goals, tool usage, API calls, responses, any processed data, secrets, and environment variables, with the exception of the conditions mentioned.

*   When the `share_crew` feature is enabled, detailed data including task descriptions, agents' backstories or goals, and other specific attributes are collected to provide deeper insights while respecting user privacy.

*   Users can disable telemetry by setting the environment variable `OTEL_SDK_DISABLED` to true.

Data collected includes:

*   Version of CrewAI
*   Version of Python
*   General OS (e.g., macOS/Windows/Linux)
*   Number of agents and tasks in a crew
*   Crew Process being used
*   If Agents are using memory or allowing delegation
*   If Tasks are being executed in parallel or sequentially
*   Language model being used
*   Roles of agents in a crew
*   Tools names available

Users can opt-in to Further Telemetry by setting the `share_crew` attribute to `True` on their Crews. Enabling `share_crew` results in the collection of detailed crew and task execution data, including `goal`, `backstory`, `context`, and `output` of tasks. This enables a deeper insight into usage patterns while respecting the user's choice to share.

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

### <a id="q-what-exactly-is-crewai"></a> Q: What exactly is CrewAI?

A: CrewAI is a standalone, lean, and fast Python framework designed for orchestrating autonomous AI agents.

### <a id="q-how-do-i-install-crewai"></a> Q: How do I install CrewAI?

A: Install CrewAI using pip:

```shell
pip install crewai
```

For additional tools, use:

```shell
pip install 'crewai[tools]'
```

### <a id="q-does-crewai-depend-on-langchain"></a> Q: Does CrewAI depend on LangChain?

A: No. CrewAI is built from the ground up, with no dependencies on LangChain or other agent frameworks.

### <a id="q-can-crewai-handle-complex-use-cases"></a> Q: Can CrewAI handle complex use cases?

A: Yes. CrewAI excels at both simple and complex scenarios.

### <a id="q-can-i-use-crewai-with-local-ai-models"></a> Q: Can I use CrewAI with local AI models?

A: Yes, including models available through Ollama and LM Studio.

### <a id="q-what-makes-crews-different-from-flows"></a> Q: What makes Crews different from Flows?

A: Crews enable agent collaboration, while Flows offer precise, event-driven control.

### <a id="q-how-is-crewai-better-than-langchain"></a> Q: How is CrewAI better than LangChain?

A: CrewAI provides simpler APIs, faster execution, more reliable results, robust documentation, and an active community.

### <a id="q-is-crewai-open-source"></a> Q: Is CrewAI open-source?

A: Yes, CrewAI is open-source.

### <a id="q-does-crewai-collect-data-from-users"></a> Q: Does CrewAI collect data from users?

A: CrewAI collects anonymous telemetry data for improvement, with no collection of sensitive data unless enabled by the user.

### <a id="q-where-can-i-find-real-world-crewai-examples"></a> Q: Where can I find real-world CrewAI examples?

A: See the [CrewAI-examples repository](https://github.com/crewAIInc/crewAI-examples).

### <a id="q-how-can-i-contribute-to-crewai"></a> Q: How can I contribute to CrewAI?

A: Contribute by forking the repository, creating a branch, making changes, and submitting a pull request.

### <a id="q-what-additional-features-does-crewai-enterprise-offer"></a> Q: What additional features does CrewAI Enterprise offer?

A: CrewAI Enterprise offers a unified control plane, real-time observability, secure integrations, and dedicated support.

### <a id="q-is-crewai-enterprise-available-for-cloud-and-on-premise-deployments"></a> Q: Is CrewAI Enterprise available for cloud and on-premise deployments?

A: Yes, CrewAI Enterprise supports both cloud and on-premise options.

### <a id="q-can-i-try-crewai-enterprise-for-free"></a> Q: Can I try CrewAI Enterprise for free?

A: Yes, try the [Crew Control Plane](https://app.crewai.com) for free.

### <a id="q-does-crewai-support-fine-tuning-or-training-custom-models"></a> Q: Does CrewAI support fine-tuning or training custom models?

A: Yes, CrewAI integrates with custom-trained or fine-tuned models.

### Q: Can CrewAI agents interact with external tools and APIs?

A: Yes, agents integrate with external tools, APIs, and databases.

### Q: Is CrewAI suitable for production environments?

A: Yes, it's designed for reliability and scalability.

### Q: How scalable is CrewAI?

A: CrewAI is highly scalable.

### Q: Does CrewAI offer debugging and monitoring tools?

A: Yes, CrewAI Enterprise includes advanced debugging, tracing, and real-time observability features, simplifying the management and troubleshooting of your automations.

### Q: What programming languages does CrewAI support?

A: CrewAI is primarily Python-based and easily integrates with services and APIs written in any programming language through its flexible API integration capabilities.

### Q: Does CrewAI offer educational resources for beginners?

A: Yes, CrewAI provides extensive beginner-friendly tutorials, courses, and documentation through learn.crewai.com, supporting developers at all skill levels.

### Q: Can CrewAI automate human-in-the-loop workflows?

A: Yes, CrewAI fully supports human-in-the-loop workflows, allowing seamless collaboration between human experts and AI agents for enhanced decision-making.