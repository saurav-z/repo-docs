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

## Supercharge Your AI Automation with CrewAI: Unleash the Power of Autonomous Agents.

CrewAI is a lightning-fast, flexible, and open-source Python framework, built from scratch, for orchestrating autonomous AI agents, offering both simplicity and precise control for any scenario.

*   [**Visit the original repository on GitHub**](https://github.com/crewAIInc/crewAI)

**Key Features:**

*   🚀 **Standalone & Lean:** Independent of LangChain and other agent frameworks, ensuring fast execution.
*   🛠️ **Flexible & Precise:** Utilize intuitive Crews (autonomous collaboration) and Flows (event-driven control) for perfect balance.
*   🔗 **Seamless Integration:** Effortlessly combine Crews and Flows to build complex, real-world automations.
*   ⚙️ **Deep Customization:** Tailor every aspect, from workflows to agent behavior and internal prompts.
*   ✅ **Reliable Performance:** Achieve consistent results across simple and enterprise-grade automations.
*   🫂 **Thriving Community:** Benefit from robust documentation and a community of over 100,000 certified developers.

## Table of Contents

*   [Why CrewAI?](#why-crewai)
*   [Getting Started](#getting-started)
    *   [Installation](#1-installation)
    *   [CLI Project Creation](#2-setting-up-your-crew-with-the-yaml-configuration)
    *   [Running Your Crew](#3-running-your-crew)
*   [Key Features](#key-features)
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

CrewAI unlocks the true potential of multi-agent automation, delivering the best-in-class combination of speed, flexibility, and control with either Crews of AI Agents or Flows of Events:

*   **Standalone Framework:** Built from scratch, independent of LangChain or any other agent framework.
*   **High Performance:** Optimized for speed and minimal resource usage, enabling faster execution.
*   **Flexible Low Level Customization:** Complete freedom to customize at both high and low levels - from overall workflows and system architecture to granular agent behaviors, internal prompts, and execution logic.
*   **Ideal for Every Use Case:** Proven effective for both simple tasks and highly complex, real-world, enterprise-grade scenarios.
*   **Robust Community:** Backed by a rapidly growing community of over **100,000 certified** developers offering comprehensive support and resources.

CrewAI empowers developers and enterprises to confidently build intelligent automations, bridging the gap between simplicity, flexibility, and performance.

## Getting Started

### 1. Installation

Ensure you have Python >=3.10 <3.14 installed. CrewAI uses [UV](https://docs.astral.sh/uv/) for dependency management.

Install CrewAI:

```shell
pip install crewai
```

To include optional features (tools):

```shell
pip install 'crewai[tools]'
```

#### Troubleshooting Dependencies

1.  **ModuleNotFoundError: No module named 'tiktoken'**: `pip install 'crewai[embeddings]'` or `pip install 'crewai[tools]'`
2.  **Failed building wheel for tiktoken**: Install Rust compiler.  For Windows, ensure Visual C++ Build Tools are installed. Upgrade pip: `pip install --upgrade pip`. Try: `pip install tiktoken --prefer-binary`

### 2. Setting Up Your Crew with the YAML Configuration

Create a new CrewAI project using the CLI:

```shell
crewai create crew <project_name>
```

This creates a project with this structure:

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

Customize your project by modifying:

*   `src/my_project/config/agents.yaml`: Define your agents.
*   `src/my_project/config/tasks.yaml`: Define your tasks.
*   `src/my_project/crew.py`: Add logic, tools, and arguments.
*   `src/my_project/main.py`: Add custom inputs.
*   `.env`: Store environment variables.

#### Example: Simple Crew

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

*   `OPENAI_API_KEY`: Your OpenAI API key (or other LLM key).
*   `SERPER_API_KEY`: Your Serper.dev API key.

Run the following in your project directory:

```bash
cd my_project
crewai install (Optional)
```

Then, run your crew:

```bash
crewai run
```

or

```bash
python src/my_project/main.py
```

If you encounter errors, try `crewai update`.

## Key Features

CrewAI provides a lean, standalone, high-performance framework.

-   **Standalone & Lean**: Independent of frameworks like LangChain.
-   **Flexible & Precise**: Orchestrate with Crews (autonomy) or Flows (precision).
-   **Seamless Integration**: Combine Crews and Flows effortlessly.
-   **Deep Customization**: Tailor workflows, agent behaviors, and prompts.
-   **Reliable Performance**: Consistent results for all tasks.
-   **Thriving Community**: Strong documentation and over 100,000 certified developers.

## Examples

*   Explore real-life examples in the [CrewAI-examples repo](https://github.com/crewAIInc/crewAI-examples?tab=readme-ov-file).

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

CrewAI's power shines when combining Crews and Flows.  Flows support `or_` and `and_`.

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

Refer to the [LLM Connections documentation](https://docs.crewai.com/how-to/LLM-Connections/) for detailed configuration.

## How CrewAI Compares

**CrewAI excels** by combining autonomous agent intelligence with precise workflow control through Crews and Flows.

*   **LangGraph**: Requires significant boilerplate and complex state management.
*   **Autogen**: Lacks inherent process, making agent orchestration complex.
*   **ChatDev**:  Limited customization and not production-ready.

*P.S. CrewAI demonstrates significant performance advantages over LangGraph, executing 5.76x faster in certain cases like this QA task example ([see comparison](https://github.com/crewAIInc/crewAI-examples/tree/main/Notebooks/CrewAI%20Flows%20%26%20Langgraph/QA%20Agent)) while achieving higher evaluation scores with faster completion times in certain coding tasks, like in this example ([detailed analysis](https://github.com/crewAIInc/crewAI-examples/blob/main/Notebooks/CrewAI%20Flows%20%26%20Langgraph/Coding%20Assistant/coding_assistant_eval.ipynb)).*

## Contribution

We welcome contributions!

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

CrewAI uses anonymous telemetry to improve the library. **NO data is collected** concerning prompts, task descriptions, or secrets unless `share_crew` is enabled.

Data collected includes:

*   CrewAI version
*   Python version
*   OS details
*   Number of agents/tasks
*   Crew Process
*   Agent memory/delegation usage
*   Task execution
*   Language model
*   Agent roles
*   Tool names

Enable `share_crew` for deeper insights: `share_crew = True`.

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

A: CrewAI is a standalone Python framework for orchestrating autonomous AI agents, built from scratch.

### Q: How do I install CrewAI?

A: Use `pip install crewai` and `pip install 'crewai[tools]'`.

### Q: Does CrewAI depend on LangChain?

A: No.

### Q: Can CrewAI handle complex use cases?

A: Yes.

### Q: Can I use CrewAI with local AI models?

A: Yes.

### Q: What makes Crews different from Flows?

A: Crews are for autonomous agent collaboration; Flows are for precise control.

### Q: How is CrewAI better than LangChain?

A: CrewAI offers simpler APIs, faster execution, and a more reliable experience.

### Q: Is CrewAI open-source?

A: Yes.

### Q: Does CrewAI collect data from users?

A: CrewAI collects anonymous telemetry data, with no sensitive data collected by default.

### Q: Where can I find real-world CrewAI examples?

A: In the [CrewAI-examples repository](https://github.com/crewAIInc/crewAI-examples?tab=readme-ov-file).

### Q: How can I contribute to CrewAI?

A: Fork, create a branch, and submit a pull request.

### Q: What additional features does CrewAI Enterprise offer?

A: A unified control plane, real-time observability, secure integrations, and enterprise support.

### Q: Is CrewAI Enterprise available for cloud and on-premise deployments?

A: Yes.

### Q: Can I try CrewAI Enterprise for free?

A: Yes, via the [Crew Control Plane](https://app.crewai.com).

### Q: Does CrewAI support fine-tuning or training custom models?

A: Yes.

### Q: Can CrewAI agents interact with external tools and APIs?

A: Yes.

### Q: Is CrewAI suitable for production environments?

A: Yes.

### Q: How scalable is CrewAI?

A: Highly scalable.

### Q: Does CrewAI offer debugging and monitoring tools?

A: Yes, CrewAI Enterprise.

### Q: What programming languages does CrewAI support?

A: Primarily Python, but integrates with other languages through APIs.

### Q: Does CrewAI offer educational resources for beginners?

A: Yes, at learn.crewai.com.

### Q: Can CrewAI automate human-in-the-loop workflows?

A: Yes.