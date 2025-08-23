<p align="center">
  <a href="https://github.com/crewAIInc/crewAI">
    <img src="docs/images/crewai_logo.png" width="600px" alt="CrewAI: Open Source Multi-Agent AI Orchestration">
  </a>
</p>

<p align="center">
  **Unlock the Power of Autonomous AI with CrewAI: Build Intelligent Automations That Scale.**
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

## Table of Contents

- [About CrewAI](#about-crewai)
- [Key Features](#key-features)
- [Why CrewAI?](#why-crewai)
- [Getting Started](#getting-started)
    - [Installation](#installation)
    - [Setting Up with YAML](#setting-up-with-the-yaml-configuration)
    - [Running Your Crew](#running-your-crew)
- [Understanding Crews and Flows](#understanding-flows-and-crews)
- [Examples](#examples)
    - [Quick Tutorial](#quick-tutorial)
    - [Write Job Descriptions](#write-job-descriptions)
    - [Trip Planner](#trip-planner)
    - [Stock Analysis](#stock-analysis)
    - [Using Crews and Flows Together](#using-crews-and-flows-together)
- [Connecting to LLMs](#connecting-your-crew-to-a-model)
- [How CrewAI Compares](#how-crewai-compares)
- [Contribution](#contribution)
- [Telemetry](#telemetry)
- [License](#license)
- [Frequently Asked Questions (FAQ)](#frequently-asked-questions-faq)

## About CrewAI

CrewAI is a **powerful, open-source Python framework designed to orchestrate multi-agent AI systems** with ease, empowering developers to build complex and autonomous AI solutions.  Built from the ground up, CrewAI offers unparalleled flexibility, speed, and control, making it the ideal choice for both simple and enterprise-grade applications.

## Key Features

*   **Standalone & Lean**:  Built from scratch, independent of LangChain or other agent frameworks, for faster execution and lighter resource usage.
*   **Flexible & Precise**: Easily orchestrate autonomous agents through intuitive [Crews](https://docs.crewai.com/concepts/crews) or precise [Flows](https://docs.crewai.com/concepts/flows), achieving the perfect balance for your needs.
*   **Seamless Integration**: Effortlessly combine Crews (autonomy) and Flows (precision) to create complex, real-world automations.
*   **Deep Customization**: Tailor every aspect—from high-level workflows down to low-level internal prompts and agent behaviors.
*   **Reliable Performance**: Consistent results across simple tasks and complex, enterprise-level automations.
*   **Thriving Community**: Backed by robust documentation and over 100,000 certified developers, providing exceptional support and guidance.

## Why CrewAI?

<div align="center" style="margin-bottom: 30px;">
  <img src="docs/images/asset.png" alt="CrewAI Logo" width="100%">
</div>

CrewAI empowers developers and enterprises to confidently build intelligent automations, bridging the gap between simplicity, flexibility, and performance.

*   **Standalone Framework**: Built from scratch, independent of LangChain or any other agent framework.
*   **High Performance**: Optimized for speed and minimal resource usage, enabling faster execution.
*   **Flexible Low Level Customization**: Complete freedom to customize at both high and low levels - from overall workflows and system architecture to granular agent behaviors, internal prompts, and execution logic.
*   **Ideal for Every Use Case**: Proven effective for both simple tasks and highly complex, real-world, enterprise-grade scenarios.
*   **Robust Community**: Backed by a rapidly growing community of over **100,000 certified** developers offering comprehensive support and resources.

## Getting Started

### Installation

Ensure you have Python >=3.10 <3.14 installed.  CrewAI uses [UV](https://docs.astral.sh/uv/) for dependency management.

1.  Install CrewAI:

    ```shell
    pip install crewai
    ```

2.  Optional Features:

    ```shell
    pip install 'crewai[tools]'
    ```

### Troubleshooting Dependencies

*   **`ModuleNotFoundError: No module named 'tiktoken'`**:  Install `tiktoken` explicitly: `pip install 'crewai[embeddings]'` or `pip install 'crewai[tools]'` if using tools.
*   **`Failed building wheel for tiktoken`**: Ensure Rust compiler is installed. Upgrade pip (`pip install --upgrade pip`). If issues persist, try: `pip install tiktoken --prefer-binary`

### Setting Up with the YAML Configuration

1.  Create a new CrewAI project:

    ```shell
    crewai create crew <project_name>
    ```

2.  Project Structure:

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

3.  Customize your project:

    *   `src/my_project/config/agents.yaml`: Define your agents.
    *   `src/my_project/config/tasks.yaml`: Define your tasks.
    *   `src/my_project/crew.py`: Add logic, tools, and arguments.
    *   `src/my_project/main.py`: Add custom inputs.
    *   `.env`: Add environment variables.

4.  Example - Simple Crew:

    *   Instantiate crew: `crewai create crew latest-ai-development`
    *   Modify `agents.yaml`:

    ```yaml
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

    *   Modify `tasks.yaml`:

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

    *   Modify `crew.py`:

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

    *   Modify `main.py`:

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

### Running Your Crew

1.  Set environment variables in `.env`:

    *   `OPENAI_API_KEY=sk-...`
    *   `SERPER_API_KEY=YOUR_KEY_HERE`

2.  Navigate to your project directory: `cd my_project`

3.  Lock dependencies & install: `crewai install` (Optional)

4.  Run your crew:

    ```bash
    crewai run
    ```

    or

    ```bash
    python src/my_project/main.py
    ```

    If poetry issues occur:  `crewai update`

## Understanding Crews and Flows

CrewAI offers two powerful, complementary approaches:

1.  **Crews**: Teams of AI agents with true autonomy, working collaboratively.  Enable natural decision-making, dynamic task delegation, and specialized roles.
2.  **Flows**: Production-ready, event-driven workflows that deliver precise control. Provide fine-grained control, secure state management, and clean integration with Python code.

The synergy of Crews and Flows enables building complex, production-grade applications with a balance of autonomy and control.

## Examples

You can test different real life examples of AI crews in the [CrewAI-examples repo](https://github.com/crewAIInc/crewAI-examples?tab=readme-ov-file):

-   [Landing Page Generator](https://github.com/crewAIInc/crewAI-examples/tree/main/landing_page_generator)
-   [Having Human input on the execution](https://docs.crewai.com/how-to/Human-Input-on-Execution)
-   [Trip Planner](https://github.com/crewAIInc/crewAI-examples/tree/main/trip_planner)
-   [Stock Analysis](https://github.com/crewAIInc/crewAI-examples/tree/main/stock_analysis)

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

CrewAI's power shines when combining Crews with Flows to create sophisticated automation pipelines.
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

## Connecting Your Crew to a Model

CrewAI supports various LLMs. By default, agents use the OpenAI API. Refer to the [Connect CrewAI to LLMs](https://docs.crewai.com/how-to/LLM-Connections/) page for model configuration.

## How CrewAI Compares

**CrewAI's Advantage**: CrewAI's architecture uniquely combines autonomous agent intelligence with precise workflow control. It excels at both high-level orchestration and low-level customization, enabling complex, production-grade systems with granular control.

*   **LangGraph**: LangGraph requires significant boilerplate and complex state management. Its tight coupling with LangChain limits flexibility.

*P.S. CrewAI demonstrates significant performance advantages over LangGraph, executing 5.76x faster in certain cases like this QA task example ([see comparison](https://github.com/crewAIInc/crewAI-examples/tree/main/Notebooks/CrewAI%20Flows%20%26%20Langgraph/QA%20Agent)) while achieving higher evaluation scores with faster completion times in certain coding tasks, like in this example ([detailed analysis](https://github.com/crewAIInc/crewAI-examples/blob/main/Notebooks/CrewAI%20Flows%20%26%20Langgraph/Coding%20Assistant/coding_assistant_eval.ipynb)).*

*   **Autogen**: Autogen lacks inherent process. Orchestrating agents requires additional programming, which becomes complex with scale.
*   **ChatDev**: ChatDev's process implementation is rigid. Customizations are limited and not geared toward production, hindering scalability.

## Contribution

CrewAI is open-source and welcomes contributions.

*   Fork the repository.
*   Create a feature branch.
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

CrewAI uses anonymous telemetry to collect usage data to help improve the library.  **NO sensitive data is collected**.

Users can disable telemetry by setting the environment variable `OTEL_SDK_DISABLED` to `true`.

Data collected includes:

*   CrewAI version
*   Python version
*   General OS info (e.g., OS, CPU count)
*   Number of agents and tasks in a crew
*   Crew process used
*   Agent memory/delegation usage
*   Parallel/sequential task execution
*   Language model being used
*   Agent roles
*   Tools used

Users can opt-in to Further Telemetry by setting `share_crew` to `True`.  This shares detailed crew execution data.

## License

CrewAI is released under the [MIT License](https://github.com/crewAIInc/crewAI/blob/main/LICENSE).

## Frequently Asked Questions (FAQ)

### General

*   [What is CrewAI?](#q-what-exactly-is-crewai)
*   [How to install?](#q-how-do-i-install-crewai)
*   [Does CrewAI depend on LangChain?](#q-does-crewai-depend-on-langchain)
*   [Is CrewAI open-source?](#q-is-crewai-open-source)
*   [Does CrewAI collect data?](#q-does-crewai-collect-data-from-users)

### Features and Capabilities

*   [Handle complex use cases?](#q-can-crewai-handle-complex-use-cases)
*   [Use with local AI models?](#q-can-i-use-crewai-with-local-ai-models)
*   [Crews vs. Flows?](#q-what-makes-crews-different-from-flows)
*   [How is CrewAI better than LangChain?](#q-how-is-crewai-better-than-langchain)
*   [Support custom models?](#q-does-crewai-support-fine-tuning-or-training-custom-models)

### Resources and Community

*   [Real-world examples?](#q-where-can-i-find-real-world-crewai-examples)
*   [How to contribute?](#q-how-can-i-contribute-to-crewai)

### Enterprise Features

*   [CrewAI Enterprise features?](#q-what-additional-features-does-crewai-enterprise-offer)
*   [Cloud and on-prem deployments?](#q-is-crewai-enterprise-available-for-cloud-and-on-premise-deployments)
*   [Free trial?](#q-can-i-try-crewai-enterprise-for-free)

### Q: What exactly is CrewAI?

A: CrewAI is a standalone, lean, and fast Python framework built specifically for orchestrating autonomous AI agents. It's built from scratch and independent of frameworks like LangChain, offering a leaner and faster experience.

### Q: How do I install CrewAI?

A: Install with `pip install crewai`. For extra tools: `pip install 'crewai[tools]'`

### Q: Does CrewAI depend on LangChain?

A: No, CrewAI is entirely independent, offering a lean and flexible experience.

### Q: Can CrewAI handle complex use cases?

A: Yes, CrewAI excels in both simple and highly complex scenarios.

### Q: Can I use CrewAI with local AI models?

A: Yes, CrewAI supports local models through tools like Ollama and LM Studio.

### Q: What makes Crews different from Flows?

A: Crews offer autonomous agent collaboration. Flows provide precise, event-driven control. Both can be combined.

### Q: How is CrewAI better than LangChain?

A: CrewAI offers simpler APIs, faster execution, more reliable results, robust documentation, and an active community.

### Q: Is CrewAI open-source?

A: Yes, CrewAI is open-source, welcoming contributions.

### Q: Does CrewAI collect data from users?

A: CrewAI collects anonymous telemetry for improvement. Sensitive data isn't collected unless users enable sharing explicitly.

### Q: Where can I find real-world CrewAI examples?

A: Find examples in the [CrewAI-examples repository](https://github.com/crewAIInc/crewAI-examples).

### Q: How can I contribute to CrewAI?

A: Fork, create a branch, make changes, and submit a pull request.

### Q: What additional features does CrewAI Enterprise offer?

A: Enterprise offers a unified control plane, real-time observability, secure integrations, and more.

### Q: Is CrewAI Enterprise available for cloud and on-premise deployments?

A: Yes.

### Q: Can I try CrewAI Enterprise for free?

A: Yes, you can access the [Crew Control Plane](https://app.crewai.com) for free.

### Q: Does CrewAI support fine-tuning or training custom models?

A: Yes, integration with custom models is supported.

### Q: Can CrewAI agents interact with external tools and APIs?

A: Yes.

### Q: Is CrewAI suitable for production environments?

A: Yes, it is designed for production-grade use.

### Q: How scalable is CrewAI?

A: Highly scalable for both simple and large-scale workflows.

### Q: Does CrewAI offer debugging and monitoring tools?

A: Yes, in the Enterprise version.

### Q: What programming languages does CrewAI support?

A: Primarily Python, but integrates with other languages via APIs.

### Q: Does CrewAI offer educational resources for beginners?

A: Yes, at learn.crewai.com.

### Q: Can CrewAI automate human-in-the-loop workflows?

A: Yes.