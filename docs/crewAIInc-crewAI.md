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

## **Unlock the Power of Autonomous AI with CrewAI: The Standalone Multi-Agent Framework**

CrewAI is a revolutionary, independent Python framework, providing developers with a powerful and flexible platform to build and deploy autonomous AI agents.  ([Explore the original repo](https://github.com/crewAIInc/crewAI)).

**Key Features:**

*   **Standalone & Lean**: Operates independently, free from dependencies like LangChain, ensuring faster execution.
*   **Autonomous Crews**:  Create collaborative teams of AI agents that work together with true autonomy for complex tasks.
*   **Precise Flows**: Implement granular control with event-driven workflows, offering detailed orchestration and management.
*   **Seamless Integration**: Effortlessly combine Crews (autonomy) and Flows (precision) for complex real-world automations.
*   **Deep Customization**: Tailor every aspect of your agents, from high-level workflows to low-level agent behaviors.
*   **High Performance**: Optimized for speed and efficiency, enabling faster execution and minimal resource usage.
*   **Growing Community**: Benefit from comprehensive support and resources from a rapidly growing community of 100,000+ developers.

## Table of Contents

*   [Why CrewAI?](#why-crewai)
*   [Getting Started](#getting-started)
    *   [Installation](#getting-started-installation)
    *   [Setting Up with YAML Configuration](#getting-started-setup-with-yaml)
    *   [Running Your Crew](#getting-started-running-your-crew)
*   [Key Features](#key-features)
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
    *   [Installing Dependencies](#installing-dependencies)
    *   [Virtual Env](#virtual-env)
    *   [Pre-commit hooks](#pre-commit-hooks)
    *   [Running Tests](#running-tests)
    *   [Running static type checks](#running-static-type-checks)
    *   [Packaging](#packaging)
    *   [Installing Locally](#installing-locally)
*   [Telemetry](#telemetry)
*   [License](#license)

## Why CrewAI?

<div align="center" style="margin-bottom: 30px;">
  <img src="docs/images/asset.png" alt="CrewAI Logo" width="100%">
</div>

CrewAI offers the best-in-class combination of speed, flexibility, and control for multi-agent automation. It provides:

*   **Standalone Framework:** Unburdened by dependencies, allowing for greater speed and efficiency.
*   **High Performance:** Optimized for speed and minimal resource usage.
*   **Flexible Customization:** Complete freedom to modify workflows, system architecture, agent behaviors, prompts, and execution logic.
*   **Ideal for Any Use Case:** Effective in both simple and complex, real-world scenarios.
*   **Robust Community:** A rapidly growing community of over **100,000 certified** developers.

## Getting Started

Get your first CrewAI agents up and running by following this tutorial:

[![CrewAI Getting Started Tutorial](https://img.youtube.com/vi/-kSOTtYzgEw/hqdefault.jpg)](https://www.youtube.com/watch?v=-kSOTtYzgEw "CrewAI Getting Started Tutorial")

### Learning Resources

Master CrewAI through these comprehensive courses:

*   [Multi AI Agent Systems with CrewAI](https://www.deeplearning.ai/short-courses/multi-ai-agent-systems-with-crewai/) - Fundamentals of multi-agent systems.
*   [Practical Multi AI Agents and Advanced Use Cases with CrewAI](https://www.deeplearning.ai/short-courses/practical-multi-ai-agents-and-advanced-use-cases-with-crewai/) - Deep dive into advanced implementations.

### <a id="getting-started-installation"></a>Installation

Ensure Python >=3.10 <3.14 is installed. CrewAI uses [UV](https://docs.astral.sh/uv/) for dependency management.

First, install CrewAI:

```shell
pip install crewai
```

For optional features:

```shell
pip install 'crewai[tools]'
```

### Troubleshooting Dependencies

*   **ModuleNotFoundError: No module named 'tiktoken'**: Install tiktoken explicitly: `pip install 'crewai[embeddings]'` or `pip install 'crewai[tools]'`.
*   **Failed building wheel for tiktoken**: Ensure Rust compiler is installed or use a pre-built wheel: `pip install tiktoken --prefer-binary`.

### <a id="getting-started-setup-with-yaml"></a>Setting Up Your Project

1.  Create a new CrewAI project:

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

2.  Customize your project:

    *   Modify `src/my_project/config/agents.yaml` to define your agents.
    *   Modify `src/my_project/config/tasks.yaml` to define your tasks.
    *   Modify `src/my_project/crew.py` to add your logic, tools, and arguments.
    *   Modify `src/my_project/main.py` to add custom inputs for agents and tasks.
    *   Add environment variables to the `.env` file.

    #### Example of a Simple Crew:

    *   Instantiate your crew:

        ```shell
        crewai create crew latest-ai-development
        ```

    *   Modify the files as needed:
        *   **agents.yaml**

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

        *   **tasks.yaml**

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

        *   **crew.py**

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

        *   **main.py**

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

### <a id="getting-started-running-your-crew"></a>Running Your Crew

1.  **Set up environment variables** in your `.env` file:

    *   `OPENAI_API_KEY=sk-...` (or your LLM API key).
    *   `SERPER_API_KEY=YOUR_KEY_HERE`.

2.  **Install dependencies:**

    ```shell
    cd my_project
    crewai install (Optional)
    ```

3.  **Run your crew:**

    ```bash
    crewai run
    ```

    or

    ```bash
    python src/my_project/main.py
    ```

    If you have Poetry-related errors, try `crewai update`.

    The output will be displayed in the console, and the `report.md` file will contain the final report.  You can explore processes like the hierarchical process as well.  More about this can be found [here](https://docs.crewai.com/core-concepts/Processes/).

## <a id="key-features"></a>Key Features

*   **Standalone & Lean**: Built from scratch, independent of other frameworks like LangChain.
*   **Flexible & Precise**: Orchestrate agents using intuitive [Crews](https://docs.crewai.com/concepts/crews) or precise [Flows](https://docs.crewai.com/concepts/flows).
*   **Seamless Integration**: Combine Crews (autonomy) and Flows (precision) to create complex automations.
*   **Deep Customization**: Tailor every aspect of your agents, workflows, and internal processes.
*   **Reliable Performance**: Achieves consistent results across a wide range of tasks.
*   **Thriving Community**: Robust documentation and a community of 100,000+ certified developers for exceptional support.

## <a id="examples"></a>Examples

Find real-world examples in the [CrewAI-examples repo](https://github.com/crewAIInc/crewAI-examples?tab=readme-ov-file):

*   [Landing Page Generator](https://github.com/crewAIInc/crewAI-examples/tree/main/landing_page_generator)
*   [Having Human Input](https://docs.crewai.com/how-to/Human-Input-on-Execution)
*   [Trip Planner](https://github.com/crewAIInc/crewAI-examples/tree/main/trip_planner)
*   [Stock Analysis](https://github.com/crewAIInc/crewAI-examples/tree/main/stock_analysis)

### <a id="quick-tutorial"></a>Quick Tutorial

[![CrewAI Tutorial](https://img.youtube.com/vi/tnejrr-0a94/maxresdefault.jpg)](https://www.youtube.com/watch?v=tnejrr-0a94 "CrewAI Tutorial")

### <a id="write-job-descriptions"></a>Write Job Descriptions

[Check out code for this example](https://github.com/crewAIInc/crewAI-examples/tree/main/job-posting) or watch a video below:

[![Jobs postings](https://img.youtube.com/vi/u98wEMz-9to/maxresdefault.jpg)](https://www.youtube.com/watch?v=u98wEMz-9to "Jobs postings")

### <a id="trip-planner"></a>Trip Planner

[Check out code for this example](https://github.com/crewAIInc/crewAI-examples/tree/main/trip_planner) or watch a video below:

[![Trip Planner](https://img.youtube.com/vi/xis7rWp-hjs/maxresdefault.jpg)](https://www.youtube.com/watch?v=xis7rWp-hjs "Trip Planner")

### <a id="stock-analysis"></a>Stock Analysis

[Check out code for this example](https://github.com/crewAIInc/crewAI-examples/tree/main/stock_analysis) or watch a video below:

[![Stock Analysis](https://img.youtube.com/vi/e0Uj4yWdaAg/maxresdefault.jpg)](https://www.youtube.com/watch?v=e0Uj4yWdaAg "Stock Analysis")

### <a id="using-crews-and-flows-together"></a>Using Crews and Flows Together

CrewAI shines when combining Crews and Flows to create sophisticated automation pipelines.

CrewAI flows support logical operators like `or_` and `and_` to combine multiple conditions. This can be used with `@start`, `@listen`, or `@router` decorators to create complex triggering conditions.

Here's how to orchestrate multiple Crews within a Flow:

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

## <a id="connecting-your-crew-to-a-model"></a>Connecting Your Crew to a Model

Connect your agents to various LLMs using different methods. By default, the OpenAI API is used. However, you can connect to local models via tools like Ollama.  Refer to the [LLM Connections page](https://docs.crewai.com/how-to/LLM-Connections/) for more details.

## <a id="how-crewai-compares"></a>How CrewAI Compares

CrewAI distinguishes itself through its Crews and Flows architecture, combining agent intelligence with precise workflow control. It offers high-level orchestration and low-level customization.

*   **LangGraph:** Requires significant boilerplate code and complex state management. Limited flexibility when customizing agent behaviors.
    *   *P.S. CrewAI executes 5.76x faster in some tasks ([see comparison](https://github.com/crewAIInc/crewAI-examples/tree/main/Notebooks/CrewAI%20Flows%20%26%20Langgraph/QA%20Agent)) while achieving higher evaluation scores with faster completion times ([detailed analysis](https://github.com/crewAIInc/crewAI-examples/blob/main/Notebooks/CrewAI%20Flows%20%26%20Langgraph/Coding%20Assistant/coding_assistant_eval.ipynb)).*
*   **Autogen:** Lacks an inherent concept of process; orchestrating interactions requires additional programming.
*   **ChatDev:** ChatDev has rigid implementations with limited customizations and is not geared towards production environments.

## <a id="contribution"></a>Contribution

CrewAI is open source.

1.  Fork the repository.
2.  Create a new branch.
3.  Add your feature/improvement.
4.  Send a pull request.

### <a id="installing-dependencies"></a>Installing Dependencies

```bash
uv lock
uv sync
```

### <a id="virtual-env"></a>Virtual Env

```bash
uv venv
```

### <a id="pre-commit-hooks"></a>Pre-commit hooks

```bash
pre-commit install
```

### <a id="running-tests"></a>Running Tests

```bash
uv run pytest .
```

### <a id="running-static-type-checks"></a>Running static type checks

```bash
uvx mypy src
```

### <a id="packaging"></a>Packaging

```bash
uv build
```

### <a id="installing-locally"></a>Installing Locally

```bash
pip install dist/*.tar.gz
```

## <a id="telemetry"></a>Telemetry

CrewAI collects anonymous usage data to improve the library. No prompts, tool usage, API calls, or secrets are collected, with the exception of conditions mentioned. You can disable telemetry by setting the environment variable `OTEL_SDK_DISABLED=true`.

Data Collected:

*   CrewAI Version
*   Python Version
*   General OS
*   Number of agents and tasks
*   Crew Process being used
*   Agents using memory/delegation
*   Tasks being executed in parallel/sequentially
*   Language model being used
*   Agent roles
*   Tools names available

Opt-in to Further Telemetry by setting `share_crew=True` to share crew/task execution data.

## <a id="license"></a>License

CrewAI is released under the [MIT License](https://github.com/crewAIInc/crewAI/blob/main/LICENSE).

## <a id="frequently-asked-questions-faq"></a>Frequently Asked Questions (FAQ)

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

### <a id="q-what-exactly-is-crewai"></a>Q: What exactly is CrewAI?

A: CrewAI is a standalone, lean, and fast Python framework built for orchestrating autonomous AI agents. Unlike frameworks like LangChain, CrewAI does not rely on external dependencies, making it leaner, faster, and simpler.

### <a id="q-how-do-i-install-crewai"></a>Q: How do I install CrewAI?

A: Install CrewAI using pip:

```shell
pip install crewai
```

For additional tools, use:

```shell
pip install 'crewai[tools]'
```

### <a id="q-does-crewai-depend-on-langchain"></a>Q: Does CrewAI depend on LangChain?

A: No. CrewAI is built entirely from the ground up, with no dependencies on LangChain or other agent frameworks. This ensures a lean, fast, and flexible experience.

### <a id="q-can-crewai-handle-complex-use-cases"></a>Q: Can CrewAI handle complex use cases?

A: Yes. CrewAI excels at both simple and highly complex real-world scenarios, offering deep customization options at both high and low levels, from internal prompts to sophisticated workflow orchestration.

### <a id="q-can-i-use-crewai-with-local-ai-models"></a>Q: Can I use CrewAI with local AI models?

A: Absolutely! CrewAI supports various language models, including local ones. Tools like Ollama and LM Studio allow seamless integration. Check the [LLM Connections documentation](https://docs.crewai.com/how-to/LLM-Connections/) for more details.

### <a id="q-what-makes-crews-different-from-flows"></a>Q: What makes Crews different from Flows?

A: Crews provide autonomous agent collaboration, ideal for tasks requiring flexible decision-making and dynamic interaction. Flows offer precise, event-driven control, ideal for managing detailed execution paths and secure state management. You can seamlessly combine both for maximum effectiveness.

### <a id="q-how-is-crewai-better-than-langchain"></a>Q: How is CrewAI better than LangChain?

A: CrewAI provides simpler, more intuitive APIs, faster execution speeds, more reliable and consistent results, robust documentation, and an active community—addressing common criticisms and limitations associated with LangChain.

### <a id="q-is-crewai-open-source"></a>Q: Is CrewAI open-source?

A: Yes, CrewAI is open-source and actively encourages community contributions and collaboration.

### <a id="q-does-crewai-collect-data-from-users"></a>Q: Does CrewAI collect data from users?

A: CrewAI collects anonymous telemetry data strictly for improvement purposes. Sensitive data such as prompts, tasks, or API responses are never collected unless explicitly enabled by the user.

### <a id="q-where-can-i-find-real-world-crewai-examples"></a>Q: Where can I find real-world CrewAI examples?

A: Check out practical examples in the [CrewAI-examples repository](https://github.com/crewAIInc/crewAI-examples), covering use cases like trip planners, stock analysis, and job postings.

### <a id="q-how-can-i-contribute-to-crewai"></a>Q: How can I contribute to CrewAI?

A: Contributions are warmly welcomed! Fork the repository, create your branch, implement your changes, and submit a pull request. See the Contribution section of the README for detailed guidelines.

### <a id="q-what-additional-features-does-crewai-enterprise-offer"></a>Q: What additional features does CrewAI Enterprise offer?

A: CrewAI Enterprise provides advanced features such as a unified control plane, real-time observability, secure integrations, advanced security, actionable insights, and dedicated 24/7 enterprise support.

### <a id="q-is-crewai-enterprise-available-for-cloud-and-on-premise-deployments"></a>Q: Is CrewAI Enterprise available for cloud and on-premise deployments?

A: Yes, CrewAI Enterprise supports both cloud-based and on-premise deployment options, allowing enterprises to meet their specific security and compliance requirements.

### <a id="q-can-i-try-crewai-enterprise-for-free"></a>Q: Can I try CrewAI Enterprise for free?

A: Yes, you can explore part of the CrewAI Enterprise Suite by accessing the [Crew Control Plane](https://app.crewai.com) for free.

### <a id="q-does-crewai-support-fine-tuning-or-training-custom-models"></a>Q: Does CrewAI support fine-tuning or training custom models?

A: Yes, CrewAI can integrate with custom-trained or fine-tuned models, allowing you to enhance your agents with domain-specific knowledge and accuracy.

### Q: Can CrewAI agents interact with external tools and APIs?

A: Absolutely! CrewAI agents can easily integrate with external tools, APIs, and databases, empowering them to leverage real-world data and resources.

### Q: Is CrewAI suitable for production environments?

A: Yes, CrewAI is explicitly designed with production-grade standards, ensuring reliability, stability, and scalability for enterprise deployments.

### Q: How scalable is CrewAI?

A: CrewAI is highly scalable, supporting simple automations and large-scale enterprise workflows involving numerous agents and complex tasks simultaneously.

### Q: Does CrewAI offer debugging and monitoring tools?

A: Yes, CrewAI Enterprise includes advanced debugging, tracing, and real-time observability features, simplifying the management and troubleshooting of your automations.

### Q: What programming languages does CrewAI support?

A: CrewAI is primarily Python-based but easily integrates with services and APIs written in any programming language through its flexible API integration capabilities.

### Q: Does CrewAI offer educational resources for beginners?

A: Yes, CrewAI provides extensive beginner-friendly tutorials, courses, and documentation through learn.crewai.com, supporting developers at all skill levels.

### Q: Can CrewAI automate human-in-the-loop workflows?

A: Yes, CrewAI fully supports human-in-the-loop workflows, allowing seamless collaboration between human experts and AI agents for enhanced decision-making.