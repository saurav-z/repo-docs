<p align="center">
  <a href="https://github.com/crewAIInc/crewAI">
    <img src="docs/images/crewai_logo.png" width="600px" alt="Open source Multi-AI Agent orchestration framework">
  </a>
</p>

<p align="center">
  <b>Unleash the power of AI: CrewAI is the lightning-fast framework for building autonomous, collaborative AI agent systems.</b>
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

## CrewAI: The Leading Framework for Multi-Agent AI Automation

[CrewAI](https://github.com/crewAIInc/crewAI) is a cutting-edge, standalone Python framework designed to simplify the creation of autonomous AI agent systems.  It gives developers the speed and flexibility to build highly customized AI solutions.  Join a community of 100,000+ certified developers!

**Key Features:**

*   🚀 **Standalone & Lean:**  Built from the ground up, independent of frameworks like LangChain, for faster performance and reduced resource usage.
*   🧠 **Autonomous Agent Orchestration:** Build collaborative AI teams with intuitive [Crews](https://docs.crewai.com/concepts/crews) or create precise event-driven workflows with [Flows](https://docs.crewai.com/concepts/flows).
*   🔄 **Seamless Integration:** Combine Crews (autonomy) and Flows (precision) for complex, real-world automations.
*   ⚙️ **Deep Customization:** Control every aspect of your AI agents, from high-level workflows to low-level prompts and agent behaviors.
*   ✅ **Reliable Performance:** Achieve consistent results across a variety of tasks, from simple to complex enterprise-level automations.
*   🤝 **Thriving Community:** Benefit from comprehensive documentation and support from a growing community of over 100,000 developers.

## Table of Contents

*   [Why CrewAI?](#why-crewai)
*   [Getting Started](#getting-started)
    *   [Installation](#getting-started-installation)
    *   [Setting Up Your Crew](#setting-up-your-crew)
    *   [Running Your Crew](#running-your-crew)
*   [Key Features](#key-features)
*   [Understanding Flows and Crews](#understanding-flows-and-crews)
*   [CrewAI vs. Other Frameworks](#how-crewai-compares)
*   [Examples](#examples)
    *   [Quick Tutorial](#quick-tutorial)
    *   [Write Job Descriptions](#write-job-descriptions)
    *   [Trip Planner](#trip-planner)
    *   [Stock Analysis](#stock-analysis)
    *   [Using Crews and Flows Together](#using-crews-and-flows-together)
*   [Connecting Your Crew to a Model](#connecting-your-crew-to-a-model)
*   [How CrewAI Compares](#how-crewai-compares)
*   [Contribution](#contribution)
*   [Telemetry](#telemetry)
*   [License](#license)
*   [Frequently Asked Questions (FAQ)](#frequently-asked-questions-faq)

## Why CrewAI?

<div align="center" style="margin-bottom: 30px;">
  <img src="docs/images/asset.png" alt="CrewAI Logo" width="100%">
</div>

CrewAI provides the best-in-class combination of speed, flexibility, and control with either Crews of AI Agents or Flows of Events:

*   **Standalone Framework**: Not dependent on external frameworks, which means better performance.
*   **High Performance**: Optimized for fast execution.
*   **Flexible Customization**: Customize workflows, agent behaviors, prompts, and execution logic.
*   **Ideal for Every Use Case**: Suitable for both simple and enterprise-grade scenarios.
*   **Robust Community**:  Backed by a large and growing community.

## Getting Started

Get up and running with your first CrewAI agents quickly.

[![CrewAI Getting Started Tutorial](https://img.youtube.com/vi/-kSOTtYzgEw/hqdefault.jpg)](https://www.youtube.com/watch?v=-kSOTtYzgEw "CrewAI Getting Started Tutorial")

### Learning Resources

Learn CrewAI through our comprehensive courses:

*   [Multi AI Agent Systems with CrewAI](https://www.deeplearning.ai/short-courses/multi-ai-agent-systems-with-crewai/) - Master the fundamentals of multi-agent systems
*   [Practical Multi AI Agents and Advanced Use Cases](https://www.deeplearning.ai/short-courses/practical-multi-ai-agents-and-advanced-use-cases-with-crewai/) - Deep dive into advanced implementations

### Installation

Ensure you have Python >=3.10 <3.14 installed.

```bash
pip install crewai
```

To install with optional tools, use:

```bash
pip install 'crewai[tools]'
```

### Setting Up Your Crew

1.  Create a new project using the CLI:

    ```bash
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

2.  Customize your project by editing these files:

    *   `src/my_project/config/agents.yaml`: Define your agents.
    *   `src/my_project/config/tasks.yaml`: Define your tasks.
    *   `src/my_project/crew.py`: Add logic, tools, and specific arguments.
    *   `src/my_project/main.py`: Add custom inputs.
    *   `.env`: Add environment variables.

3.  Example:

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

### Running Your Crew

1.  Set up your environment variables in `.env`:

    *   `OPENAI_API_KEY=sk-...` (or other LLM API key)
    *   `SERPER_API_KEY=YOUR_KEY_HERE` (for web search)

2.  Lock dependencies:

    ```bash
    cd my_project
    crewai install
    ```

3.  Run your crew:

    ```bash
    crewai run
    ```
    or
    ```bash
    python src/my_project/main.py
    ```

## Understanding Flows and Crews

CrewAI offers two powerful, synergistic approaches:

1.  **Crews:**  Teams of autonomous AI agents designed for collaborative problem-solving.
    *   Enable natural, autonomous decision-making between agents.
    *   Facilitate dynamic task delegation and collaboration.
    *   Allow specialized roles with defined goals and expertise.
    *   Provide flexible problem-solving approaches.

2.  **Flows:** Event-driven workflows for precise control over complex automations.
    *   Provide fine-grained control over execution paths.
    *   Offer secure and consistent state management.
    *   Enable clean integration of agents with Python code.
    *   Support conditional branching for complex logic.

Combining Crews and Flows allows you to:

*   Build complex, production-grade applications.
*   Balance autonomy and control.
*   Handle sophisticated real-world scenarios.
*   Maintain clean and manageable code.

## CrewAI vs. Other Frameworks

**CrewAI's Advantage**: CrewAI combines autonomous agent intelligence with precise workflow control through its unique Crews and Flows architecture. The framework excels at both high-level orchestration and low-level customization, enabling complex, production-grade systems with granular control.

*   **LangGraph**:  LangGraph requires significant boilerplate code and complex state management. Its tight coupling with LangChain can limit flexibility.
*   **Autogen**: While Autogen excels at creating conversational agents, it lacks an inherent concept of process. Orchestrating agents' interactions requires extra programming, which grows complex with scale.
*   **ChatDev**: ChatDev has rigid implementations and limited customizations, which hinder scalability and flexibility in real-world applications.

## Examples

Find various real-life examples of AI crews in the [CrewAI-examples repo](https://github.com/crewAIInc/crewAI-examples?tab=readme-ov-file):

*   [Landing Page Generator](https://github.com/crewAIInc/crewAI-examples/tree/main/landing_page_generator)
*   [Human input on the execution](https://docs.crewai.com/how-to/Human-Input-on-Execution)
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

Leverage the power of Crews and Flows to create powerful automation pipelines.

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

CrewAI supports different LLMs. By default, agents use the OpenAI API. For local models, configure agents via the Ollama tool.

*Refer to the [LLM Connections](https://docs.crewai.com/how-to/LLM-Connections/) page.*

## How CrewAI Compares

CrewAI focuses on autonomous agent intelligence with precise workflow control through Crews and Flows, providing robust high-level and low-level customization.

*   **LangGraph**:  CrewAI significantly executes faster than LangGraph (see [comparison](https://github.com/crewAIInc/crewAI-examples/tree/main/Notebooks/CrewAI%20Flows%20%26%20Langgraph/QA%20Agent)) and also achieves higher evaluation scores and faster completion times ([detailed analysis](https://github.com/crewAIInc/crewAI-examples/blob/main/Notebooks/CrewAI%20Flows%20%26%20Langgraph/Coding%20Assistant/coding_assistant_eval.ipynb))
*   **Autogen**:  Limited built-in process concepts.
*   **ChatDev**: Limited customizations, not geared toward production environments.

## Contribution

We welcome contributions! Follow these steps:

1.  Fork the repository.
2.  Create a new branch.
3.  Add your feature/improvement.
4.  Submit a pull request.

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

CrewAI uses anonymous telemetry to help improve the library by focusing on the most used features, integrations, and tools.

**No data is collected** regarding prompts, task descriptions, agents' backstories/goals, tool usage, API calls, or any data processed by agents.  The only exception is if the `share_crew` feature is enabled.

Collected data includes:

*   CrewAI version
*   Python version
*   General OS (e.g., macOS/Windows/Linux)
*   Number of agents/tasks in a crew
*   Crew Process being used
*   If Agents are using memory or allowing delegation
*   If Tasks are executed in parallel or sequentially
*   Language model being used
*   Roles of agents in a crew
*   Tool names available

Users can opt-in to *Further Telemetry*, sharing the complete telemetry data by setting the `share_crew` attribute to `True` on their Crews. Enabling `share_crew` results in the collection of detailed crew and task execution data, including `goal`, `backstory`, `context`, and `output` of tasks. This enables a deeper insight into usage patterns while respecting the user's choice to share.

## License

CrewAI is released under the [MIT License](https://github.com/crewAIInc/crewAI/blob/main/LICENSE).

## Frequently Asked Questions (FAQ)

### General

*   [What is CrewAI?](#q-what-is-crewai)
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

A: CrewAI is a standalone, fast Python framework for building autonomous AI agent systems. It's independent of frameworks like LangChain.

### Q: How do I install CrewAI?

A: Use `pip install crewai`. For extra tools, use `pip install 'crewai[tools]'`.

### Q: Does CrewAI depend on LangChain?

A: No. CrewAI is independent of LangChain.

### Q: Can CrewAI handle complex use cases?

A: Yes, with deep customization.

### Q: Can I use CrewAI with local AI models?

A: Yes, via tools like Ollama.

### Q: What makes Crews different from Flows?

A: Crews offer autonomous collaboration; Flows offer precise control.

### Q: How is CrewAI better than LangChain?

A: CrewAI provides simpler APIs, faster speeds, and a more active community.

### Q: Is CrewAI open-source?

A: Yes, with community contributions welcome.

### Q: Does CrewAI collect data from users?

A: Anonymous telemetry for improvement. No sensitive data is collected unless `share_crew` is enabled.

### Q: Where can I find real-world CrewAI examples?

A: Check out the [CrewAI-examples repo](https://github.com/crewAIInc/crewAI-examples).

### Q: How can I contribute to CrewAI?

A: Fork, branch, add your changes, and submit a pull request.

### Q: What additional features does CrewAI Enterprise offer?

A: Control plane, observability, integrations, security, insights, and support.

### Q: Is CrewAI Enterprise available for cloud and on-premise deployments?

A: Yes.

### Q: Can I try CrewAI Enterprise for free?

A: Try the [Crew Control Plane](https://app.crewai.com) for free.

### Q: Does CrewAI support fine-tuning or training custom models?

A: Yes, CrewAI integrates with custom models.

### Q: Can CrewAI agents interact with external tools and APIs?

A: Yes, CrewAI agents easily integrate with external tools and APIs.

### Q: Is CrewAI suitable for production environments?

A: Yes, it's designed for production.

### Q: How scalable is CrewAI?

A: Highly scalable.

### Q: Does CrewAI offer debugging and monitoring tools?

A: CrewAI Enterprise includes debugging, tracing, and real-time observability features.

### Q: What programming languages does CrewAI support?

A: Python is the primary language, but it integrates with services in any language.

### Q: Does CrewAI offer educational resources for beginners?

A: Yes, CrewAI provides tutorials and courses via learn.crewai.com.

### Q: Can CrewAI automate human-in-the-loop workflows?

A: Yes, CrewAI supports human-in-the-loop workflows.