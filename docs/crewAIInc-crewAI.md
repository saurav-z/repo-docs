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

## Revolutionize AI Automation with CrewAI: The Ultimate Multi-Agent Framework

**CrewAI** is a cutting-edge Python framework that empowers developers to build sophisticated, autonomous AI agents with unparalleled speed, flexibility, and control. ([Explore the original repo](https://github.com/crewAIInc/crewAI))

### Key Features

*   **Standalone & Lightweight:** Independent of LangChain and other agent frameworks, ensuring faster execution and reduced resource consumption.
*   **Autonomous Crews & Precise Flows:** Choose between collaborative, intelligent Crews or event-driven Flows for optimal task orchestration.
*   **Seamless Integration:** Combine Crews and Flows to create powerful, production-ready AI automations.
*   **Deep Customization:** Tailor every aspect of your AI agents, from workflows and agent behavior to internal prompts.
*   **High Performance:** Optimized for speed and efficiency, making complex tasks run smoothly.
*   **Robust Community:** Benefit from comprehensive documentation and a rapidly growing community of 100,000+ certified developers.

### Why Choose CrewAI?

<div align="center" style="margin-bottom: 30px;">
  <img src="docs/images/asset.png" alt="CrewAI Logo" width="100%">
</div>

CrewAI offers the perfect blend of simplicity, flexibility, and performance, setting the standard for AI automation.

*   **Complete Independence:** Built from the ground up, CrewAI eliminates the complexities and limitations of other frameworks.
*   **Unrivaled Performance:** Optimized for speed and minimal resource usage, allowing faster execution.
*   **Unmatched Flexibility:** Complete freedom to customize at both high and low levels.
*   **Enterprise-Ready:** Proven effective for both simple tasks and highly complex, real-world scenarios.

### Getting Started

Follow these simple steps to install and run your first CrewAI agents:

[![CrewAI Getting Started Tutorial](https://img.youtube.com/vi/-kSOTtYzgEw/hqdefault.jpg)](https://www.youtube.com/watch?v=-kSOTtYzgEw "CrewAI Getting Started Tutorial")

#### Learning Resources

Master CrewAI with these comprehensive courses:

*   [Multi AI Agent Systems with CrewAI](https://www.deeplearning.ai/short-courses/multi-ai-agent-systems-with-crewai/) - Learn the fundamentals of multi-agent systems.
*   [Practical Multi AI Agents and Advanced Use Cases with CrewAI](https://www.deeplearning.ai/short-courses/practical-multi-ai-agents-and-advanced-use-cases-with-crewai/) - Dive into advanced implementations.

### Understanding Flows and Crews

CrewAI leverages two powerful approaches:

1.  **Crews:** Teams of AI agents that work autonomously through role-based collaboration.
    *   Natural, autonomous decision-making between agents
    *   Dynamic task delegation and collaboration
    *   Specialized roles with defined goals and expertise
    *   Flexible problem-solving approaches
2.  **Flows:** Event-driven workflows, providing precise control over automations.
    *   Fine-grained control over execution paths.
    *   Secure, consistent state management.
    *   Clean integration of agents with production code.
    *   Conditional branching for complex business logic.

The true power of CrewAI is combining Crews and Flows, enabling you to:

*   Build complex, production-grade applications
*   Balance autonomy with precision
*   Handle sophisticated real-world scenarios
*   Maintain clean, maintainable code structure

### Installation

To get started, ensure you have Python >=3.10 <3.14 installed. CrewAI uses [UV](https://docs.astral.sh/uv/) for dependency management.

1.  Install CrewAI:

```shell
pip install crewai
```

2.  Install optional tools:

```shell
pip install 'crewai[tools]'
```

#### Troubleshooting Dependencies

Address common issues during installation:

1.  **ModuleNotFoundError: No module named 'tiktoken'**
    *   Install tiktoken explicitly: `pip install 'crewai[embeddings]'`
    *   Or for embedchain: `pip install 'crewai[tools]'`
2.  **Failed building wheel for tiktoken**
    *   Install Rust compiler (see installation steps).
    *   Verify Visual C++ Build Tools on Windows.
    *   Upgrade pip: `pip install --upgrade pip`.
    *   Use a pre-built wheel: `pip install tiktoken --prefer-binary`.

### Project Setup

Create your CrewAI project using the CLI:

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

Customize your project by editing:

*   `src/my_project/config/agents.yaml` (define agents)
*   `src/my_project/config/tasks.yaml` (define tasks)
*   `src/my_project/crew.py` (add logic, tools)
*   `src/my_project/main.py` (add inputs)
*   `.env` (add environment variables)

#### Simple Crew Example

Create and run your crew:

```shell
crewai create crew latest-ai-development
```

Modify the configuration files:

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

1.  Set required environment variables in your `.env` file:

    *   `OPENAI_API_KEY=sk-...` (or other LLM API key)
    *   `SERPER_API_KEY=YOUR_KEY_HERE`

2.  Navigate to your project directory: `cd my_project`

3.  Install dependencies (optional): `crewai install`

4.  Run your crew:

```bash
crewai run
```

or

```bash
python src/my_project/main.py
```

If there are issues with poetry, update the package:

```bash
crewai update
```

You should see output and the `report.md` file in your project root.

### Key Features

CrewAI's standout features provide a flexible and powerful way to orchestrate AI agents:

*   **Standalone & Lean:** No dependency on other frameworks.
*   **Crews & Flows:** Combine Crews (autonomy) and Flows (control).
*   **Deep Customization:** Tailor workflows and agent behaviors.
*   **High Performance:** Optimized for speed and minimal resource usage.
*   **Thriving Community:** Extensive support and over 100,000 certified developers.

### Examples

Explore real-world examples in the [CrewAI-examples repo](https://github.com/crewAIInc/crewAI-examples?tab=readme-ov-file):

*   [Landing Page Generator](https://github.com/crewAIInc/crewAI-examples/tree/main/landing_page_generator)
*   [Human Input on Execution](https://docs.crewai.com/how-to/Human-Input-on-Execution)
*   [Trip Planner](https://github.com/crewAIInc/crewAI-examples/tree/main/trip_planner)
*   [Stock Analysis](https://github.com/crewAIInc/crewAI-examples/tree/main/stock_analysis)

#### Quick Tutorials

[![CrewAI Tutorial](https://img.youtube.com/vi/tnejrr-0a94/maxresdefault.jpg)](https://www.youtube.com/watch?v=tnejrr-0a94 "CrewAI Tutorial")

#### Job Descriptions

[Check out code for this example](https://github.com/crewAIInc/crewAI-examples/tree/main/job-posting) or watch a video below:

[![Jobs postings](https://img.youtube.com/vi/u98wEMz-9to/maxresdefault.jpg)](https://www.youtube.com/watch?v=u98wEMz-9to "Jobs postings")

#### Trip Planner

[Check out code for this example](https://github.com/crewAIInc/crewAI-examples/tree/main/trip_planner) or watch a video below:

[![Trip Planner](https://img.youtube.com/vi/xis7rWp-hjs/maxresdefault.jpg)](https://www.youtube.com/watch?v=xis7rWp-hjs "Trip Planner")

#### Stock Analysis

[Check out code for this example](https://github.com/crewAIInc/crewAI-examples/tree/main/stock_analysis) or watch a video below:

[![Stock Analysis](https://img.youtube.com/vi/e0Uj4yWdaAg/maxresdefault.jpg)](https://www.youtube.com/watch?v=e0Uj4yWdaAg "Stock Analysis")

### Using Crews and Flows Together

Combine Crews and Flows for sophisticated automation pipelines:

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

### Connecting to Models

CrewAI supports various LLMs via a variety of connection options. By default your agents will use the OpenAI API when querying the model. Configure model connections, learn more about [LLM Connections](https://docs.crewai.com/how-to/LLM-Connections/).

### How CrewAI Compares

**CrewAI's Advantage**: CrewAI's unique Crews and Flows architecture balances agent intelligence with precise workflow control.

-   **LangGraph**: CrewAI's leaner architecture and lower boilerplate offer increased flexibility.

*P.S. CrewAI runs 5.76x faster in specific scenarios like a QA task ([comparison](https://github.com/crewAIInc/crewAI-examples/tree/main/Notebooks/CrewAI%20Flows%20%26%20Langgraph/QA%20Agent)), and achieves better performance in coding tasks as well ([detailed analysis](https://github.com/crewAIInc/crewAI-examples/blob/main/Notebooks/CrewAI%20Flows%20%26%20Langgraph/Coding%20Assistant/coding_assistant_eval.ipynb)).*

-   **Autogen**: Autogen can become complex.

-   **ChatDev**: ChatDev is limited and not geared towards production environments

### Contribution

CrewAI welcomes contributions.

*   Fork the repository.
*   Create a new branch.
*   Add your features or improvements.
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

### Telemetry

CrewAI uses anonymous telemetry.
Users can disable telemetry by setting the environment variable OTEL_SDK_DISABLED to true.

Data collected includes:
- Version of CrewAI
- Version of Python
- General OS
- Number of agents and tasks in a crew
- Crew Process being used
- If Agents are using memory or allowing delegation
- If Tasks are being executed in parallel or sequentially
- Language model being used
- Roles of agents in a crew
- Tools names available

Users can enable Further Telemetry by sharing the complete telemetry data by setting the `share_crew` attribute to `True` on their Crews.

### License

CrewAI is released under the [MIT License](https://github.com/crewAIInc/crewAI/blob/main/LICENSE).

### Frequently Asked Questions (FAQ)

#### General

*   **Q: What exactly is CrewAI?**
*   **Q: How do I install CrewAI?**
*   **Q: Does CrewAI depend on LangChain?**
*   **Q: Is CrewAI open-source?**
*   **Q: Does CrewAI collect data from users?**

#### Features and Capabilities

*   **Q: Can CrewAI handle complex use cases?**
*   **Q: Can I use CrewAI with local AI models?**
*   **Q: What makes Crews different from Flows?**
*   **Q: How is CrewAI better than LangChain?**
*   **Q: Does CrewAI support fine-tuning or training custom models?**

#### Resources and Community

*   **Q: Where can I find real-world CrewAI examples?**
*   **Q: How can I contribute to CrewAI?**

#### Enterprise Features

*   **Q: What additional features does CrewAI Enterprise offer?**
*   **Q: Is CrewAI Enterprise available for cloud and on-premise deployments?**
*   **Q: Can I try CrewAI Enterprise for free?**

#### Q: What exactly is CrewAI?

A: CrewAI is a standalone, lean, and fast Python framework for autonomous AI agent orchestration, independent of LangChain.

#### Q: How do I install CrewAI?

A: Install via `pip install crewai`.

#### Q: Does CrewAI depend on LangChain?

A: No, CrewAI is independent of LangChain and other frameworks.

#### Q: Can CrewAI handle complex use cases?

A: Yes, CrewAI excels in both simple and complex, real-world scenarios.

#### Q: Can I use CrewAI with local AI models?

A: Yes, CrewAI supports local models via tools like Ollama.

#### Q: What makes Crews different from Flows?

A: Crews are for autonomous agent collaboration, while Flows are for precise, event-driven control.

#### Q: How is CrewAI better than LangChain?

A: CrewAI has a simpler API, faster execution, more reliable results, and a thriving community.

#### Q: Is CrewAI open-source?

A: Yes.

#### Q: Does CrewAI collect data from users?

A: CrewAI collects anonymous telemetry for improvement; personal data is not collected.

#### Q: Where can I find real-world CrewAI examples?

A: See the [CrewAI-examples repo](https://github.com/crewAIInc/crewAI-examples?tab=readme-ov-file).

#### Q: How can I contribute to CrewAI?

A: Fork the repo, create a branch, make changes, and submit a pull request.

#### Q: What additional features does CrewAI Enterprise offer?

A: Control Plane, observability, integrations, security, insights, and 24/7 support.

#### Q: Is CrewAI Enterprise available for cloud and on-premise deployments?

A: Yes.

#### Q: Can I try CrewAI Enterprise for free?

A: Yes, explore the [Crew Control Plane](https://app.crewai.com) for free.

#### Q: Does CrewAI support fine-tuning or training custom models?

A: Yes.

#### Q: Can CrewAI agents interact with external tools and APIs?

A: Yes.

#### Q: Is CrewAI suitable for production environments?

A: Yes.

#### Q: How scalable is CrewAI?

A: Highly scalable, supporting all levels.

#### Q: Does CrewAI offer debugging and monitoring tools?

A: Yes, CrewAI Enterprise offers advanced debugging.

#### Q: What programming languages does CrewAI support?

A: Primarily Python; API integration available.

#### Q: Does CrewAI offer educational resources for beginners?

A: Yes, via learn.crewai.com.

#### Q: Can CrewAI automate human-in-the-loop workflows?

A: Yes.