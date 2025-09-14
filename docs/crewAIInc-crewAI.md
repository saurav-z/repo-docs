<p align="center">
  <a href="https://github.com/crewAIInc/crewAI">
    <img src="docs/images/crewai_logo.png" width="600px" alt="Open source Multi-AI Agent orchestration framework">
  </a>
</p>

<p align="center">
  **Unlock the Power of Autonomous AI with CrewAI: The Lightning-Fast Framework for Multi-Agent Orchestration.**
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
  ·
  <a href="https://github.com/crewAIInc/crewAI">GitHub</a>
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

## Key Features of CrewAI

CrewAI is a revolutionary, **standalone** Python framework, designed for lightning-fast, flexible, and enterprise-ready multi-agent automation.

*   **Standalone & Lean:** Completely independent of LangChain and other agent frameworks, ensuring faster execution and minimal resource usage.
*   **Crews:** Effortlessly create autonomous agent teams for collaborative intelligence and complex task execution.
*   **Flows:** Gain granular, event-driven control for precise task orchestration and production-ready workflows.
*   **Seamless Integration:** Combine Crews and Flows for unparalleled control and flexibility.
*   **Deep Customization:** Tailor every aspect from high-level workflows to low-level agent behaviors and prompts.
*   **High Performance:** Optimized for speed and minimal resource usage, enabling faster execution.
*   **Robust Community:** Benefit from a rapidly growing community with comprehensive support and extensive documentation.
*   **Enterprise-Ready:** Designed for simple tasks to highly complex, real-world, enterprise-grade scenarios.

[Explore the CrewAI GitHub Repository](https://github.com/crewAIInc/crewAI)

## Why Choose CrewAI?

<div align="center" style="margin-bottom: 30px;">
  <img src="docs/images/asset.png" alt="CrewAI Logo" width="100%">
</div>

CrewAI empowers developers and enterprises to build powerful, adaptable, and production-ready AI automations with ease, offering the best combination of speed, flexibility, and control.
CrewAI simplifies the development of autonomous AI agents, allowing you to build complex solutions with less code and more control. With its lean design, extensive customization options, and strong community support, CrewAI is the go-to choice for creating intelligent and efficient AI-powered systems.

*   **Unmatched Flexibility**: Craft solutions that perfectly fit your needs, whether you’re tackling simple tasks or complex, real-world challenges.
*   **Exceptional Performance**: Experience lightning-fast execution and efficient resource utilization.
*   **Complete Control**: Customize every aspect of your AI agents, from their roles and goals to the logic that governs their interactions.
*   **Community-Driven Excellence**: Join a thriving community with over 100,000 certified developers, and access extensive documentation and support.

## Getting Started

Get started with CrewAI in minutes. Watch our getting started tutorial:

[![CrewAI Getting Started Tutorial](https://img.youtube.com/vi/-kSOTtYzgEw/hqdefault.jpg)](https://www.youtube.com/watch?v=-kSOTtYzgEw "CrewAI Getting Started Tutorial")

### Learning Resources

Master CrewAI with our in-depth courses:

*   [Multi AI Agent Systems with CrewAI](https://www.deeplearning.ai/short-courses/multi-ai-agent-systems-with-crewai/) - Master the fundamentals of multi-agent systems
*   [Practical Multi AI Agents and Advanced Use Cases](https://www.deeplearning.ai/short-courses/practical-multi-ai-agents-and-advanced-use-cases-with-crewai/) - Deep dive into advanced implementations

### Installation

1.  **Prerequisites:** Ensure you have Python >=3.10 and <3.14 installed on your system.

2.  **Install CrewAI:**

    ```bash
    pip install crewai
    ```

    For extended functionality, including tools, install with:

    ```bash
    pip install 'crewai[tools]'
    ```

### Troubleshooting

*   **`ModuleNotFoundError: No module named 'tiktoken'`**: Install `tiktoken` explicitly with:
    ```bash
    pip install 'crewai[embeddings]'
    ```
    or if using embedchain or other tools:
    ```bash
    pip install 'crewai[tools]'
    ```

*   **`Failed building wheel for tiktoken`**:
    *   Ensure Rust compiler is installed.
    *   On Windows: Verify Visual C++ Build Tools are installed.
    *   Upgrade pip: `pip install --upgrade pip`.
    *   If issues persist, try a pre-built wheel: `pip install tiktoken --prefer-binary`.

### Basic Project Structure

Create a new CrewAI project using the CLI:

```bash
crewai create crew <project_name>
```

This generates a project with the following structure:

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

Customize your project by editing the files within the `src/my_project` folder.

*   `main.py`: Project entry point.
*   `crew.py`: Define your crew.
*   `agents.yaml`: Define your agents.
*   `tasks.yaml`: Define your tasks.

Add environment variables (e.g., OpenAI API key) to the `.env` file.

### Example Project Configuration

**agents.yaml:**

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

**tasks.yaml:**

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

**crew.py:**

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

**main.py:**

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

1.  **Set Environment Variables:** Ensure your `.env` file contains your OpenAI API key (`OPENAI_API_KEY=sk-...`) and, if applicable, a Serper.dev API key (`SERPER_API_KEY=YOUR_KEY_HERE`).

2.  **Lock and Install Dependencies (Optional):**

    ```bash
    cd my_project
    crewai install
    ```

3.  **Run Your Crew:**

    ```bash
    crewai run
    ```

    or

    ```bash
    python src/my_project/main.py
    ```

    If you face issues with Poetry, update the CrewAI package:

    ```bash
    crewai update
    ```

    The output will display in the console, and the generated `report.md` will be in your project's root directory.

    You can also use the hierarchical process, automatically assigning a manager to properly coordinate the planning and execution of tasks.

## Examples

Explore real-world applications of CrewAI:

*   [**Landing Page Generator**](https://github.com/crewAIInc/crewAI-examples/tree/main/crews/landing_page_generator)
*   [**Human Input on the Execution**](https://docs.crewai.com/how-to/Human-Input-on-Execution)
*   [**Trip Planner**](https://github.com/crewAIInc/crewAI-examples/tree/main/crews/trip_planner)
*   [**Stock Analysis**](https://github.com/crewAIInc/crewAI-examples/tree/main/crews/stock_analysis)

### Additional Examples

*   **Quick Tutorial**

    [![CrewAI Tutorial](https://img.youtube.com/vi/tnejrr-0a94/maxresdefault.jpg)](https://www.youtube.com/watch?v=tnejrr-0a94 "CrewAI Tutorial")

*   **Write Job Descriptions**

    [Check out code for this example](https://github.com/crewAIInc/crewAI-examples/tree/main/crews/job-posting) or watch a video below:

    [![Jobs postings](https://img.youtube.com/vi/u98wEMz-9to/maxresdefault.jpg)](https://www.youtube.com/watch?v=u98wEMz-9to "Jobs postings")

*   **Trip Planner**

    [Check out code for this example](https://github.com/crewAIInc/crewAI-examples/tree/main/crews/trip_planner) or watch a video below:

    [![Trip Planner](https://img.youtube.com/vi/xis7rWp-hjs/maxresdefault.jpg)](https://www.youtube.com/watch?v=xis7rWp-hjs "Trip Planner")

*   **Stock Analysis**

    [Check out code for this example](https://github.com/crewAIInc/crewAI-examples/tree/main/crews/stock_analysis) or watch a video below:

    [![Stock Analysis](https://img.youtube.com/vi/e0Uj4yWdaAg/maxresdefault.jpg)](https://www.youtube.com/watch?v=e0Uj4yWdaAg "Stock Analysis")

## Combining Crews and Flows

CrewAI's power expands when combining Crews with Flows for sophisticated automation pipelines. Flows support logical operators like or\_ and and\_ to combine multiple conditions.

*   `or_`: Triggers if any specified condition is met.
*   `and_`: Triggers if all specified conditions are met.

Example:

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

This example illustrates the use of Python code for data operations, creating and executing Crews, and managing workflow sequences.

## Connecting to Models

CrewAI supports various LLMs. Agents use the OpenAI API by default, but you can configure agents to use a local model. See the [LLM Connections documentation](https://docs.crewai.com/how-to/LLM-Connections/) for configuration details.

## How CrewAI Compares

**CrewAI Advantages**: CrewAI combines autonomous agent intelligence with precise workflow control through its unique Crews and Flows architecture. The framework excels at both high-level orchestration and low-level customization, enabling complex, production-grade systems with granular control.

*   **LangGraph**:  LangGraph requires significant boilerplate code, complex state management, and tight coupling with LangChain.
    *   CrewAI executes faster (e.g., 5.76x faster in a QA task)
*   **Autogen**: Autogen lacks inherent process concepts, and orchestrating interactions requires complex programming.
*   **ChatDev**: Limited customization and focus on production environments can hinder scalability and flexibility.

## Contributing

Contributions are welcome!

1.  Fork the repository.
2.  Create a new branch.
3.  Add your improvements.
4.  Submit a pull request.

### Installation

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

CrewAI uses anonymous telemetry to improve the library by focusing on the most used features, integrations, and tools.

NO data is collected concerning prompts, task descriptions, agents' backstories, or goals, tools used, API calls, responses, or secrets, with the exception of the conditions mentioned. When the `share_crew` feature is enabled, detailed data including task descriptions, agents' backstories, or goals, and other specific attributes are collected to provide deeper insights while respecting user privacy. Users can disable telemetry by setting the environment variable OTEL_SDK_DISABLED to true.

Data collected includes:

*   CrewAI Version
*   Python Version
*   General OS
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

A: CrewAI is a standalone, lean, and fast Python framework built specifically for orchestrating autonomous AI agents, offering a faster, leaner, and simpler experience without dependencies on frameworks like LangChain.

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

A: No, CrewAI is built from the ground up with no dependencies on LangChain or other agent frameworks, ensuring a lean, fast, and flexible experience.

### Q: Can CrewAI handle complex use cases?

A: Yes, CrewAI excels at both simple and highly complex real-world scenarios by offering deep customization options.

### Q: Can I use CrewAI with local AI models?

A: Yes, CrewAI supports various language models, including local ones via tools like Ollama.

### Q: What makes Crews different from Flows?

A: Crews are for autonomous agent collaboration, while Flows are for precise, event-driven control. You can combine both for maximum effectiveness.

### Q: How is CrewAI better than LangChain?

A: CrewAI provides simpler APIs, faster execution, more reliable results, robust documentation, and an active community, addressing common limitations.

### Q: Is CrewAI open-source?

A: Yes, CrewAI is open-source.

### Q: Does CrewAI collect data from users?

A: CrewAI collects anonymous telemetry data strictly for improvement. Sensitive data are never collected unless explicitly enabled by the user.

### Q: Where can I find real-world CrewAI examples?

A: Check out practical examples in the [CrewAI-examples repository](https://github.com/crewAIInc/crewAI-examples).

### Q: How can I contribute to CrewAI?

A: Contributions are welcome! Fork the repository, create your branch, implement your changes, and submit a pull request. See the Contribution section.

### Q: What additional features does CrewAI Enterprise offer?

A: CrewAI Enterprise offers a unified control plane, real-time observability, secure integrations, advanced security, actionable insights, and dedicated 24/7 enterprise support.

### Q: Is CrewAI Enterprise available for cloud and on-premise deployments?

A: Yes, CrewAI Enterprise supports cloud-based and on-premise deployments.

### Q: Can I try CrewAI Enterprise for free?

A: Yes, explore the Crew Control Plane for free at [https://app.crewai.com](https://app.crewai.com).

### Q: Does CrewAI support fine-tuning or training custom models?

A: Yes, CrewAI integrates with custom-trained or fine-tuned models.

### Q: Can CrewAI agents interact with external tools and APIs?

A: Absolutely! CrewAI agents can easily integrate with external tools, APIs, and databases.

### Q: Is CrewAI suitable for production environments?

A: Yes, CrewAI is explicitly designed with production-grade standards.

### Q: How scalable is CrewAI?

A: CrewAI is highly scalable, supporting simple automations and large-scale enterprise workflows.

### Q: Does CrewAI offer debugging and monitoring tools?

A: Yes, CrewAI Enterprise includes advanced debugging, tracing, and real-time observability features.

### Q: What programming languages does CrewAI support?

A: CrewAI is primarily Python-based, but integrates with other languages via API integration.

### Q: Does CrewAI offer educational resources for beginners?

A: Yes, CrewAI provides extensive beginner-friendly tutorials and documentation at learn.crewai.com.

### Q: Can CrewAI automate human-in-the-loop workflows?

A: Yes, CrewAI fully supports human-in-the-loop workflows.