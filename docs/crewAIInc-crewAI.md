<p align="center">
  <a href="https://github.com/crewAIInc/crewAI">
    <img src="docs/images/crewai_logo.png" width="600px" alt="Open source Multi-AI Agent orchestration framework">
  </a>
</p>

<p align="center">
  **Unleash the Power of Autonomous AI Agents: Build Lightning-Fast, Flexible Automations with CrewAI!**
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

## What is CrewAI?

CrewAI is a cutting-edge, **standalone Python framework** designed to revolutionize multi-agent automation. Build powerful, intelligent, and production-ready AI applications with unprecedented speed, flexibility, and control.  Visit the [original repo](https://github.com/crewAIInc/crewAI) for more details.

**Key Features:**

*   **🚀 Standalone & Lean:**  Completely independent of LangChain and other frameworks, ensuring faster execution and lighter resource usage.
*   **🧠 Intelligent Crews:**  Orchestrate autonomous AI agents that collaborate to solve complex tasks.
*   **⚙️ Granular Flows:**  Precise, event-driven workflows for fine-grained control over complex automations.
*   **🛠️ Deep Customization:** Tailor everything from workflows to agent behavior and prompts.
*   **⚡ High Performance:** Optimized for speed and minimal resource usage, resulting in faster execution and reduced costs.
*   **🤝 Thriving Community:** Backed by a rapidly growing community of over 100,000 certified developers with extensive documentation and support.

## Why Choose CrewAI?

<div align="center" style="margin-bottom: 30px;">
  <img src="docs/images/asset.png" alt="CrewAI Logo" width="100%">
</div>

CrewAI empowers you to build intelligent automations that bridge the gap between simplicity, flexibility, and performance.  It offers:

*   **No external Dependencies:** A standalone framework built from scratch, independent of LangChain or any other agent framework.
*   **Optimized Performance:** High performance and minimal resource usage, enabling faster execution.
*   **Unparalleled Customization:** Complete freedom to customize at both high and low levels - from overall workflows and system architecture to granular agent behaviors, internal prompts, and execution logic.
*   **Versatile for Any Use Case:** Proven effective for both simple tasks and highly complex, real-world, enterprise-grade scenarios.
*   **Robust Community Support:** Backed by a rapidly growing community of over **100,000 certified** developers offering comprehensive support and resources.

## Getting Started

Get up and running with CrewAI in minutes!

[![CrewAI Getting Started Tutorial](https://img.youtube.com/vi/-kSOTtYzgEw/hqdefault.jpg)](https://www.youtube.com/watch?v=-kSOTtYzgEw "CrewAI Getting Started Tutorial")

### Learning Resources

Master CrewAI with comprehensive courses:

*   [Multi AI Agent Systems with CrewAI](https://www.deeplearning.ai/short-courses/multi-ai-agent-systems-with-crewai/) - Learn the fundamentals.
*   [Practical Multi AI Agents and Advanced Use Cases with CrewAI](https://www.deeplearning.ai/short-courses/practical-multi-ai-agents-and-advanced-use-cases-with-crewai/) - Explore advanced implementations.

### Installation

1.  **Prerequisites:**  Ensure you have Python >=3.10 <3.14 installed.

2.  **Install CrewAI:**

    ```bash
    pip install crewai
    ```

    For additional tools:

    ```bash
    pip install 'crewai[tools]'
    ```

### Troubleshooting Dependencies

*   **`ModuleNotFoundError: No module named 'tiktoken'`**: Install explicitly with `pip install 'crewai[embeddings]'` or if using embedchain: `pip install 'crewai[tools]'`
*   **`Failed building wheel for tiktoken`**: Ensure the Rust compiler is installed (see the install steps). Upgrade pip with `pip install --upgrade pip`

### Setting up your Crew

1.  Create a new project with the CLI:

    ```bash
    crewai create crew <project_name>
    ```

2.  Modify project files (`src/my_project/`):

    *   `agents.yaml`: Define your agents.
    *   `tasks.yaml`: Define your tasks.
    *   `crew.py`: Add custom logic, tools, and arguments.
    *   `main.py`: Add custom inputs for your agents and tasks.
    *   `.env`: Store your environment variables.

3.  **Example Crew (Sequential Process):**

    *   Instantiate a crew: `crewai create crew latest-ai-development`
    *   Modify the files:
        *   **agents.yaml:**
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

        *   **tasks.yaml:**
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

        *   **crew.py:**
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

        *   **main.py:**
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

4.  **Running Your Crew:**

    *   Set OpenAI and Serper.dev API keys as environment variables in your `.env` file: `OPENAI_API_KEY=sk-...` and `SERPER_API_KEY=YOUR_KEY_HERE`.
    *   Navigate to your project directory: `cd my_project`
    *   Lock and install your dependencies:
        ```bash
        crewai install (Optional)
        ```

    *   Run your crew:

        ```bash
        crewai run
        ```
        or
        ```bash
        python src/my_project/main.py
        ```
    *   If an error happens due to the usage of poetry, please run the following command to update your crewai package:
        ```bash
        crewai update
        ```

    *   Check your console output and the `report.md` file for results.

## Key Features

CrewAI empowers you to build powerful, adaptable, and production-ready AI automations.  Here's why it stands out:

*   **🛠️ Standalone & Lean:** Independent of LangChain and other frameworks, resulting in faster execution and reduced overhead.
*   **💡 Flexible & Precise:**  Leverage both [Crews](https://docs.crewai.com/concepts/crews) and [Flows](https://docs.crewai.com/concepts/flows) for tailored automation.
*   **🤝 Seamless Integration:** Effortlessly combine Crews (autonomy) and Flows (precision) for complex, real-world applications.
*   **⚙️ Deep Customization:**  Control every detail, from high-level workflows to internal prompts and agent behaviors.
*   **⚡ Reliable Performance:** Get consistent results across various tasks, from simple to complex enterprise-level automations.
*   **🌱 Thriving Community:** Benefit from extensive documentation and the support of a vibrant community with over 100,000 certified developers.

## Examples

Explore real-world use cases with our example repos:

*   [CrewAI-examples repo](https://github.com/crewAIInc/crewAI-examples?tab=readme-ov-file)
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

Unleash the power of CrewAI by combining Crews and Flows:

```python
from crewai.flow.flow import Flow, listen, start, router, or_
from crewai import Crew, Agent, Task, Process
from pydantic import BaseModel

class MarketState(BaseModel):
    sentiment: str = "neutral"
    confidence: float = 0.0
    recommendations: list = []

class AdvancedAnalysisFlow(Flow[MarketState]):
    @start()
    def fetch_market_data(self):
        self.state.sentiment = "analyzing"
        return {"sector": "tech", "timeframe": "1W"}

    @listen(fetch_market_data)
    def analyze_with_crew(self, market_data):
        analyst = Agent(...)
        researcher = Agent(...)
        analysis_task = Task(...)
        research_task = Task(...)

        analysis_crew = Crew(
            agents=[analyst, researcher],
            tasks=[analysis_task, research_task],
            process=Process.sequential,
            verbose=True
        )
        return analysis_crew.kickoff(inputs=market_data)

    @router(analyze_with_crew)
    def determine_next_steps(self):
        if self.state.confidence > 0.8:
            return "high_confidence"
        elif self.state.confidence > 0.5:
            return "medium_confidence"
        return "low_confidence"

    @listen("high_confidence")
    def execute_strategy(self):
        strategy_crew = Crew(...)
        return strategy_crew.kickoff()

    @listen(or_("medium_confidence", "low_confidence"))
    def request_additional_analysis(self):
        self.state.recommendations.append("Gather more data")
        return "Additional analysis required"
```

*This example shows how to combine Crews and Flows for advanced automation.*

## Connecting Your Crew to a Model

CrewAI supports a variety of LLMs.  Use the default OpenAI API, or configure agents to use local models like Ollama.  See [LLM Connections](https://docs.crewai.com/how-to/LLM-Connections/) for details.

## How CrewAI Compares

**CrewAI's Advantage:** CrewAI offers the best of both worlds: autonomous agent intelligence and precise workflow control through Crews and Flows.  It excels at both high-level orchestration and low-level customization.

*   **LangGraph:**  LangGraph requires significant boilerplate and complex state management.  Its coupling with LangChain can limit flexibility.
*   **Autogen:** Autogen lacks inherent process, so orchestrating agents' interactions requires additional programming, which can become complex.
*   **ChatDev:** ChatDev's implementations are rigid and not geared toward production, which can hinder scalability and flexibility in real-world applications.

*P.S. CrewAI demonstrates significant performance advantages over LangGraph, executing 5.76x faster in certain cases like this QA task example ([see comparison](https://github.com/crewAIInc/crewAI-examples/tree/main/Notebooks/CrewAI%20Flows%20%26%20Langgraph/QA%20Agent)) while achieving higher evaluation scores with faster completion times in certain coding tasks, like in this example ([detailed analysis](https://github.com/crewAIInc/crewAI-examples/blob/main/Notebooks/CrewAI%20Flows%20%26%20Langgraph/Coding%20Assistant/coding_assistant_eval.ipynb)).*

## Contribution

We welcome contributions!

1.  Fork the repository.
2.  Create a new branch.
3.  Add your feature or improvement.
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

CrewAI uses anonymous telemetry to improve the library by focusing on the most used features. No sensitive data is collected. You can disable telemetry by setting the `OTEL_SDK_DISABLED` environment variable to `true`.
You can opt-in to Further Telemetry, sharing the complete telemetry data by setting the `share_crew` attribute to `True` on their Crews.

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

A: CrewAI is a standalone, lean, and fast Python framework built specifically for orchestrating autonomous AI agents. Unlike frameworks like LangChain, CrewAI does not rely on external dependencies, making it leaner, faster, and simpler.

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

A: No. CrewAI is built entirely from the ground up, with no dependencies on LangChain or other agent frameworks. This ensures a lean, fast, and flexible experience.

### Q: Can CrewAI handle complex use cases?

A: Yes. CrewAI excels at both simple and highly complex real-world scenarios, offering deep customization options at both high and low levels, from internal prompts to sophisticated workflow orchestration.

### Q: Can I use CrewAI with local AI models?

A: Absolutely! CrewAI supports various language models, including local ones. Tools like Ollama and LM Studio allow seamless integration. Check the [LLM Connections documentation](https://docs.crewai.com/how-to/LLM-Connections/) for more details.

### Q: What makes Crews different from Flows?

A: Crews provide autonomous agent collaboration, ideal for tasks requiring flexible decision-making and dynamic interaction. Flows offer precise, event-driven control, ideal for managing detailed execution paths and secure state management. You can seamlessly combine both for maximum effectiveness.

### Q: How is CrewAI better than LangChain?

A: CrewAI provides simpler, more intuitive APIs, faster execution speeds, more reliable and consistent results, robust documentation, and an active community—addressing common criticisms and limitations associated with LangChain.

### Q: Is CrewAI open-source?

A: Yes, CrewAI is open-source and actively encourages community contributions and collaboration.

### Q: Does CrewAI collect data from users?

A: CrewAI collects anonymous telemetry data strictly for improvement purposes. Sensitive data such as prompts, tasks, or API responses are never collected unless explicitly enabled by the user.

### Q: Where can I find real-world CrewAI examples?

A: Check out practical examples in the [CrewAI-examples repository](https://github.com/crewAIInc/crewAI-examples), covering use cases like trip planners, stock analysis, and job postings.

### Q: How can I contribute to CrewAI?

A: Contributions are warmly welcomed! Fork the repository, create your branch, implement your changes, and submit a pull request. See the Contribution section of the README for detailed guidelines.

### Q: What additional features does CrewAI Enterprise offer?

A: CrewAI Enterprise provides advanced features such as a unified control plane, real-time observability, secure integrations, advanced security, actionable insights, and dedicated 24/7 enterprise support.

### Q: Is CrewAI Enterprise available for cloud and on-premise deployments?

A: Yes, CrewAI Enterprise supports both cloud-based and on-premise deployment options, allowing enterprises to meet their specific security and compliance requirements.

### Q: Can I try CrewAI Enterprise for free?

A: Yes, you can explore part of the CrewAI Enterprise Suite by accessing the [Crew Control Plane](https://app.crewai.com) for free.

### Q: Does CrewAI support fine-tuning or training custom models?

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