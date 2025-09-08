<p align="center">
  <a href="https://github.com/crewAIInc/crewAI">
    <img src="docs/images/crewai_logo.png" width="600px" alt="Open source Multi-AI Agent orchestration framework">
  </a>
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

## CrewAI: Unleash the Power of Autonomous AI Agents for Any Task!

[**Explore the CrewAI GitHub Repository**](https://github.com/crewAIInc/crewAI)

CrewAI is a Python framework that helps you quickly create and orchestrate multi-agent AI systems with superior speed and flexibility, allowing you to build intelligent automations for any use case.

**Key Features:**

*   **Standalone & Lean:** Built from scratch, independent of LangChain or other frameworks, for faster execution and lighter resource usage.
*   **Flexible & Precise:** Easily orchestrate autonomous agents through intuitive [Crews](https://docs.crewai.com/concepts/crews) or precise [Flows](https://docs.crewai.com/concepts/flows), achieving perfect balance for your needs.
*   **Seamless Integration:** Effortlessly combine Crews (autonomy) and Flows (precision) to create complex, real-world automations.
*   **Deep Customization:** Tailor every aspect—from high-level workflows down to low-level internal prompts and agent behaviors.
*   **Reliable Performance:** Consistent results across simple tasks and complex, enterprise-level automations.
*   **Thriving Community:** Backed by robust documentation and over 100,000 certified developers, providing exceptional support and guidance.

**[Get Started with CrewAI Today!](https://docs.crewai.com)**

**Why Choose CrewAI?**

*   **Speed and Efficiency:** Optimize performance with a lean, standalone framework that eliminates external dependencies, making execution significantly faster.
*   **Unmatched Flexibility:** Easily customize workflows at both high and low levels, from overall system architecture to granular agent behavior.
*   **Production-Ready:** Build robust AI automations for any scenario, from basic tasks to complex, enterprise-grade projects.
*   **Community Support:** Join a vibrant community of over 100,000 certified developers and access comprehensive documentation and support.

**Key Benefits:**

*   Accelerate development with a fast and intuitive framework.
*   Create intelligent agents with true autonomy and collaboration capabilities.
*   Achieve precise control over your AI workflows with event-driven flows.
*   Easily integrate with existing enterprise systems and data sources.
*   Build scalable solutions that meet the demands of complex projects.

**Getting Started:**

Follow these simple steps to get started with CrewAI:

### 1. Installation

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

### 2. Setting Up Your Crew with the YAML Configuration

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

### 3. Running Your Crew

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

**Examples:**

Explore the power of CrewAI with these real-world examples:

*   [Landing Page Generator](https://github.com/crewAIInc/crewAI-examples/tree/main/crews/landing_page_generator)
*   [Having Human Input on Execution](https://docs.crewai.com/how-to/Human-Input-on-Execution)
*   [Trip Planner](https://github.com/crewAIInc/crewAI-examples/tree/main/crews/trip_planner)
*   [Stock Analysis](https://github.com/crewAIInc/crewAI-examples/tree/main/crews/stock_analysis)

**Watch the Tutorials!**

*   [Quick Tutorial](https://www.youtube.com/watch?v=tnejrr-0a94):  A quick introduction to CrewAI.
*   [Write Job Descriptions](https://www.youtube.com/watch?v=u98wEMz-9to): Create job postings using AI.
*   [Trip Planner](https://www.youtube.com/watch?v=xis7rWp-hjs): Plan your next trip with AI.
*   [Stock Analysis](https://www.youtube.com/watch?v=e0Uj4yWdaAg): Analyze stocks with AI.

**Understanding Flows and Crews:**

*   **Crews:** For true agent autonomy, enabling dynamic task delegation and specialized roles.
*   **Flows:** For production-ready workflows, providing precise control and consistent state management.

**How CrewAI Compares**

*   **Advantage:** CrewAI combines autonomous agent intelligence with precise workflow control through its unique Crews and Flows architecture. The framework excels at both high-level orchestration and low-level customization, enabling complex, production-grade systems with granular control.

*   **LangGraph:** While LangGraph provides a foundation for building agent workflows, its approach requires significant boilerplate code and complex state management patterns. The framework's tight coupling with LangChain can limit flexibility when implementing custom agent behaviors or integrating with external systems.

*P.S. CrewAI demonstrates significant performance advantages over LangGraph, executing 5.76x faster in certain cases like this QA task example ([see comparison](https://github.com/crewAIInc/crewAI-examples/tree/main/Notebooks/CrewAI%20Flows%20%26%20Langgraph/QA%20Agent)) while achieving higher evaluation scores with faster completion times in certain coding tasks, like in this example ([detailed analysis](https://github.com/crewAIInc/crewAI-examples/blob/main/Notebooks/CrewAI%20Flows%20%26%20Langgraph/Coding%20Assistant/coding_assistant_eval.ipynb)).*

*   **Autogen:** While Autogen excels at creating conversational agents capable of working together, it lacks an inherent concept of process. In Autogen, orchestrating agents' interactions requires additional programming, which can become complex and cumbersome as the scale of tasks grows.
*   **ChatDev:** ChatDev introduced the idea of processes into the realm of AI agents, but its implementation is quite rigid. Customizations in ChatDev are limited and not geared towards production environments, which can hinder scalability and flexibility in real-world applications.

**Connect Your Crew to a Model**

Configure your agents to use various LLMs through different connection options. By default your agents will use the OpenAI API when querying the model. However, there are several other ways to allow your agents to connect to models. For example, you can configure your agents to use a local model via the Ollama tool.

Please refer to the [Connect CrewAI to LLMs](https://docs.crewai.com/how-to/LLM-Connections/) page for details on configuring your agents' connections to models.

**Contribution:**

*   Contribute to the project: We welcome contributions! Fork the repository, create a feature branch, and send a pull request.

**Telemetry**

*   CrewAI uses anonymous telemetry to collect usage data to improve the library. Users can disable telemetry by setting the environment variable OTEL_SDK_DISABLED to true.

**License:**

*   CrewAI is released under the [MIT License](https://github.com/crewAIInc/crewAI/blob/main/LICENSE).

**Frequently Asked Questions (FAQ):**

**General**

*   [What exactly is CrewAI?](#q-what-exactly-is-crewai)
*   [How do I install CrewAI?](#q-how-do-i-install-crewai)
*   [Does CrewAI depend on LangChain?](#q-does-crewai-depend-on-langchain)
*   [Is CrewAI open-source?](#q-is-crewai-open-source)
*   [Does CrewAI collect data from users?](#q-does-crewai-collect-data-from-users)

**Features and Capabilities**

*   [Can CrewAI handle complex use cases?](#q-can-crewai-handle-complex-use-cases)
*   [Can I use CrewAI with local AI models?](#q-can-i-use-crewai-with-local-ai-models)
*   [What makes Crews different from Flows?](#q-what-makes-crews-different-from-flows)
*   [How is CrewAI better than LangChain?](#q-how-is-crewai-better-than-langchain)
*   [Does CrewAI support fine-tuning or training custom models?](#q-does-crewai-support-fine-tuning-or-training-custom-models)

**Resources and Community**

*   [Where can I find real-world CrewAI examples?](#q-where-can-i-find-real-world-crewai-examples)
*   [How can I contribute to CrewAI?](#q-how-can-i-contribute-to-crewai)

**Enterprise Features**

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