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

# CrewAI: Build Autonomous AI Agents with Unmatched Speed, Flexibility, and Control

**[CrewAI](https://github.com/crewAIInc/crewAI) empowers developers to rapidly build and deploy intelligent, autonomous AI agents tailored for any task, from simple automations to complex enterprise solutions.**

## Key Features

*   ✨ **Standalone & Lean**:  Independent of frameworks like LangChain, providing faster execution and lighter resource demands.
*   ⚙️ **Flexible & Precise**: Easily orchestrate autonomous agents with intuitive [Crews](https://docs.crewai.com/concepts/crews) or precise [Flows](https://docs.crewai.com/concepts/flows).
*   🤝 **Seamless Integration**: Combine Crews (autonomy) and Flows (precision) for complex real-world automation.
*   🛠️ **Deep Customization**: Tailor every aspect, from workflows to low-level agent behaviors and prompts.
*   🚀 **Reliable Performance**: Consistent results across simple tasks and complex, enterprise-level automations.
*   👨‍💻 **Thriving Community**: Backed by robust documentation and over 100,000 certified developers, providing exceptional support.

## Core Concepts

*   **Crews**: Teams of AI agents that work together, with true autonomy and agency, to accomplish complex tasks through role-based collaboration.
*   **Flows**: Production-ready, event-driven workflows that deliver precise control over complex automations.

## Why CrewAI?

CrewAI offers a unique combination of speed, flexibility, and control:

*   **Standalone Framework**: Built from scratch, independent of LangChain or any other agent framework.
*   **High Performance**: Optimized for speed and minimal resource usage, enabling faster execution.
*   **Flexible Low Level Customization**: Complete freedom to customize at both high and low levels - from overall workflows and system architecture to granular agent behaviors, internal prompts, and execution logic.
*   **Ideal for Every Use Case**: Proven effective for both simple tasks and highly complex, real-world, enterprise-grade scenarios.
*   **Robust Community**: Backed by a rapidly growing community of over **100,000 certified** developers offering comprehensive support and resources.

## Getting Started

1.  **Installation:**

    ```bash
    pip install crewai
    ```
    For optional tools:
    ```bash
    pip install 'crewai[tools]'
    ```
2.  **Create a new project:**
    ```bash
    crewai create crew <project_name>
    ```

    This will create a project with the basic files already set up:
    *   `my_project/`
        *   `.gitignore`
        *   `pyproject.toml`
        *   `README.md`
        *   `.env`
        *   `src/`
            *   `my_project/`
                *   `__init__.py`
                *   `main.py`
                *   `crew.py`
                *   `tools/`
                    *   `custom_tool.py`
                    *   `__init__.py`
                *   `config/`
                    *   `agents.yaml`
                    *   `tasks.yaml`

3.  **Run your crew:**

    ```bash
    cd my_project
    crewai run
    ```
    or
    ```bash
    python src/my_project/main.py
    ```

    Make sure to set up your `.env` file as described below.

###  Key setup requirements

*   **Python:** Make sure you have python version >=3.10 <3.14 installed on your system.
*   **.env file:** Set these variables with your preferred values.
    *   `OPENAI_API_KEY`
    *   `SERPER_API_KEY`

### Learn More

*   [Multi AI Agent Systems with CrewAI](https://www.deeplearning.ai/short-courses/multi-ai-agent-systems-with-crewai/)
*   [Practical Multi AI Agents and Advanced Use Cases](https://www.deeplearning.ai/short-courses/practical-multi-ai-agents-and-advanced-use-cases-with-crewai/)

## Examples

Explore real-world CrewAI applications:

*   [Landing Page Generator](https://github.com/crewAIInc/crewAI-examples/tree/main/crews/landing_page_generator)
*   [Having Human input on the execution](https://docs.crewai.com/how-to/Human-Input-on-Execution)
*   [Trip Planner](https://github.com/crewAIInc/crewAI-examples/tree/main/crews/trip_planner)
*   [Stock Analysis](https://github.com/crewAIInc/crewAI-examples/tree/main/crews/stock_analysis)

###  Quick Tutorials
[![CrewAI Tutorial](https://img.youtube.com/vi/tnejrr-0a94/maxresdefault.jpg)](https://www.youtube.com/watch?v=tnejrr-0a94 "CrewAI Tutorial")

### Write Job Descriptions
[Check out code for this example](https://github.com/crewAIInc/crewAI-examples/tree/main/crews/job-posting) or watch a video below:

[![Jobs postings](https://img.youtube.com/vi/u98wEMz-9to/maxresdefault.jpg)](https://www.youtube.com/watch?v=u98wEMz-9to "Jobs postings")

### Trip Planner
[Check out code for this example](https://github.com/crewAIInc/crewAI-examples/tree/main/crews/trip_planner) or watch a video below:

[![Trip Planner](https://img.youtube.com/vi/xis7rWp-hjs/maxresdefault.jpg)](https://www.youtube.com/watch?v=xis7rWp-hjs "Trip Planner")

### Stock Analysis
[Check out code for this example](https://github.com/crewAIInc/crewAI-examples/tree/main/crews/stock_analysis) or watch a video below:

[![Stock Analysis](https://img.youtube.com/vi/e0Uj4yWdaAg/maxresdefault.jpg)](https://www.youtube.com/watch?v=e0Uj4yWdaAg "Stock Analysis")

### Combining Crews and Flows

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

CrewAI supports using various LLMs through a variety of connection options. By default your agents will use the OpenAI API when querying the model. However, there are several other ways to allow your agents to connect to models. For example, you can configure your agents to use a local model via the Ollama tool.

Please refer to the [Connect CrewAI to LLMs](https://docs.crewai.com/how-to/LLM-Connections/) page for details on configuring your agents' connections to models.

## How CrewAI Compares

**CrewAI's Advantage**: CrewAI combines autonomous agent intelligence with precise workflow control through its unique Crews and Flows architecture. The framework excels at both high-level orchestration and low-level customization, enabling complex, production-grade systems with granular control.

-   **LangGraph**: While LangGraph provides a foundation for building agent workflows, its approach requires significant boilerplate code and complex state management patterns. The framework's tight coupling with LangChain can limit flexibility when implementing custom agent behaviors or integrating with external systems.

    *P.S. CrewAI demonstrates significant performance advantages over LangGraph, executing 5.76x faster in certain cases like this QA task example ([see comparison](https://github.com/crewAIInc/crewAI-examples/tree/main/Notebooks/CrewAI%20Flows%20%26%20Langgraph/QA%20Agent)) while achieving higher evaluation scores with faster completion times in certain coding tasks, like in this example ([detailed analysis](https://github.com/crewAIInc/crewAI-examples/blob/main/Notebooks/CrewAI%20Flows%20%26%20Langgraph/Coding%20Assistant/coding_assistant_eval.ipynb)).*

-   **Autogen**: While Autogen excels at creating conversational agents capable of working together, it lacks an inherent concept of process. In Autogen, orchestrating agents' interactions requires additional programming, which can become complex and cumbersome as the scale of tasks grows.
-   **ChatDev**: ChatDev introduced the idea of processes into the realm of AI agents, but its implementation is quite rigid. Customizations in ChatDev are limited and not geared towards production environments, which can hinder scalability and flexibility in real-world applications.

## Contribution

CrewAI is open-source and welcomes contributions.

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

CrewAI uses anonymous telemetry to collect usage data with the main purpose of helping us improve the library by focusing our efforts on the most used features, integrations and tools.

It's pivotal to understand that **NO data is collected** concerning prompts, task descriptions, agents' backstories or goals, usage of tools, API calls, responses, any data processed by the agents, or secrets and environment variables, with the exception of the conditions mentioned. When the `share_crew` feature is enabled, detailed data including task descriptions, agents' backstories or goals, and other specific attributes are collected to provide deeper insights while respecting user privacy. Users can disable telemetry by setting the environment variable OTEL_SDK_DISABLED to true.

Data collected includes:

-   Version of CrewAI
    -   So we can understand how many users are using the latest version
-   Version of Python
    -   So we can decide on what versions to better support
-   General OS (e.g. number of CPUs, macOS/Windows/Linux)
    -   So we know what OS we should focus on and if we could build specific OS related features
-   Number of agents and tasks in a crew
    -   So we make sure we are testing internally with similar use cases and educate people on the best practices
-   Crew Process being used
    -   Understand where we should focus our efforts
-   If Agents are using memory or allowing delegation
    -   Understand if we improved the features or maybe even drop them
-   If Tasks are being executed in parallel or sequentially
    -   Understand if we should focus more on parallel execution
-   Language model being used
    -   Improved support on most used languages
-   Roles of agents in a crew
    -   Understand high level use cases so we can build better tools, integrations and examples about it
-   Tools names available
    -   Understand out of the publicly available tools, which ones are being used the most so we can improve them

Users can opt-in to Further Telemetry, sharing the complete telemetry data by setting the `share_crew` attribute to `True` on their Crews. Enabling `share_crew` results in the collection of detailed crew and task execution data, including `goal`, `backstory`, `context`, and `output` of tasks. This enables a deeper insight into usage patterns while respecting the user's choice to share.

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

A: CrewAI is a standalone, lean, and fast Python framework for building and deploying autonomous AI agents.

### Q: How do I install CrewAI?

A: Install CrewAI using pip:
```bash
pip install crewai
```

### Q: Does CrewAI depend on LangChain?

A: No. CrewAI is built entirely from the ground up, with no dependencies on LangChain.

### Q: Can CrewAI handle complex use cases?

A: Yes, CrewAI excels at complex real-world scenarios, with deep customization.

### Q: Can I use CrewAI with local AI models?

A: Yes.  Integrate local models using tools like Ollama and LM Studio.

### Q: What makes Crews different from Flows?

A: Crews are for autonomous agent collaboration; Flows provide precise, event-driven control.

### Q: How is CrewAI better than LangChain?

A: CrewAI offers simpler APIs, faster execution, and robust documentation.

### Q: Is CrewAI open-source?

A: Yes, CrewAI is open-source and encourages community contributions.

### Q: Does CrewAI collect data from users?

A: CrewAI collects anonymous telemetry for improvement. Sensitive data is *not* collected unless explicitly enabled.

### Q: Where can I find real-world CrewAI examples?

A: Check out practical examples in the [CrewAI-examples repository](https://github.com/crewAIInc/crewAI-examples).

### Q: How can I contribute to CrewAI?

A: Fork the repository, create your branch, implement changes, and submit a pull request.

### Q: What additional features does CrewAI Enterprise offer?

A:  CrewAI Enterprise provides advanced features.

### Q: Is CrewAI Enterprise available for cloud and on-premise deployments?

A:  Yes, supports both cloud-based and on-premise deployment.

### Q: Can I try CrewAI Enterprise for free?

A:  Yes, by accessing the [Crew Control Plane](https://app.crewai.com).

### Q: Does CrewAI support fine-tuning or training custom models?

A: Yes, integrates with custom-trained models.