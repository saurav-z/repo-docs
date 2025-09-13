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

## CrewAI: Build Powerful, Autonomous AI Agent Workflows with Ease

CrewAI is a cutting-edge, open-source Python framework, designed to empower developers to build sophisticated multi-agent systems that are both powerful and easy to use.  [Explore the power of CrewAI on GitHub!](https://github.com/crewAIInc/crewAI)

**Key Features:**

*   **Standalone & Lean**: Built from scratch, independent of frameworks like LangChain, ensuring faster execution and lower resource usage.
*   **Flexible Orchestration**: Easily create autonomous agents using intuitive [Crews](https://docs.crewai.com/concepts/crews) for collaboration or precise [Flows](https://docs.crewai.com/concepts/flows) for event-driven control.
*   **Seamless Integration**: Combine Crews and Flows to build complex, real-world automations.
*   **Deep Customization**: Fine-tune every aspect of your agents and workflows, from high-level design down to internal prompts and agent behaviors.
*   **High Performance**: Experience consistent and reliable results for both simple tasks and complex enterprise-grade scenarios.
*   **Community-Driven**: Benefit from a thriving community and comprehensive resources, including over **100,000 certified developers**.

## Table of Contents

*   [Why CrewAI?](#why-crewai)
*   [Getting Started](#getting-started)
    *   [Installation](#installation)
    *   [Project Setup](#project-setup)
    *   [Running Your Crew](#running-your-crew)
*   [Understanding Flows and Crews](#understanding-flows-and-crews)
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
    *   [General](#general)
    *   [Features and Capabilities](#features-and-capabilities)
    *   [Resources and Community](#resources-and-community)
    *   [Enterprise Features](#enterprise-features)

## Why CrewAI?

<div align="center" style="margin-bottom: 30px;">
  <img src="docs/images/asset.png" alt="CrewAI Logo" width="100%">
</div>

CrewAI offers a streamlined approach to multi-agent automation, combining speed, flexibility, and control for optimal results:

*   **Standalone Framework:** Operates independently, eliminating dependencies on other agent frameworks.
*   **High Performance:** Designed for speed and minimal resource utilization, facilitating rapid execution.
*   **Customization Options:** Offers both high-level and granular control over every aspect of workflows.
*   **Adaptable:** Well-suited for a wide range of tasks, from basic to advanced enterprise applications.
*   **Robust Community:** Benefit from a large and growing community that offers extensive support and valuable resources.

## Getting Started

Follow these steps to begin using CrewAI:

### Installation

Ensure you have Python >=3.10 <3.14 installed. CrewAI uses [UV](https://docs.astral.sh/uv/) for dependency management.

1.  Install CrewAI:

    ```shell
    pip install crewai
    ```

2.  Install optional tools:

    ```shell
    pip install 'crewai[tools]'
    ```

    **Troubleshooting:**

    *   **ModuleNotFoundError: No module named 'tiktoken'**:  Install `tiktoken` explicitly: `pip install 'crewai[embeddings]'` or `pip install 'crewai[tools]'`.
    *   **Failed building wheel for tiktoken**: Ensure Rust compiler is installed, or use a pre-built wheel: `pip install tiktoken --prefer-binary`.
### Project Setup

Create and set up your first project using the CrewAI CLI:

```shell
crewai create crew <project_name>
```

This generates a project structure, including:

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
### Running Your Crew

1.  **Configure Environment Variables**:  Set your API keys (e.g., OpenAI, Serper.dev) in a `.env` file:

    ```
    OPENAI_API_KEY=sk-...
    SERPER_API_KEY=YOUR_KEY_HERE
    ```

2.  **Install Dependencies**: From your project directory:
    ```bash
    cd my_project
    crewai install (Optional)
    ```
3.  **Run Your Crew**:
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
    You should see output in the console, and a report file (e.g., `report.md`) will be generated.

## Understanding Flows and Crews

CrewAI provides two powerful approaches:

1.  **Crews**: Groups of autonomous agents working collaboratively through role-based interaction, facilitating:
    *   Autonomous decision-making
    *   Dynamic task delegation
    *   Specialized roles and expertise
    *   Flexible problem-solving

2.  **Flows**: Event-driven workflows that provide precise control over automations, offering:
    *   Fine-grained control over execution paths
    *   Secure state management
    *   Integration with Python code
    *   Conditional branching

Combining Crews and Flows allows you to build robust, production-ready applications with both flexibility and control.

## Examples

Explore real-world applications in the [CrewAI-examples repo](https://github.com/crewAIInc/crewAI-examples?tab=readme-ov-file):

*   [Landing Page Generator](https://github.com/crewAIInc/crewAI-examples/tree/main/crews/landing_page_generator)
*   [Having Human input on the execution](https://docs.crewai.com/how-to/Human-Input-on-Execution)
*   [Trip Planner](https://github.com/crewAIInc/crewAI-examples/tree/main/crews/trip_planner)
*   [Stock Analysis](https://github.com/crewAIInc/crewAI-examples/tree/main/crews/stock_analysis)

### Quick Tutorial

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

### Using Crews and Flows Together

Combine Crews with Flows for sophisticated automation pipelines.  Use `or_` and `and_` logical operators within Flows.

```python
from crewai.flow.flow import Flow, listen, start, router, or_
from crewai import Crew, Agent, Task, Process
from pydantic import BaseModel

# Define structured state
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

CrewAI supports using various LLMs.  By default, agents use the OpenAI API. For alternative model connections, see [Connect CrewAI to LLMs](https://docs.crewai.com/how-to/LLM-Connections/).

## How CrewAI Compares

**CrewAI's Edge**: CrewAI's architecture uniquely combines autonomous agent intelligence with precise workflow control via Crews and Flows. It excels at both high-level orchestration and low-level customization.

*   **LangGraph**: Requires significant boilerplate code and complex state management.  Coupling with LangChain can limit flexibility.

    *P.S. CrewAI shows substantial performance gains over LangGraph, executing 5.76x faster in certain cases ([see comparison](https://github.com/crewAIInc/crewAI-examples/tree/main/Notebooks/CrewAI%20Flows%20%26%20Langgraph/QA%20Agent)) and scoring higher in coding tasks ([detailed analysis](https://github.com/crewAIInc/crewAI-examples/blob/main/Notebooks/CrewAI%20Flows%20%26%20Langgraph/Coding%20Assistant/coding_assistant_eval.ipynb)).*
*   **Autogen**: Lacks inherent process concepts; orchestrating agent interactions requires extra programming.
*   **ChatDev**: Limited customization, not production-ready, restricting scalability and flexibility.

## Contribution

Contributions are welcome.  Follow these steps:

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

CrewAI uses anonymous telemetry to help improve the library.

*   **Data Collected**: Version, Python version, OS, agent/task counts, crew process, memory/delegation usage, model used, agent roles, tool names.
*   **Opt-in to Further Telemetry**: Enable by setting `share_crew=True`, which then collects detailed data like task descriptions and outputs.
*   **Disable Telemetry**: Set the environment variable `OTEL_SDK_DISABLED` to `true`.

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

### General

**Q: What exactly is CrewAI?**

A: CrewAI is a standalone, lean, and fast Python framework built specifically for orchestrating autonomous AI agents. Unlike frameworks like LangChain, CrewAI does not rely on external dependencies, making it leaner, faster, and simpler.

**Q: How do I install CrewAI?**

A: Install CrewAI using pip:

```shell
pip install crewai
```

For additional tools, use:

```shell
pip install 'crewai[tools]'
```

**Q: Does CrewAI depend on LangChain?**

A: No. CrewAI is built entirely from the ground up, with no dependencies on LangChain or other agent frameworks. This ensures a lean, fast, and flexible experience.

**Q: Is CrewAI open-source?**

A: Yes, CrewAI is open-source and actively encourages community contributions and collaboration.

**Q: Does CrewAI collect data from users?**

A: CrewAI collects anonymous telemetry data strictly for improvement purposes. Sensitive data such as prompts, tasks, or API responses are never collected unless explicitly enabled by the user.

### Features and Capabilities

**Q: Can CrewAI handle complex use cases?**

A: Yes. CrewAI excels at both simple and highly complex real-world scenarios, offering deep customization options at both high and low levels, from internal prompts to sophisticated workflow orchestration.

**Q: Can I use CrewAI with local AI models?**

A: Absolutely! CrewAI supports various language models, including local ones. Tools like Ollama and LM Studio allow seamless integration. Check the [LLM Connections documentation](https://docs.crewai.com/how-to/LLM-Connections/) for more details.

**Q: What makes Crews different from Flows?**

A: Crews provide autonomous agent collaboration, ideal for tasks requiring flexible decision-making and dynamic interaction. Flows offer precise, event-driven control, ideal for managing detailed execution paths and secure state management. You can seamlessly combine both for maximum effectiveness.

**Q: How is CrewAI better than LangChain?**

A: CrewAI provides simpler, more intuitive APIs, faster execution speeds, more reliable and consistent results, robust documentation, and an active community—addressing common criticisms and limitations associated with LangChain.

**Q: Does CrewAI support fine-tuning or training custom models?**

A: Yes, CrewAI can integrate with custom-trained or fine-tuned models, allowing you to enhance your agents with domain-specific knowledge and accuracy.

### Resources and Community

**Q: Where can I find real-world CrewAI examples?**

A: Check out practical examples in the [CrewAI-examples repository](https://github.com/crewAIInc/crewAI-examples), covering use cases like trip planners, stock analysis, and job postings.

**Q: How can I contribute to CrewAI?**

A: Contributions are warmly welcomed! Fork the repository, create your branch, implement your changes, and submit a pull request. See the Contribution section of the README for detailed guidelines.

### Enterprise Features

**Q: What additional features does CrewAI Enterprise offer?**

A: CrewAI Enterprise provides advanced features such as a unified control plane, real-time observability, secure integrations, advanced security, actionable insights, and dedicated 24/7 enterprise support.

**Q: Is CrewAI Enterprise available for cloud and on-premise deployments?**

A: Yes, CrewAI Enterprise supports both cloud-based and on-premise deployment options, allowing enterprises to meet their specific security and compliance requirements.

**Q: Can I try CrewAI Enterprise for free?**

A: Yes, you can explore part of the CrewAI Enterprise Suite by accessing the [Crew Control Plane](https://app.crewai.com) for free.