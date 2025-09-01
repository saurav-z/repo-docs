<p align="center">
  <a href="https://github.com/crewAIInc/crewAI">
    <img src="docs/images/crewai_logo.png" width="600px" alt="Open source Multi-AI Agent orchestration framework">
  </a>
</p>

<p align="center">
  **Unleash the Power of Autonomous AI: Build Intelligent Workflows with CrewAI!**
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


[View the original repository on GitHub](https://github.com/crewAIInc/crewAI)

## Key Features

CrewAI is a cutting-edge Python framework, **independent of LangChain**, revolutionizing multi-agent automation.  It offers unmatched flexibility and performance for building intelligent workflows.

*   **Autonomous Crews:** Empower agents with true autonomy for collaborative problem-solving.
*   **Precise Flows:**  Gain granular, event-driven control for robust task orchestration.
*   **Standalone & Lean:** Built from scratch, ensuring high performance and minimal resource usage.
*   **Deep Customization:** Tailor workflows from the high-level down to the individual agent behavior.
*   **Seamless Integration:** Easily combine Crews and Flows for complex, production-ready applications.
*   **Thriving Community:** Benefit from a large and active community (100,000+ certified developers) and comprehensive documentation.

## Getting Started

Jumpstart your multi-agent automation journey with CrewAI!

[![CrewAI Getting Started Tutorial](https://img.youtube.com/vi/-kSOTtYzgEw/hqdefault.jpg)](https://www.youtube.com/watch?v=-kSOTtYzgEw "CrewAI Getting Started Tutorial")

### Installation

Install CrewAI using pip:

```bash
pip install crewai
```

For optional tools, use:

```bash
pip install 'crewai[tools]'
```

### Learning Resources

Master CrewAI with these courses:

*   [Multi AI Agent Systems with CrewAI](https://www.deeplearning.ai/short-courses/multi-ai-agent-systems-with-crewai/)
*   [Practical Multi AI Agents and Advanced Use Cases with CrewAI](https://www.deeplearning.ai/short-courses/practical-multi-ai-agents-and-advanced-use-cases-with-crewai/)

### Understanding Flows and Crews

CrewAI offers two powerful approaches that seamlessly work together:

1.  **Crews:** Teams of autonomous AI agents designed for role-based collaboration.
    *   Natural, autonomous decision-making between agents
    *   Dynamic task delegation and collaboration
    *   Specialized roles with defined goals and expertise
    *   Flexible problem-solving approaches
2.  **Flows:** Event-driven workflows for precise control over automations.
    *   Fine-grained control over execution paths for real-world scenarios
    *   Secure, consistent state management between tasks
    *   Clean integration of AI agents with production Python code
    *   Conditional branching for complex business logic

  Combining Crews and Flows allows you to:

- Build complex, production-grade applications
- Balance autonomy with precise control
- Handle sophisticated real-world scenarios
- Maintain clean, maintainable code structure

###  CLI Project Setup

```bash
crewai create crew <project_name>
cd <project_name>
crewai install (optional)
crewai run
```

  You can now start developing your crew by editing the files in the `src/my_project` folder:

-   `main.py`: The entry point of the project
-   `crew.py`: Define your crew here
-   `agents.yaml`: Define your agents configuration
-   `tasks.yaml`: Define your tasks configuration

## Examples

Explore real-world applications using CrewAI:

-   [Landing Page Generator](https://github.com/crewAIInc/crewAI-examples/tree/main/landing_page_generator)
-   [Having Human input on the execution](https://docs.crewai.com/how-to/Human-Input-on-Execution)
-   [Trip Planner](https://github.com/crewAIInc/crewAI-examples/tree/main/trip_planner)
-   [Stock Analysis](https://github.com/crewAIInc/crewAI-examples/tree/main/stock_analysis)

###  Quick Tutorial

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

## Connecting Your Crew to a Model

CrewAI supports multiple LLM connections, including OpenAI and local models.  See the [LLM Connections](https://docs.crewai.com/how-to/LLM-Connections/) documentation for full details.

## How CrewAI Compares

**CrewAI's Advantage**: CrewAI uniquely combines autonomous agent intelligence with precise workflow control using Crews and Flows, excelling in high-level orchestration and low-level customization.

-   **LangGraph:**  While valuable, requires significant boilerplate and can limit flexibility.
-   **Autogen:**  Lacks inherent process, increasing complexity in agent orchestration.
-   **ChatDev:**  Rigid structure, limiting customization and hindering scalability.

## Contribution

Contribute to CrewAI!  See the [Contribution](#contribution) section in the original README for details.

## Telemetry

CrewAI uses anonymous telemetry data to improve the library. Users can disable telemetry by setting `OTEL_SDK_DISABLED` to `true` or opt-in to further telemetry, sharing the complete telemetry data by setting the `share_crew` attribute to `True` on their Crews.

## License

CrewAI is released under the [MIT License](https://github.com/crewAIInc/crewAI/blob/main/LICENSE).

## Frequently Asked Questions (FAQ)

**(See the full list in the original README)**