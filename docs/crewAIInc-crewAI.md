<p align="center">
  <a href="https://github.com/crewAIInc/crewAI">
    <img src="docs/images/crewai_logo.png" width="600px" alt="Open source Multi-AI Agent orchestration framework">
  </a>
</p>
<p align="center">
  **Unleash the Power of Autonomous AI with CrewAI: Build Intelligent Agents that Collaborate, Execute, and Excel.**

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

##  What is CrewAI?

CrewAI is a powerful, open-source Python framework designed for orchestrating multi-agent AI systems.  Unlike other frameworks, CrewAI is **standalone, lean, and lightning-fast** – built from scratch, free from dependencies like LangChain, and focused on providing both high-level simplicity and granular control for creating sophisticated, autonomous AI applications.  [Explore the source on GitHub](https://github.com/crewAIInc/crewAI).

**Key Features:**

*   **Standalone & Lean:** Built from scratch, independent of LangChain and other agent frameworks for faster execution.
*   **Crews & Flows:**  Use intuitive Crews for agent collaboration and Flows for precise workflow control.
*   **High Performance:** Optimized for speed and minimal resource usage, enabling faster execution.
*   **Deep Customization:** Tailor workflows and agent behaviors to your exact needs.
*   **Easy Integration:** Seamlessly combine Crews (autonomy) and Flows (precision) to create complex, real-world automations.
*   **Active Community:** Backed by a rapidly growing community of over 100,000 certified developers and comprehensive documentation.
*   **Enterprise-Ready:**  Production-ready design with robust performance across simple and complex use cases.

##  Getting Started

Get up and running with CrewAI quickly!

[![CrewAI Getting Started Tutorial](https://img.youtube.com/vi/-kSOTtYzgEw/hqdefault.jpg)](https://www.youtube.com/watch?v=-kSOTtYzgEw "CrewAI Getting Started Tutorial")

### Quick Installation

```bash
pip install crewai
```
For optional tools such as local models or embeddings, use:
```bash
pip install 'crewai[tools]'
```

##  Core Concepts: Crews and Flows

CrewAI offers two primary building blocks:

*   **Crews:**  Teams of AI agents designed for autonomous collaboration and complex task completion.  Crews excel at:
    *   Dynamic task delegation and role-based teamwork.
    *   Flexible and adaptive problem-solving strategies.
    *   Real-world scenario management.
*   **Flows:**  Event-driven workflows providing precise control over the execution of complex automations. Flows support:
    *   Granular control over execution paths.
    *   Consistent state management.
    *   Conditional branching for intricate business logic.

The synergy between Crews and Flows allows you to build powerful, production-ready applications.

##  Use Cases & Examples

Explore the potential of CrewAI with these examples:

*   [Write Job Descriptions](https://github.com/crewAIInc/crewAI-examples/tree/main/crews/job-posting)
[![Jobs postings](https://img.youtube.com/vi/u98wEMz-9to/maxresdefault.jpg)](https://www.youtube.com/watch?v=u98wEMz-9to "Jobs postings")

*   [Trip Planner](https://github.com/crewAIInc/crewAI-examples/tree/main/crews/trip_planner)
[![Trip Planner](https://img.youtube.com/vi/xis7rWp-hjs/maxresdefault.jpg)](https://www.youtube.com/watch?v=xis7rWp-hjs "Trip Planner")

*   [Stock Analysis](https://github.com/crewAIInc/crewAI-examples/tree/main/crews/stock_analysis)
[![Stock Analysis](https://img.youtube.com/vi/e0Uj4yWdaAg/maxresdefault.jpg)](https://www.youtube.com/watch?v=e0Uj4yWdaAg "Stock Analysis")
*   [View More Examples](https://github.com/crewAIInc/crewAI-examples?tab=readme-ov-file)

##  CrewAI vs. the Competition

CrewAI distinguishes itself by offering a unique combination of **speed, flexibility, and control**:

*   **LangGraph:** While LangGraph offers a foundation, CrewAI avoids the complexity and limitations of LangChain dependencies, resulting in better performance.
*   **Autogen:**  Autogen's orchestration requires additional programming for agent interaction. CrewAI provides a concept of process.
*   **ChatDev:**  ChatDev has implementation limitations with rigid customizations, hindering scalability.

##  Connecting to LLMs

CrewAI supports various LLMs through different connection options, by default, the agents will use the OpenAI API to query the model. Refer to [Connect CrewAI to LLMs](https://docs.crewai.com/how-to/LLM-Connections/) for details on the configuration.

##  Enterprise Suite

For organizations seeking a secure, scalable solution, explore the [CrewAI Enterprise Suite](https://app.crewai.com).

**Key Features of the Crew Control Plane:**

*   **Tracing & Observability:** Monitor agents in real-time.
*   **Unified Control Plane:** A central platform for management.
*   **Seamless Integrations:** Connect with existing systems.
*   **Advanced Security:** Robust security measures.
*   **Actionable Insights:** Real-time analytics for performance optimization.
*   **24/7 Support:** Dedicated enterprise support.
*   **Flexible Deployment:** On-premise and cloud options.

##  Contribution

We welcome your contributions!  Please review our [Contribution Guidelines](CONTRIBUTING.md).

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

##  Telemetry

CrewAI uses anonymous telemetry to collect usage data with the main purpose of helping us improve the library by focusing our efforts on the most used features, integrations and tools.

It's pivotal to understand that **NO data is collected** concerning prompts, task descriptions, agents' backstories or goals, usage of tools, API calls, responses, any data processed by the agents, or secrets and environment variables, with the exception of the conditions mentioned. When the `share_crew` feature is enabled, detailed data including task descriptions, agents' backstories or goals, and other specific attributes are collected to provide deeper insights while respecting user privacy. Users can disable telemetry by setting the environment variable OTEL_SDK_DISABLED to true.

Data collected includes:

- Version of CrewAI
  - So we can understand how many users are using the latest version
- Version of Python
  - So we can decide on what versions to better support
- General OS (e.g. number of CPUs, macOS/Windows/Linux)
  - So we know what OS we should focus on and if we could build specific OS related features
- Number of agents and tasks in a crew
  - So we make sure we are testing internally with similar use cases and educate people on the best practices
- Crew Process being used
  - Understand where we should focus our efforts
- If Agents are using memory or allowing delegation
  - Understand if we improved the features or maybe even drop them
- If Tasks are being executed in parallel or sequentially
  - Understand if we should focus more on parallel execution
- Language model being used
  - Improved support on most used languages
- Roles of agents in a crew
  - Understand high level use cases so we can build better tools, integrations and examples about it
- Tools names available
  - Understand out of the publicly available tools, which ones are being used the most so we can improve them

Users can opt-in to Further Telemetry, sharing the complete telemetry data by setting the `share_crew` attribute to `True` on their Crews. Enabling `share_crew` results in the collection of detailed crew and task execution data, including `goal`, `backstory`, `context`, and `output` of tasks. This enables a deeper insight into usage patterns while respecting the user's choice to share.

##  License

CrewAI is released under the [MIT License](https://github.com/crewAIInc/crewAI/blob/main/LICENSE).

## Frequently Asked Questions (FAQ)

**(Note: Consider expanding the FAQ with more detailed answers based on the original README and common user questions.)**

*   **What is CrewAI?** A: CrewAI is a versatile Python framework for orchestrating autonomous AI agents.
*   **How do I install CrewAI?** A: `pip install crewai`
*   **Does CrewAI rely on LangChain?** A: No, CrewAI is independent.
*   **Can CrewAI handle complex tasks?** A: Yes, from simple to enterprise-grade scenarios.
*   **How can I contribute?** A: See the [Contribution Guidelines](CONTRIBUTING.md).
*   **Where can I find examples?** A:  See the [Examples](#examples) section.
*   **Is CrewAI open-source?** A: Yes, it's released under the MIT License.
*   **Does CrewAI collect user data?** A: Yes, but limited to improving the library. No sensitive data.

---