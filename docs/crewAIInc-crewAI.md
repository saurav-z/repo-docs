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

## Supercharge Your AI Automation with CrewAI: The Lightning-Fast, Standalone Framework

CrewAI is a powerful, open-source Python framework designed for orchestrating multi-agent AI systems, offering unparalleled flexibility and control.  [Explore the source code on GitHub](https://github.com/crewAIInc/crewAI).

### Key Features

*   **Standalone & Lean:** Built from scratch, independent of LangChain and other agent frameworks, resulting in faster execution and less resource consumption.
*   **Crews for Autonomy:** Easily create teams of autonomous AI agents that collaborate to solve complex problems.
*   **Flows for Precision:** Implement event-driven workflows for granular control over tasks and precise orchestration.
*   **Deep Customization:** Tailor every aspect of your AI agents, from workflows and system architecture to agent behaviors and execution logic.
*   **Production-Ready Performance:** Achieve consistent and reliable results across both simple and complex enterprise-grade scenarios.
*   **Thriving Community:** Benefit from extensive documentation and a rapidly growing community of over 100,000 certified developers offering comprehensive support and resources.

### Why Choose CrewAI?

CrewAI provides the best-in-class combination of speed, flexibility, and control for multi-agent automation:

*   **Standalone Framework:** No reliance on external agent frameworks.
*   **High Performance:** Optimized for speed and minimal resource usage.
*   **Flexible Customization:** Complete control over workflows, architecture, and agent behavior.
*   **Ideal for All Use Cases:** Proven effective for simple to complex enterprise-grade scenarios.
*   **Robust Community:** Backed by a large, supportive community.

### Getting Started

#### Installation

Make sure you have Python >=3.10 <3.14 installed on your system. CrewAI uses [UV](https://docs.astral.sh/uv/) for dependency management and package handling, offering a seamless setup and execution experience.

```bash
pip install crewai
```

To install with optional tools:

```bash
pip install 'crewai[tools]'
```

#### Troubleshooting

1.  **ModuleNotFoundError: No module named 'tiktoken'**: Install with `pip install 'crewai[embeddings]'` or  `pip install 'crewai[tools]'` if using embedchain.
2.  **Failed building wheel for tiktoken**: Ensure Rust and Visual C++ Build Tools are installed. Try upgrading pip or using a pre-built wheel: `pip install tiktoken --prefer-binary`.

#### Example: Create a new CrewAI project

```bash
crewai create crew <project_name>
```

Edit the files in the `src/my_project` folder to customize your agents and tasks.

#### Run a simple crew

1.  Set API keys as environment variables in your `.env` file: `OPENAI_API_KEY=sk-...`, `SERPER_API_KEY=YOUR_KEY_HERE`.
2.  Navigate to your project directory: `cd my_project`
3.  Install and run:

```bash
crewai install  # Optional
```

```bash
crewai run
```

or

```bash
python src/my_project/main.py
```

If you get an error due to poetry, run the following command:

```bash
crewai update
```

This will produce your report.md in the root of your project.

### Learning Resources

*   [Multi AI Agent Systems with CrewAI](https://www.deeplearning.ai/short-courses/multi-ai-agent-systems-with-crewai/)
*   [Practical Multi AI Agents and Advanced Use Cases with CrewAI](https://www.deeplearning.ai/short-courses/practical-multi-ai-agents-and-advanced-use-cases-with-crewai/)

### Understanding Flows and Crews

CrewAI offers two complementary approaches:

1.  **Crews:** Teams of autonomous AI agents working together through role-based collaboration.
    *   Natural, autonomous decision-making
    *   Dynamic task delegation
    *   Specialized roles
    *   Flexible problem-solving
2.  **Flows:** Event-driven workflows for precise control.
    *   Fine-grained control
    *   Secure state management
    *   Integration with production code
    *   Conditional branching

#### Example combining Crews and Flows
```python
# Code for combining Crews and Flows (example)
# ... (See original README for full example code)
```

### Key Features

*   **Standalone & Lean**: Independent, offering faster execution and less resource consumption.
*   **Crews for Autonomy**: Easily create teams of autonomous AI agents that collaborate to solve complex problems.
*   **Flows for Precision**: Implement event-driven workflows for granular control over tasks and precise orchestration.
*   **Deep Customization**: Tailor every aspect of your AI agents, from workflows and system architecture to agent behaviors and execution logic.
*   **Production-Ready Performance**: Achieve consistent and reliable results across both simple and complex enterprise-grade scenarios.
*   **Thriving Community**: Benefit from extensive documentation and a rapidly growing community of over 100,000 certified developers offering comprehensive support and resources.

### Examples

*   [Landing Page Generator](https://github.com/crewAIInc/crewAI-examples/tree/main/landing_page_generator)
*   [Having Human input on the execution](https://docs.crewai.com/how-to/Human-Input-on-Execution)
*   [Trip Planner](https://github.com/crewAIInc/crewAI-examples/tree/main/trip_planner)
*   [Stock Analysis](https://github.com/crewAIInc/crewAI-examples/tree/main/stock_analysis)

### Tutorials

*   [CrewAI Tutorial](https://www.youtube.com/watch?v=tnejrr-0a94 "CrewAI Tutorial")
*   [Jobs postings](https://www.youtube.com/watch?v=u98wEMz-9to "Jobs postings")
*   [Trip Planner](https://www.youtube.com/watch?v=xis7rWp-hjs "Trip Planner")
*   [Stock Analysis](https://www.youtube.com/watch?v=e0Uj4yWdaAg "Stock Analysis")

### Connecting Your Crew to a Model

CrewAI supports various LLMs through connection options.  See [Connect CrewAI to LLMs](https://docs.crewai.com/how-to/LLM-Connections/) for details.

### How CrewAI Compares

CrewAI's advantage is its unique Crews and Flows architecture, excelling at both high-level orchestration and low-level customization.

*   **LangGraph:** CrewAI has significant performance advantages over LangGraph ([see comparison](https://github.com/crewAIInc/crewAI-examples/tree/main/Notebooks/CrewAI%20Flows%20%26%20Langgraph/QA%20Agent))
*   **Autogen:** Lacks an inherent concept of process.
*   **ChatDev:** Limited customizations and not geared towards production environments.

### Contribution

Open-source contributions are welcome!

1.  Fork the repository.
2.  Create a new branch.
3.  Add your feature or improvement.
4.  Send a pull request.

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

CrewAI uses anonymous telemetry to collect usage data to improve the library.

*   No sensitive data (prompts, tasks, etc.) is collected unless explicitly enabled.
*   Users can disable telemetry with `OTEL_SDK_DISABLED=true`.
*   `share_crew=True` enables further data collection.

### License

Released under the [MIT License](https://github.com/crewAIInc/crewAI/blob/main/LICENSE).

### Frequently Asked Questions (FAQ)

**General**

*   [What is CrewAI?](#q-what-exactly-is-crewai)
*   [How to install?](#q-how-do-i-install-crewai)
*   [Does it depend on LangChain?](#q-does-crewai-depend-on-langchain)
*   [Is it open-source?](#q-is-crewai-open-source)
*   [Does it collect data?](#q-does-crewai-collect-data-from-users)

**Features and Capabilities**

*   [Handle complex use cases?](#q-can-crewai-handle-complex-use-cases)
*   [Use local AI models?](#q-can-i-use-crewai-with-local-ai-models)
*   [Difference between Crews and Flows?](#q-what-makes-crews-different-from-flows)
*   [How is it better than LangChain?](#q-how-is-crewai-better-than-langchain)
*   [Support fine-tuning or custom models?](#q-does-crewai-support-fine-tuning-or-training-custom-models)

**Resources and Community**

*   [Where to find real-world examples?](#q-where-can-i-find-real-world-crewai-examples)
*   [How to contribute?](#q-how-can-i-contribute-to-crewai)

**Enterprise Features**

*   [What are the additional Enterprise features?](#q-what-additional-features-does-crewai-enterprise-offer)
*   [Is it for cloud and on-premise?](#q-is-crewai-enterprise-available-for-cloud-and-on-premise-deployments)
*   [Can you try CrewAI Enterprise for free?](#q-can-i-try-crewai-enterprise-for-free)

**Q: What exactly is CrewAI?**
A: A framework for orchestrating autonomous AI agents.

**Q: How to install?**
A: Use pip: `pip install crewai`

**Q: Does it depend on LangChain?**
A: No, it's built from the ground up.

**Q: Is it open-source?**
A: Yes, and contributions are welcome.

**Q: Does it collect data?**
A: Anonymous telemetry for improvement (sensitive data is not collected).

**Q: Handle complex use cases?**
A: Yes, from simple to complex, enterprise-grade scenarios.

**Q: Use local AI models?**
A: Yes, through tools like Ollama.

**Q: Difference between Crews and Flows?**
A: Crews for collaboration, Flows for precise control.

**Q: How is it better than LangChain?**
A: Simpler APIs, faster execution, more reliable results.

**Q: Support fine-tuning or custom models?**
A: Yes.

**Q: Where to find real-world examples?**
A: In the [CrewAI-examples repository](https://github.com/crewAIInc/crewAI-examples).

**Q: How to contribute?**
A: Fork, branch, contribute, and submit a pull request.

**Q: What are the additional Enterprise features?**
A: Control plane, observability, secure integrations, and more.

**Q: Is it for cloud and on-premise?**
A: Yes.

**Q: Can you try CrewAI Enterprise for free?**
A: Access the [Crew Control Plane](https://app.crewai.com) for free.