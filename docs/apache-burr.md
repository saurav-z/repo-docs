# Burr: Build Stateful AI Applications with Ease

**Burr is a Python library that simplifies building and managing stateful AI applications, offering a powerful framework for creating decision-making systems.**

[![Discord](https://img.shields.io/badge/Join-Burr_Discord-7289DA?logo=discord)](https://discord.gg/6Zy2DwP4f3)
[![Downloads](https://static.pepy.tech/badge/burr/month)](https://pepy.tech/project/burr)
![PyPI Downloads](https://static.pepy.tech/badge/burr)
[![GitHub Last Commit](https://img.shields.io/github/last-commit/dagworks-inc/burr)](https://github.com/dagworks-inc/burr/pulse)
[![X](https://img.shields.io/badge/follow-%40burr_framework-1DA1F2?logo=x&style=social)](https://twitter.com/burr_framework)
<a target="_blank" href="https://linkedin.com/showcase/dagworks-inc" style="background:none">
  <img src="https://img.shields.io/badge/DAGWorks-Follow-purple.svg?logo=linkedin" />
</a>
<a href="https://twitter.com/burr_framework" target="_blank">
  <img src="https://img.shields.io/badge/burr_framework-Follow-purple.svg?logo=X"/>
</a>
<a href="https://twitter.com/dagworks" target="_blank">
  <img src="https://img.shields.io/badge/DAGWorks-Follow-purple.svg?logo=X"/>
</a>

Burr allows you to develop AI applications like chatbots, agents, and simulations with ease, integrating seamlessly with your favorite frameworks and offering a user-friendly UI for monitoring and debugging.

## Key Features

*   **State Machine Architecture:** Define your application logic as a state machine, allowing for clear and manageable workflows.
*   **Real-Time UI:** Track, monitor, and trace your system's execution in real-time with the included user interface.
*   **Pluggable Persisters:** Save and load application state using various pluggable persisters for memory management.
*   **Framework Agnostic:** Burr works with any LLM or framework, offering flexibility in your development choices.
*   **Extensive Integrations:** Leverage integrations for state persistence, telemetry, and connections with other systems.

## Getting Started

1.  **Installation:** Install Burr from PyPI:

    ```bash
    pip install "burr[start]"
    ```
2.  **Run the UI:** Start the Burr telemetry UI:

    ```bash
    burr
    ```
3.  **Explore the Demo:** Use the UI's demo chat application (requires an OpenAI API key).
4.  **Run Examples:** Clone the repository and run the hello-world counter example:

    ```bash
    git clone https://github.com/dagworks-inc/burr && cd burr/examples/hello-world-counter
    python application.py
    ```
5.  **Dive Deeper:** Check out the [Getting Started Guide](https://burr.dagworks.io/getting_started/simple-example/) for more details.

## How Burr Works

Burr simplifies the creation of applications by representing your application as a state machine.  This allows you to effectively manage state, complex decisions, incorporate feedback, and create idempotent workflows.

*   **Core API:**  The core API leverages simple Python functions to build and manage state machines.
*   **UI for Introspection:** Burr includes a dependency-free python library and a UI to help you understand your application's inner workings.
*   **Integrations:** It provides out-of-the-box tools to persist state, connect to telemetry, and integrate with other systems.

[See the full documentation for more information.](https://burr.dagworks.io/)

## Applications of Burr

Burr empowers you to build various AI applications:

*   Chatbots (GPT-like, RAG-based)
*   LLM-based Adventure Games
*   Interactive Assistants (e.g., email writing)
*   Simulations (e.g., time-series forecasting)
*   Hyperparameter Tuning
*   And much more!

## Comparison with Similar Frameworks

Burr offers a unique approach. See the comparison table to better understand its strengths:

| Criteria                                          | Burr | Langgraph | temporal | Langchain | Superagent | Hamilton |
| ------------------------------------------------- | :--: | :-------: | :------: | :-------: | :--------: | :------: |
| Explicitly models a state machine                 |  ✅  |    ✅     |    ❌    |    ❌     |     ❌     |    ❌    |
| Framework-agnostic                                |  ✅  |    ✅     |    ✅    |    ✅     |     ❌     |    ✅    |
| Asynchronous event-based orchestration            |  ❌  |    ❌     |    ✅    |    ❌     |     ❌     |    ❌    |
| Built for core web-service logic                  |  ✅  |    ✅     |    ❌    |    ✅     |     ✅     |    ✅    |
| Open-source user-interface for monitoring/tracing |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |
| Works with non-LLM use-cases                      |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |

## Contributing & Community

Burr welcomes contributions!  See the [developer-facing docs](https://burr.dagworks.io/contributing) to get started.
Join the community on [Discord](https://discord.gg/6Zy2DwP4f3) for support and discussions.

## Learn More

*   [Documentation](https://burr.dagworks.io/)
*   [Quick Intro Video](https://www.loom.com/share/a10f163428b942fea55db1a84b1140d8?sid=1512863b-f533-4a42-a2f3-95b13deb07c9)
*   [Longer Video & Walkthrough](https://www.youtube.com/watch?v=rEZ4oDN0GdU)
*   [Blog Post](https://blog.dagworks.io/p/burr-develop-stateful-ai-applications)

[Back to the original repository](https://github.com/apache/burr)