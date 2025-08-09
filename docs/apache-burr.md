# Burr: Build Stateful AI Applications with Ease

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

**Burr simplifies the development of stateful AI applications by providing a flexible, easy-to-use framework for building decision-making systems.**

Burr is designed to streamline the creation of chatbots, agents, simulations, and other applications that leverage LLMs and other building blocks. It seamlessly integrates with your favorite frameworks and includes a user-friendly UI for real-time monitoring, tracing, and pluggable persisters.

**[Visit the Burr GitHub Repository](https://github.com/apache/burr)**

## Key Features

*   **State Machine-Based Design:** Express your application logic as a state machine for clear, manageable workflows.
*   **Simple Python API:** Build and manage state machines using straightforward Python functions.
*   **Real-time UI:** Track, monitor, and trace your system's execution with a built-in UI.
*   **Pluggable Persisters:** Easily save and load application state for persistence.
*   **Framework Agnostic:** Burr works well with any LLM or Python Framework, and easily integrates into existing projects.
*   **Versatile Use Cases:** From chatbots and RAG applications to simulations and interactive assistants, Burr adapts to a wide range of AI applications.

## Quick Start

Get up and running quickly with Burr:

1.  **Install:** `pip install "burr[start]"`
2.  **Run UI:** `burr` (This opens the telemetry UI).
3.  **Run Example:** `git clone https://github.com/dagworks-inc/burr && cd burr/examples/hello-world-counter && python application.py`

For detailed instructions, refer to the [getting started guide](https://burr.dagworks.io/getting_started/simple-example/).

## How Burr Works

Burr's core API allows you to build state machines by defining actions and transitions. See the example in the original README.

Burr provides a dependency-free Python library, an execution UI for introspection and debugging, and several integrations for persistence, telemetry, and other integrations.

## Use Cases

Burr can be applied to a variety of applications, including but not limited to:

1.  Chatbots
2.  RAG-based applications
3.  LLM-powered games
4.  Interactive assistants (email writers)
5.  Simulations
6.  Hyperparameter tuning

Burr allows you to integrate with your favorite vendors (LLM observability, storage, etc...) and build custom actions using your favorite libraries.

## Comparisons

| Criteria                                          | Burr | Langgraph | temporal | Langchain | Superagent | Hamilton |
| ------------------------------------------------- | :--: | :-------: | :------: | :-------: | :--------: | :------: |
| Explicitly models a state machine                 |  ✅  |    ✅     |    ❌    |    ❌     |     ❌     |    ❌    |
| Framework-agnostic                                |  ✅  |    ✅     |    ✅    |    ✅     |     ❌     |    ✅    |
| Asynchronous event-based orchestration            |  ❌  |    ❌     |    ✅    |    ❌     |     ❌     |    ❌    |
| Built for core web-service logic                  |  ✅  |    ✅     |    ❌    |    ✅     |     ✅     |    ✅    |
| Open-source user-interface for monitoring/tracing |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |
| Works with non-LLM use-cases                      |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |

## Why the name Burr?

Burr is named after Aaron Burr, the founder and third VP of the United States. This is the second open-source library from DAGWorks after Hamilton library.

## Testimonials

*(Please add the testimonials from the original README here)*

## Roadmap

Burr's roadmap includes:

1.  FastAPI integration + hosted deployment
2.  Improvements to the core library (retry, exception management, integrations, metadata)
3.  Tooling for hosted execution (Ray, modal, etc.)
4.  Additional storage integrations

## Contributing

Contributions are welcome! See the [developer documentation](https://burr.dagworks.io/contributing) to get started.

## Contributors

*(Please add the contributors from the original README here)*