# Burr: Build and Monitor Stateful AI Applications

Burr simplifies the development of stateful AI applications like chatbots, agents, and simulations with its intuitive Python building blocks and real-time UI.

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

**Key Features:**

*   **State Machine Framework:**  Express your applications as easy-to-understand state machines using simple Python functions.
*   **Real-Time UI:** Monitor, trace, and debug your application's execution with a user-friendly interface.
*   **Pluggable Persistence:** Save and load application state with various integration options.
*   **LLM-Agnostic:** Easily integrate with your favorite LLMs and other AI frameworks.
*   **Flexible Use Cases:** Supports diverse applications, including chatbots, RAG systems, simulations, and more.
*   **Open Source:** Benefit from a community-driven project with clear documentation and a welcoming discord server.

## Getting Started

### Installation

Install Burr from PyPI:

```bash
pip install "burr[start]"
```

(See [the docs](https://burr.dagworks.io/getting_started/install/) if you're using poetry)

### Run the UI

Run the UI server:

```bash
burr
```

The UI comes loaded with some default data and a demo chatbot to show real-time changes.

### Example

Clone the repository and run a simple example:

```bash
git clone https://github.com/dagworks-inc/burr && cd burr/examples/hello-world-counter
python application.py
```

Find the counter trace in the UI.

For more details, see the [getting started guide](https://burr.dagworks.io/getting_started/simple-example/).

## How Burr Works

Burr allows you to manage state, track decisions, and define idempotent workflows.

Burr consists of:

1.  A Python library for building and managing state machines.
2.  A UI to view execution telemetry for debugging.
3.  Integrations to persist state, connect to telemetry, and integrate with other systems.

![Burr at work](https://github.com/DAGWorks-Inc/burr/blob/main/chatbot.gif)

## Use Cases

Burr can power:

*   Chatbots (simple and RAG-based)
*   LLM-based adventure games
*   Interactive assistants
*   Simulations
*   Hyperparameter tuning and other (non-LLM) applications

Integrate with vendors and delegate to your favorite libraries (e.g. [Hamilton](https://github.com/DAGWorks-Inc/hamilton))

Burr helps you connect your system's logic in a way that scales.

## Comparison

Burr compared to other frameworks:

| Criteria                                          | Burr | Langgraph | temporal | Langchain | Superagent | Hamilton |
| ------------------------------------------------- | :--: | :-------: | :------: | :-------: | :--------: | :------: |
| Explicitly models a state machine                 |  ✅  |    ✅     |    ❌    |    ❌     |     ❌     |    ❌    |
| Framework-agnostic                                |  ✅  |    ✅     |    ✅    |    ✅     |     ❌     |    ✅    |
| Asynchronous event-based orchestration            |  ❌  |    ❌     |    ✅    |    ❌     |     ❌     |    ❌    |
| Built for core web-service logic                  |  ✅  |    ✅     |    ❌    |    ✅     |     ✅     |    ✅    |
| Open-source user-interface for monitoring/tracing |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |
| Works with non-LLM use-cases                      |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |

## Why Burr?

Burr is named after Aaron Burr, reflecting the second open-source library release from DAGWorks (after Hamilton). Burr focuses on harmonious state management, designed to simplify the development of your AI applications.

## Testimonials

See what others are saying about Burr.

## Roadmap

Future features:

1.  FastAPI integration + hosted deployment.
2.  Efficiency and usability improvements.
3.  Tooling for hosted execution.
4.  Additional storage integrations.

Sign up for the Burr Cloud waitlist:  [https://forms.gle/w9u2QKcPrztApRedA](https://forms.gle/w9u2QKcPrztApRedA)

## Contributing

We welcome contributions! See the [developer-facing docs](https://burr.dagworks.io/contributing).

## Contributors

*   [Code contributions](https://github.com/apache/burr/graphs/contributors)
*   [Bug hunters/special mentions](https://github.com/apache/burr/graphs/contributors)

**Learn more and contribute on [GitHub](https://github.com/apache/burr)!**