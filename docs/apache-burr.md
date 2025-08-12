# Burr: Build, Observe, and Scale AI Applications with Stateful Workflows

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

Burr is a powerful Python framework that streamlines the development and management of stateful AI applications, offering a clear path for building, debugging, and scaling your projects. [Check out the original repo for more details](https://github.com/apache/burr)!

## Key Features

*   **State Machine Framework:** Build applications as state machines using simple Python functions, making your AI workflows easier to understand and maintain.
*   **Real-time UI:**  Track, monitor, and trace your system's execution in real-time with an intuitive user interface.
*   **Pluggable Persistence:** Save and load application state with a variety of pluggable persisters (e.g., memory, databases).
*   **Framework Agnostic:** Works well with LLMs and integrates seamlessly with your favorite frameworks and libraries.
*   **Extensible:** Easily integrate with LLM observability tools, storage solutions, and custom actions.
*   **Versatile Use Cases:** Build chatbots, agents, simulations, and more.

## Getting Started

### Installation

Install Burr from PyPI:

```bash
pip install "burr[start]"
```

(See the [documentation](https://burr.dagworks.io/getting_started/install/) if you're using Poetry)

### Run the UI

Start the Burr UI server:

```bash
burr
```

The UI provides a real-time view of your application's execution, along with pre-loaded demo data, including a chatbot example.  Remember to set the `OPENAI_API_KEY` environment variable for the chatbot demo to work.

### Example

Run the hello-world counter example:

```bash
git clone https://github.com/dagworks-inc/burr && cd burr/examples/hello-world-counter
python application.py
```

Explore the [getting started guide](https://burr.dagworks.io/getting_started/simple-example/) for more details.

## How Burr Works

Burr empowers you to express your application logic as a state machine, enabling you to manage state, track decisions, and build idempotent workflows.

**Core API:**  A simple, dependency-free Python library to build and manage state machines using Python functions.

**Components:**

1.  **Python Library:** For building and managing state machines.
2.  **UI:**  Provides execution telemetry for introspection and debugging.
3.  **Integrations:**  To persist state, connect to telemetry, and integrate with other systems.

## Use Cases

Burr supports a variety of applications:

*   [GPT-like Chatbot](https://github.com/dagworks-inc/burr/tree/main/examples/multi-modal-chatbot)
*   [Stateful RAG-based Chatbot](https://github.com/dagworks-inc/burr/tree/main/examples/conversational-rag/simple_example)
*   [LLM-based Adventure Game](https://github.com/DAGWorks-Inc/burr/tree/main/examples/llm-adventure-game)
*   [Interactive Email Assistant](https://github.com/DAGWorks-Inc/burr/tree/main/examples/email-assistant)
*   Simulation, ML Training

And many more!

## Why Burr?

Burr simplifies the development of AI applications by providing a robust framework for state management and observability, unlike "all-in-one" libraries, allowing your systems to scale with your needs.  Burr is actively maintained by the Dagworks team.

## Comparison

| Criteria                                          | Burr | Langgraph | temporal | Langchain | Superagent | Hamilton |
| ------------------------------------------------- | :--: | :-------: | :------: | :-------: | :--------: | :------: |
| Explicitly models a state machine                 |  ✅  |    ✅     |    ❌    |    ❌     |     ❌     |    ❌    |
| Framework-agnostic                                |  ✅  |    ✅     |    ✅    |    ✅     |     ❌     |    ✅    |
| Asynchronous event-based orchestration            |  ❌  |    ❌     |    ✅    |    ❌     |     ❌     |    ❌    |
| Built for core web-service logic                  |  ✅  |    ✅     |    ❌    |    ✅     |     ✅     |    ✅    |
| Open-source user-interface for monitoring/tracing |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |
| Works with non-LLM use-cases                      |  ✅  |    ❌     |    ❌    |    ❌     |     ❌     |    ✅    |

## Roadmap

The Burr team is actively working on new features, including:

*   FastAPI integration.
*   Efficiency and usability improvements.
*   Tooling for hosted execution.
*   Additional storage integrations.
*   Burr Cloud (sign up [here](https://forms.gle/w9u2QKcPrztApRedA) for the waitlist).

## Contributing

We welcome contributions!  See the [developer-facing docs](https://burr.dagworks.io/contributing) to get started.

## Contributors

### Code contributions

*   [Elijah ben Izzy](https://github.com/elijahbenizzy)
*   [Stefan Krawczyk](https://github.com/skrawcz)
*   [Joseph Booth](https://github.com/jombooth)
*   [Nandani Thakur](https://github.com/NandaniThakur)
*   [Thierry Jean](https://github.com/zilto)
*   [Hamza Farhan](https://github.com/HamzaFarhan)
*   [Abdul Rafay](https://github.com/proftorch)
*   [Margaret Lange](https://github.com/margaretlange)

### Bug hunters/special mentions

*   [Luke Chadwick](https://github.com/vertis)
*   [Evans](https://github.com/sudoevans)
*   [Sasmitha Manathunga](https://github.com/mmz-001)